"""Finite-value instrumented probe for the separate_encoder_matched_control NaN crash.

Loads the EXACT matched config used in the failed overnight run, builds the real model and a
real training batch, and checks finiteness at each stage of forward -> loss -> backward:
    input image -> segmentation output -> segmentation_latent -> classification_latent
    -> classification logits -> loss components -> gradients (per-backbone)

This is diagnostic only. It does not change LR, loss weighting, or architecture -- if the
instability is a pure fp16/autocast numerical issue (the same class of bug already found and
fixed in A1_background_only), that is fixable without touching the preregistered single-variable
comparison. If it is not, this script stops short of proposing a fix.
"""
import sys

import torch

sys.path.insert(0, ".")
from torch.utils.data import DataLoader

from tumortrust_vlm.config import load_config
from tumortrust_vlm.engine import build_dataset, build_model, collate_training
from tumortrust_vlm.models.losses import MultiTaskObjective

CONFIG = "artifacts/private/overnight/separate_encoder_matched_control.yaml"
DEVICE = "cuda:1"
N_STEPS = 12


def finite(name, t):
    if t is None:
        print(f"    {name}: None"); return True
    ok = torch.isfinite(t).all().item()
    if ok:
        print(f"    {name}: finite  (min={t.float().min().item():.4g} max={t.float().max().item():.4g})")
    else:
        n_nan = torch.isnan(t).sum().item(); n_inf = torch.isinf(t).sum().item()
        print(f"    {name}: *** NOT FINITE *** nan={n_nan} inf={n_inf} / {t.numel()}")
    return ok


def main():
    config = load_config(CONFIG)
    dev = torch.device(DEVICE)
    torch.manual_seed(int(config["project"]["seed"]))

    print(f"[build] model architecture = {config['model']['architecture']}, amp = {config['training'].get('amp')}")
    model = build_model(config).to(dev)
    objective = MultiTaskObjective(
        mode=config["model"].get("joint_weighting", "fixed"),
        lambda_cls=float(config["model"].get("lambda_cls", 0.2)),
    ).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()),
                            lr=float(config["training"]["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    train_ds = build_dataset(config, "train", True)
    dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_training)
    it = iter(dl)

    model.train(); objective.train()
    for step in range(1, N_STEPS + 1):
        batch = next(it)
        image = batch["image"].to(dev); label = batch["label"].to(dev)
        class_label = batch["class_label"].to(dev); presence = batch["modality_presence"].to(dev)

        print(f"\n=== step {step} ===")
        finite("input image", image)

        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(image, presence)
            finite("segmentation output", outputs["segmentation"])
            finite("segmentation_latent", outputs["latent"])
            finite("classification_latent", outputs["classification_latent"])
            finite("classification logits", outputs["classification"])
            losses = objective(outputs, label, class_label)
            finite("segmentation_loss", losses["segmentation_loss"])
            finite("classification_loss", losses["classification_loss"])
            finite("total loss", losses["loss"])

        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["loss"]).backward()

        # gradient finiteness, split by which backbone a param belongs to
        seg_grad_ok = cls_grad_ok = True
        seg_grad_max = cls_grad_max = 0.0
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            ok = torch.isfinite(p.grad).all().item()
            gmax = p.grad.abs().max().item() if ok else float("nan")
            if "classification_backbone" in n:
                cls_grad_ok &= ok; cls_grad_max = max(cls_grad_max, gmax) if ok else float("nan")
            elif n.startswith("backbone") or "backbone." in n and "classification" not in n:
                seg_grad_ok &= ok; seg_grad_max = max(seg_grad_max, gmax) if ok else float("nan")
        print(f"    shared/backbone grads finite: {seg_grad_ok}  (max |grad| ~{seg_grad_max:.4g})")
        print(f"    classification_backbone grads finite: {cls_grad_ok}  (max |grad| ~{cls_grad_max:.4g})")

        scaler.step(opt); scaler.update()

        if not (torch.isfinite(losses["loss"]).all() and seg_grad_ok and cls_grad_ok):
            print(f"\n!!! First non-finite state detected at step {step} !!!")
            break
    else:
        print(f"\nAll {N_STEPS} steps stayed finite -- instability (if real) develops later than this probe covers.")


if __name__ == "__main__":
    main()
