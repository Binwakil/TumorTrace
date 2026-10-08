# TumorTrust-VLM

TumorTrust is a trust-audited, segmentation-first framework for four-sequence 3D brain-tumor MRI
across glioma (GLI), meningioma (MEN), and metastasis (MET). Rather than optimizing one headline
metric, it treats classification, multitask learning, calibration, uncertainty, referral, and
learned report generation as auditable candidates: each is retained, qualified, or rejected against
prespecified development gates before a single, one-shot evaluation on a held-out lockbox.

## What the framework does

- **Segmentation anchor.** A standard nnU-Net (ResEnc-M) is the retained perception core; a matched
  single-task SegResNet is the controlled comparator. No new segmentation architecture is proposed.
- **Deterministic evidence engine.** Predicted masks are converted into whole-tumor/tumor-core/
  enhancing-tumor/peritumoral-signal volumes, development-fitted conformal intervals, laterality,
  and component counts — all traceable back to a specific mask, never invented by a language model.
- **Auxiliary tumor-family classification, audited for shortcuts.** A classifier trained on
  background-only image content (tumor voxels removed) still discriminates cohort well above
  chance, and a linear probe decodes acquisition source better than the diagnostic label itself.
  Classification is retained only as an explicitly source-limited auxiliary output, not a headline
  diagnostic claim.
- **Multitask learning, tested and rejected.** Every joint segmentation/classification strategy
  tested (fixed-weight, uncertainty-weighted, gradient-balanced/PCGrad, separate-encoder) regressed
  segmentation Dice in every cohort relative to the single-task baseline.
- **Calibration, uncertainty, and referral, tested and disabled.** Predictive entropy, Monte Carlo
  dropout, and deep ensembles all failed to rank true segmentation failures above chance-level
  detection under the prespecified gates. No referral threshold is exposed.
- **Evidence-conditioned reporting.** A deterministic renderer copies evidence-card values into
  templated findings text with exact traceability and zero unsupported/contradictory fields. Two
  LoRA-fine-tuned 7B vision-language decoder families were evaluated as learned alternatives under
  text-only, evidence-only, and visual+evidence conditions and did not pass the prespecified
  factual-grounding and intervention gates; the deterministic renderer remains the retained
  reporting path.

## Held-out lockbox headline results

On an independently held-out, one-shot evaluation (never touched during development):

| Component | Result |
|---|---|
| nnU-Net (retained segmentation anchor) | macro Dice 0.8457 (95% CI 0.8276–0.8617) |
| vs. matched single-task SegResNet | +0.0433 macro Dice (Holm-adjusted p=0.0050) |
| Auxiliary tumor-family classifier | balanced accuracy 0.8538 — source-limited, not a diagnostic claim |
| Deterministic report renderer | structured-field recall 0.9953, zero unsupported/contradictory fields |
| Learned 7B reporters (Vicuna-7B, LLaVA-Med/Mistral-7B) | both rejected — failed factual-grounding and evidence-intervention gates |

## Repository layout

- `tumortrust_vlm/` — the installable package: data/preprocessing, models, training engine,
  evaluation (segmentation, classification, volumetry, calibration, trust), and the evidence/
  reporting pipeline.
- `scripts/` — CLIs for inventory/split construction, training, evaluation, ablations, and
  figure/table generation.
- `tests/` — the automated test suite.
- `configs/` — training/ablation configuration templates.
- `checkpoints/` — the three retained model checkpoints (Git LFS; see `checkpoints/README.md`
  for provenance hashes). Every other experiment run during development — ablations, LODO/LOSO
  folds, out-of-fold reporter folds, and the rejected learned-reporter decoders — is not included.

## Data and reproducibility

This repository does not include raw MRI, radiology reports, or any per-subject prediction or
evidence output — only the three retained model checkpoints and the code. The underlying cohorts
are the BraTS-organized adult glioma, meningioma, and metastasis collections; access to those
source volumes follows their own data-use terms. Reproducing a training run requires pointing the
configuration at your own local copy of that inventory and generating your own split manifest —
nothing subject-level is published here or reused from ours.

## Quick start

```bash
git lfs install
git clone https://github.com/Binwakil/TumorTrust-VLM.git
conda env create -f environment.yml
conda activate <env-name>
python -m pytest -q
```

Training and evaluation entry points are under `scripts/`; each accepts an explicit configuration
file (see `configs/`) rather than hardcoding a dataset path, so a fresh checkout requires no changes
beyond pointing those configs at your own data. To run inference with a retained checkpoint directly,
pass its path under `checkpoints/` to the corresponding `scripts/evaluate*.py` entry point.

## Safety and scope

- Inputs are T1, T1c, T2, and FLAIR MRI from already tumor-positive cases; this is not a
  detection/screening system.
- Ground-truth masks are used only to supervise training — they never select the inference crop,
  choose slices, or enter the deployed model.
- Classification is limited to GLI/MEN/MET family among tumor-positive cases and is explicitly
  demoted to an auxiliary, source-limited output — not a validated diagnostic claim.
- Reported volumetry is measured burden, not a clinical-severity or prognostic statement.
- No referral or abstention decision is exposed: the tested uncertainty signals failed their
  prespecified reliability gates.
- No output from this framework — segmentation, classification, volumetry, or generated text — has
  undergone radiologist review or constitutes a clinical recommendation.

## Status

A manuscript describing the complete audit and its one-shot locked-test results is in preparation.
This repository will be updated with a citation once it is available.

## License

Released under the Apache License 2.0 — see [`LICENSE`](LICENSE).
