from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

REPORT_PROMPT = (
    "Generate a concise brain MRI findings report grounded only in the supplied "
    "visual and structured evidence. Do not provide treatment advice.\nFindings:"
)


class VicunaReporterDataset(Dataset):
    def __init__(
        self,
        records_path: str | Path,
        tokenizer,
        maximum_length: int = 256,
    ) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        prompt_ids = self.tokenizer.encode(REPORT_PROMPT, add_special_tokens=True)
        target_ids = self.tokenizer.encode(" " + record["report_text"], add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        target_ids = [*target_ids, eos_id]
        available = max(1, self.maximum_length - len(prompt_ids))
        target_ids = target_ids[:available]
        target_ids[-1] = eos_id
        input_ids = torch.tensor([*prompt_ids, *target_ids], dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "evidence": torch.tensor(record["evidence_vector"], dtype=torch.float32),
            "visual_tokens": torch.tensor(record["visual_tokens"], dtype=torch.float32),
            "subject_id": record["subject_id"],
        }


def collate_vicuna(batch: list[dict], pad_token_id: int) -> dict:
    maximum = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), maximum), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), maximum), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), maximum), dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[index, :length] = item["input_ids"]
        labels[index, :length] = item["labels"]
        attention_mask[index, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "evidence": torch.stack([item["evidence"] for item in batch]),
        "visual_tokens": torch.stack([item["visual_tokens"] for item in batch]),
        "subject_id": [item["subject_id"] for item in batch],
    }


def encode_prompt(tokenizer, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(tokenizer.encode(REPORT_PROMPT, add_special_tokens=True), dtype=torch.long)
    input_ids = ids[None].repeat(batch_size, 1)
    return input_ids, torch.ones_like(input_ids)
