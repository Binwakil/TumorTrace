from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tumortrust_vlm.utils import sha256_json

TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+(?:[-'][A-Za-z]+)?|[^\w\s]")


def validate_reporter_record_sets(
    train_records: list[dict], val_records: list[dict]
) -> dict[str, int | str]:
    """Verify paired reporter records and enforce deployment-matched OOF evidence."""

    def index(records: list[dict], expected_split: str) -> dict[str, dict]:
        indexed = {}
        for record in records:
            subject_id = record["subject_id"]
            if subject_id in indexed:
                raise ValueError(f"Duplicate {expected_split} reporter subject: {subject_id}")
            if record.get("report_split") != expected_split:
                raise ValueError(f"Reporter record {subject_id} is not in {expected_split} split")
            if not record.get("subject_provenance", {}).get(
                "excluded_from_checkpoint_training", False
            ):
                raise ValueError(
                    f"Reporter evidence for {subject_id} is not proven out-of-fold"
                )
            indexed[subject_id] = record
        if not indexed:
            raise ValueError(f"Reporter {expected_split} records are empty")
        return indexed

    train = index(train_records, "train")
    validation = index(val_records, "val")
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"Reporter train/validation overlap: {sorted(overlap)[:5]}")
    evidence_dimensions = {
        len(record["evidence_vector"]) for record in (*train.values(), *validation.values())
    }
    visual_dimensions = {
        len(record["visual_tokens"][0])
        for record in (*train.values(), *validation.values())
        if record.get("visual_tokens")
    }
    if len(evidence_dimensions) != 1 or len(visual_dimensions) > 1:
        raise ValueError("Reporter feature dimensions are inconsistent")
    target_schemas = {
        (record.get("report_target_schema"), record.get("finding_target_schema"))
        for record in (*train.values(), *validation.values())
    }
    if len(target_schemas) != 1 or None in next(iter(target_schemas)):
        raise ValueError("Reporter target schemas are missing or inconsistent")
    return {
        "train_subjects": len(train),
        "validation_subjects": len(validation),
        "train_subjects_sha256": sha256_json(sorted(train)),
        "validation_subjects_sha256": sha256_json(sorted(validation)),
        "evidence_dimension": next(iter(evidence_dimensions)),
        "visual_dimension": next(iter(visual_dimensions)) if visual_dimensions else 0,
        "report_target_schema": next(iter(target_schemas))[0],
        "finding_target_schema": next(iter(target_schemas))[1],
    }


class WordTokenizer:
    SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = vocabulary
        self.inverse = {index: token for token, index in vocabulary.items()}
        self.pad_id = vocabulary["<pad>"]
        self.bos_id = vocabulary["<bos>"]
        self.eos_id = vocabulary["<eos>"]
        self.unk_id = vocabulary["<unk>"]

    @classmethod
    def fit(cls, texts: list[str], minimum_frequency: int = 2, maximum_size: int = 16000) -> WordTokenizer:
        counts = Counter(token.lower() for text in texts for token in TOKEN_PATTERN.findall(text))
        tokens = [token for token, count in counts.most_common() if count >= minimum_frequency]
        tokens = tokens[: max(0, maximum_size - len(cls.SPECIAL))]
        return cls({token: index for index, token in enumerate((*cls.SPECIAL, *tokens))})

    def encode(self, text: str, maximum_length: int = 256) -> list[int]:
        tokens = [self.vocabulary.get(token.lower(), self.unk_id) for token in TOKEN_PATTERN.findall(text)]
        return [self.bos_id, *tokens[: maximum_length - 2], self.eos_id]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse.get(index, "<unk>") for index in ids]
        tokens = [token for token in tokens if token not in self.SPECIAL]
        text = " ".join(tokens)
        return re.sub(r"\s+([.,;:!?])", r"\1", text)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocabulary, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.vocabulary)


class ReporterDataset(Dataset):
    def __init__(self, records_path: str | Path, tokenizer: WordTokenizer, maximum_length: int = 256) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        ids = torch.tensor(self.tokenizer.encode(record["report_text"], self.maximum_length), dtype=torch.long)
        visual = record.get("visual_tokens")
        return {
            "input_ids": ids[:-1],
            "labels": ids[1:],
            "evidence": torch.tensor(record["evidence_vector"], dtype=torch.float32),
            "visual_tokens": torch.tensor(visual, dtype=torch.float32) if visual is not None else None,
            "subject_id": record["subject_id"],
        }


def collate_reporter(batch: list[dict], pad_id: int = 0) -> dict:
    maximum = max(len(item["input_ids"]) for item in batch)
    inputs = torch.full((len(batch), maximum), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maximum), -100, dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["input_ids"])
        inputs[index, :length] = item["input_ids"]
        labels[index, :length] = item["labels"]
    visuals = None
    if all(item["visual_tokens"] is not None for item in batch):
        visuals = torch.stack([item["visual_tokens"] for item in batch])
    return {
        "input_ids": inputs,
        "labels": labels,
        "evidence": torch.stack([item["evidence"] for item in batch]),
        "visual_tokens": visuals,
        "subject_id": [item["subject_id"] for item in batch],
    }
