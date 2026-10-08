from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tumortrust_vlm.evaluation.reporting import (
    EVIDENCE_EXTRACTOR_VERSION,
    bleu,
    evidence_consistency,
    extract_evidence_fields,
    report_diversity,
    rouge_l,
    targeted_intervention_accuracy,
)
from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.findings import (
    extract_normalized_findings,
    structured_finding_metrics,
    structured_laterality_metrics,
)
from tumortrust_vlm.reporting.llm_api import (
    SYSTEM_INSTRUCTIONS,
    evidence_prompt,
    extract_response_text,
    sanitize_evidence,
    swap_evidence_field,
)
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

API_ROOT = "https://api.openai.com/v1"


def api_request(
    path: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    retries: int = 5,
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=data,
        method="POST" if payload is not None else "GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code not in {408, 409, 429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise RuntimeError(f"OpenAI API error {error.code}: {body[:500]}") from error
        except urllib.error.URLError:
            if attempt + 1 == retries:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("OpenAI API request exhausted retries")


def generate(
    evidence: dict[str, Any], api_key: str, model: str, maximum_output_tokens: int
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    response = api_request(
        "/responses",
        api_key,
        payload={
            "model": model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": evidence_prompt(evidence),
            "reasoning": {"effort": "none"},
            "max_output_tokens": maximum_output_tokens,
            "store": False,
        },
    )
    latency = time.perf_counter() - started
    if response.get("status") != "completed":
        raise RuntimeError(f"Incomplete response status: {response.get('status')}")
    usage = response.get("usage", {})
    metadata = {
        "resolved_model": response.get("model"),
        "latency_seconds": latency,
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return extract_response_text(response), metadata


def prediction_row(record: dict[str, Any], text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    reference = record["report_text"]
    reference_findings = record.get("normalized_findings") or extract_normalized_findings(reference)
    return {
        "subject_id": record["subject_id"],
        "report_split": record.get("report_split"),
        "reference": reference,
        "prediction": text,
        "bleu_1": bleu(text, reference, 1),
        "bleu_4": bleu(text, reference, 4),
        "rouge_l": rouge_l(text, reference),
        "predicted_normalized_findings": extract_normalized_findings(text),
        "reference_normalized_findings": reference_findings,
        **evidence_consistency(text, record["evidence"]),
        **metadata,
    }


def total_cost(rows: list[dict[str, Any]], input_price: float, output_price: float) -> float:
    return (
        sum(row.get("input_tokens", 0) for row in rows) * input_price
        + sum(row.get("output_tokens", 0) for row in rows) * output_price
    ) / 1_000_000


def summarize(
    rows: list[dict[str, Any]], model: str, input_price: float, output_price: float
) -> dict[str, Any]:
    texts = [row["prediction"] for row in rows]
    references = [row["reference_normalized_findings"] for row in rows]
    elapsed = sum(row["latency_seconds"] for row in rows)
    return {
        "subjects": len(rows),
        "model_requested": model,
        "models_resolved": sorted({row["resolved_model"] for row in rows}),
        "mode": "structured_evidence_only_zero_shot",
        "fine_tuned": False,
        "evidence_extractor_version": EVIDENCE_EXTRACTOR_VERSION,
        "locked_final_test_opened": False,
        "bleu_1": float(np.mean([row["bleu_1"] for row in rows])),
        "bleu_4": float(np.mean([row["bleu_4"] for row in rows])),
        "rouge_l": float(np.mean([row["rouge_l"] for row in rows])),
        "structured_field_recall": float(
            np.mean([row["structured_field_recall"] for row in rows])
        ),
        "structured_field_unsupported_rate": float(
            np.mean([row["structured_field_unsupported_rate"] for row in rows])
        ),
        "structured_field_contradiction_rate": float(
            np.mean([row["structured_field_contradiction_rate"] for row in rows])
        ),
        "broad_unsupported_claim_rate": None,
        "broad_unsupported_claim_status": "requires_radiologist_review",
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "sequential_latency_seconds": elapsed,
        "reports_per_minute_sequential": 60 * len(rows) / elapsed if elapsed else None,
        "estimated_api_cost_usd": total_cost(rows, input_price, output_price),
        **report_diversity(texts),
        **structured_finding_metrics(texts, references),
        **structured_laterality_metrics(texts, references),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a deidentified evidence-only reporter through the Responses API."
    )
    parser.add_argument("--protocol", default="configs/llm_evidence_reporter.yaml")
    parser.add_argument("--output", default="outputs/gpt_5_6_sol_evidence_only")
    parser.add_argument("--include-interventions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = ROOT / args.protocol
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    records_path = ROOT / protocol["data"]["records"]
    records = json.loads(records_path.read_text(encoding="utf-8"))
    expected = int(protocol["data"]["expected_subjects"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} development records, found {len(records)}")
    if any(record.get("report_split") != protocol["data"]["expected_split"] for record in records):
        raise ValueError("LLM reporter records do not match the frozen development split")
    if any(record.get("report_split") == "test" for record in records):
        raise SystemExit("Final report test is excluded from the LLM reporter protocol")
    for record in records:
        sanitize_evidence(record["evidence"])

    intervention = protocol["evaluation"]["targeted_intervention"]
    complete_batches = len(records) // int(intervention["batch_size"])
    planned_interventions = complete_batches * int(intervention["batch_size"]) * len(
        intervention["fields"]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "validated_no_api_calls",
                    "subjects": len(records),
                    "model": protocol["model"]["api_model_id"],
                    "base_calls": len(records),
                    "intervention_calls": planned_interventions
                    if args.include_interventions
                    else 0,
                    "protocol_sha256": sha256_file(protocol_path),
                    "records_sha256": sha256_file(records_path),
                    "external_payload_contains_identifiers": False,
                    "locked_final_test_opened": False,
                },
                indent=2,
            )
        )
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required; no API calls were made")
    model = protocol["model"]["api_model_id"]
    model_record = api_request(f"/models/{model}", api_key)
    if model_record.get("id") != model:
        raise RuntimeError(f"Requested model is not available: {model}")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "val_predictions_unconstrained.json"
    rows = json.loads(prediction_path.read_text()) if prediction_path.exists() else []
    records_by_subject = {record["subject_id"]: record for record in records}
    for row in rows:
        row.update(
            evidence_consistency(
                row["prediction"], records_by_subject[row["subject_id"]]["evidence"]
            )
        )
    if rows:
        atomic_json_dump(rows, prediction_path)
    completed = {row["subject_id"] for row in rows}
    for record in records:
        if record["subject_id"] in completed:
            continue
        text, metadata = generate(
            record["evidence"],
            api_key,
            model,
            int(protocol["model"]["maximum_output_tokens"]),
        )
        rows.append(prediction_row(record, text, metadata))
        atomic_json_dump(rows, prediction_path)

    pricing = protocol["pricing_snapshot"]
    summary = summarize(
        rows,
        model,
        float(pricing["input_usd_per_million_tokens"]),
        float(pricing["output_usd_per_million_tokens"]),
    )
    summary["protocol_sha256"] = sha256_file(protocol_path)
    summary["records_sha256"] = sha256_file(records_path)
    atomic_json_dump(summary, prediction_path.with_suffix(".summary.json"))

    if args.include_interventions:
        fields = intervention["fields"]
        batch_size = int(intervention["batch_size"])
        baseline = {row["subject_id"]: row for row in rows}
        probe_path = output / "val_input_probes.json"
        probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
        completed_probes = {(row["subject_id"], row["field"]) for row in probes}
        jobs = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            if len(batch) <= 1:
                continue
            donors = [batch[-1], *batch[:-1]]
            for record, donor in zip(batch, donors, strict=True):
                before = baseline[record["subject_id"]]["prediction"]
                expected = extract_evidence_fields(
                    render_findings(EvidenceCard.from_dict(donor["evidence"]))
                )
                for field in fields:
                    key = (record["subject_id"], field)
                    if key in completed_probes:
                        continue
                    changed = swap_evidence_field(record["evidence"], donor["evidence"], field)
                    jobs.append(
                        {
                            "subject_id": record["subject_id"],
                            "field": field,
                            "before": before,
                            "expected": expected,
                            "changed": changed,
                        }
                    )

        def execute_probe(job: dict[str, Any]) -> dict[str, Any]:
            after, metadata = generate(
                job["changed"],
                api_key,
                model,
                int(protocol["model"]["maximum_output_tokens"]),
            )
            return {
                "subject_id": job["subject_id"],
                "field": job["field"],
                "before": job["before"],
                "after": after,
                "expected": job["expected"],
                **metadata,
            }

        workers = int(intervention.get("workers", 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute_probe, job) for job in jobs]
            for future in as_completed(futures):
                probes.append(future.result())
                atomic_json_dump(probes, probe_path)

        field_scores = {}
        for field in fields:
            selected = [row for row in probes if row["field"] == field]
            score = targeted_intervention_accuracy(
                [extract_evidence_fields(row["before"]) for row in selected],
                [extract_evidence_fields(row["after"]) for row in selected],
                [row["expected"] for row in selected],
                field,
            )
            field_scores[field] = None if np.isnan(score) else score
        available = [score for score in field_scores.values() if score is not None]
        probe_summary = {
            "subjects": len(records),
            "concurrent_workers": workers,
            "intervention_generations": len(probes),
            "targeted_evidence_intervention_accuracy": float(np.mean(available))
            if available
            else None,
            "targeted_evidence_intervention_by_field": field_scores,
            "input_tokens": sum(row["input_tokens"] for row in probes),
            "output_tokens": sum(row["output_tokens"] for row in probes),
            "sequential_latency_seconds": sum(row["latency_seconds"] for row in probes),
            "estimated_api_cost_usd": total_cost(
                probes,
                float(pricing["input_usd_per_million_tokens"]),
                float(pricing["output_usd_per_million_tokens"]),
            ),
        }
        atomic_json_dump(probe_summary, probe_path.with_suffix(".summary.json"))
        summary["targeted_evidence_intervention_accuracy"] = probe_summary[
            "targeted_evidence_intervention_accuracy"
        ]
        atomic_json_dump(summary, prediction_path.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
