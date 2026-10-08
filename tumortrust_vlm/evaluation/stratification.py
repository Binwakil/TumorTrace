from __future__ import annotations

from collections import defaultdict

import numpy as np

from tumortrust_vlm.evaluation.metrics import aggregate_case_metrics


def _summarize(groups: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        name: {"subjects": len(cases), **aggregate_case_metrics(cases)}
        for name, cases in sorted(groups.items())
        if cases
    }


def stratified_case_metrics(cases: list[dict]) -> dict[str, dict]:
    if not cases:
        return {}
    cohort: dict[str, list[dict]] = defaultdict(list)
    source: dict[str, list[dict]] = defaultdict(list)
    lesion_count: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        cohort[str(case["cohort"])].append(case)
        source[f"{case['cohort']}:{case['source_branch']}"].append(case)
        count = int(case.get("target_component_count_WT", 0))
        lesion_count["none" if count == 0 else "single" if count == 1 else "multifocal"].append(case)

    values = np.asarray([case["target_volume_ml_WT"] for case in cases], dtype=float)
    boundaries = np.quantile(values, (0.25, 0.50, 0.75))
    burden: dict[str, list[dict]] = defaultdict(list)
    for case, value in zip(cases, values, strict=True):
        quartile = int(np.searchsorted(boundaries, value, side="right")) + 1
        burden[f"Q{quartile}"].append(case)
    return {
        "cohort": _summarize(cohort),
        "source": _summarize(source),
        "burden_quartile": _summarize(burden),
        "lesion_count": _summarize(lesion_count),
        "burden_quartile_boundaries_ml": boundaries.tolist(),
    }
