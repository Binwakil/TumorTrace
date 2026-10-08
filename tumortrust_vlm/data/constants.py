from __future__ import annotations

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
COHORTS = ("GLI", "MEN", "MET")
COHORT_TO_INDEX = {cohort: index for index, cohort in enumerate(COHORTS)}

COHORT_DIRECTORIES = {
    "GLI": "2023GLI",
    "MEN": "2023MEN",
    "MET": "2023MET",
}
LABELED_BRANCHES = {
    "GLI": ("TrainingData",),
    "MEN": ("TrainingData",),
    "MET": ("TrainingData", "MET--TrainingData2"),
}
UNLABELED_BRANCHES = {
    "GLI": ("ValidationData",),
    "MEN": ("ValidationData",),
    "MET": ("ValidationData",),
}
CORRECTION_BRANCH = {"MEN": "BraTS-MEN-TRAIN-FIX-V4"}
REPORT_DIRECTORIES = {"GLI": "BraTS_GLI", "MEN": "BraTS_MEN", "MET": "BraTS_MET"}

REGION_LABELS = {
    "WT": (1, 2, 3, 4),
    "TC": (1, 3, 4),
    "ET": (3, 4),
    "SNFH": (2,),
}

