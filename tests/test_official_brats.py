import nibabel as nib
import numpy as np
from brats_evaluation import config_path, evaluate_single_exam
from panoptica import Panoptica_Evaluator


def test_official_brats_evaluator_identity(tmp_path):
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 3
    path = tmp_path / "case.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), path)
    evaluator = Panoptica_Evaluator.load_from_config(str(config_path("gli")))
    result = evaluate_single_exam(str(path), str(path), "case", evaluator)
    assert isinstance(result, dict)
    assert result

