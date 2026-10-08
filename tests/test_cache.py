from pathlib import Path

from tumortrust_vlm.data.cache import cache_summary_path, preprocessing_key


def test_cache_keys_and_summaries_are_preprocessing_specific(tmp_path: Path) -> None:
    robust = {
        "data": {
            "target_spacing": [1, 1, 1],
            "normalization": "robust_nonzero",
            "brain_crop_margin": 8,
            "canonicalize": True,
        }
    }
    zscore = {
        "data": {
            **robust["data"],
            "normalization": "zscore_nonzero",
        }
    }

    robust_key = preprocessing_key(robust)
    zscore_key = preprocessing_key(zscore)
    assert robust_key != zscore_key
    assert cache_summary_path(tmp_path / "cache", robust_key) == (
        tmp_path / "cache_summaries" / f"{robust_key}.json"
    )
    assert cache_summary_path(tmp_path / "cache", robust_key) != cache_summary_path(
        tmp_path / "cache", zscore_key
    )
