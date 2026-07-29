"""Shared path configuration for ADNI preprocessing and partitioning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


DEFAULT_CONFIG_PATH = Path(__file__).with_name("adni_paths.json")


def load_adni_paths(
    config_path: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """Load the private ADNI path config and derive commonly used paths."""
    path = Path(
        config_path
        or os.environ.get("ADNI_PATH_CONFIG")
        or DEFAULT_CONFIG_PATH
    ).expanduser()

    if not path.is_file():
        example = Path(__file__).with_name("adni_paths_example.json")
        raise FileNotFoundError(
            f"ADNI path config not found: {path}\n"
            f"Copy {example} to {DEFAULT_CONFIG_PATH} and update its paths, "
            "or set ADNI_PATH_CONFIG."
        )

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = {
        "adni_root",
        "downloaded_date",
        "mri_reference_atlas",
        "mri_rename_source_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    root = Path(config["adni_root"]).expanduser()
    downloaded_date = str(config["downloaded_date"])

    config.update(
        {
            "config_path": path,
            "adni_root": root,
            "raw_dir": root / "raw",
            "processed_dir": root / "processed",
            "partitioned_dir": root / "partitioned",
            "diagnosis_file": root / "raw" / "diagnosis" / f"DXSUM_{downloaded_date}.csv",
            "mri_data_dir": root / "raw" / "MRI" / "data",
            "mri_processed_dir": root / "processed" / "MRI",
            "mri_reference_atlas": Path(config["mri_reference_atlas"]).expanduser(),
            "mri_rename_source_dir": Path(config["mri_rename_source_dir"]).expanduser(),
        }
    )
    return config
