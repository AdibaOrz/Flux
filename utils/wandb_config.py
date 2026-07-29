"""Load Weights & Biases entity and project names from wandb_config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "wandb_config.json"


def load_wandb_config(
    config_path: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    path = Path(
        config_path or os.environ.get("WANDB_CONFIG") or DEFAULT_CONFIG_PATH
    ).expanduser()

    if not path.is_file():
        example = REPO_ROOT / "wandb_config_example.json"
        raise FileNotFoundError(
            f"W&B config not found: {path}\n"
            f"Copy {example} to {DEFAULT_CONFIG_PATH} and update entity/projects, "
            "or set WANDB_CONFIG."
        )

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = {"entity", "wearable_project", "adni_project"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    config["config_path"] = path
    return config


def apply_wandb_cli_defaults(args, pipeline: str = "wearable") -> None:
    """Fill unset W&B CLI fields from wandb_config.json."""
    project_attr = "project" if pipeline == "wearable" else "wandb_project"
    if (
        getattr(args, "wandb_entity", None) is not None
        and getattr(args, project_attr, None) is not None
    ):
        return

    cfg = load_wandb_config()

    if getattr(args, "wandb_entity", None) is None:
        args.wandb_entity = cfg["entity"]

    if getattr(args, project_attr, None) is None:
        project_key = "wearable_project" if pipeline == "wearable" else "adni_project"
        setattr(args, project_attr, cfg[project_key])
