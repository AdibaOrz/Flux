
from __future__ import annotations

from adni_pipeline.configs.adni_config import ADNIConfig, build_default_config

from collections import Counter
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch



@dataclass(frozen=True)
class ModalityBundle:
    """Holds tabular modality array and PTID->row map."""
    data: np.ndarray
    pid_to_idx: Dict[str, int]


def _resolve_paths(adni_root: str, partition_name: str) -> Dict[str, Path]:
    from adni_pipeline.configs.adni_config import ADNI_PARTITION_MAP
    dir_name = ADNI_PARTITION_MAP.get(partition_name, partition_name)
    root = Path(adni_root).expanduser().resolve()
    split_dir = root / "partitioned" / dir_name

    partition_labels = split_dir / "labels.csv"
    if partition_labels.exists():
        labels_csv = partition_labels
    else:
        labels_csv = root / "processed" / "labels.csv"

    image_root = root / "processed" / "MRI"
    # print(f"  [_resolve_paths] partition_name='{partition_name}' -> dir='{dir_name}'")
    # print(f"    split_dir : {split_dir}")
    # print(f"    labels_csv: {labels_csv}")
    return {"root": root, "split_dir": split_dir, "labels_csv": labels_csv, "image_root": image_root}


def load_split_metadata(adni_root: str, partition_name: str = "stratified") -> Dict:
    paths = _resolve_paths(adni_root, partition_name)
    metadata_path = paths["split_dir"] / "split_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_silos(adni_root: str, partition_name: str = "stratified") -> List[str]:
    split_meta = load_split_metadata(adni_root, partition_name)
    return sorted(split_meta["data"]["federated"].keys())


def get_split_ptids(
    adni_root: str,
    silo_key: str,
    split_name: str,
    partition_name: str = "stratified",
) -> List[str]:
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split_name: {split_name}")
    split_meta = load_split_metadata(adni_root, partition_name)
    federated = split_meta["data"]["federated"]
    if silo_key not in federated:
        raise KeyError(f"Unknown silo_key '{silo_key}'. Available: {sorted(federated.keys())}")
    pids = federated[silo_key].get(split_name, [])
    if not pids:
        raise ValueError(f"No PTIDs found for {silo_key}/{split_name}")
    return [str(pid) for pid in pids]


def _load_labels(labels_csv: Path, cfg: ADNIConfig) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    missing_cols = [c for c in cfg.required_label_columns if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"labels.csv missing required columns: {missing_cols}")
    return labels_df


def _load_tabular_bundle(data_npy_path: str, pid_to_idx_json_path: str) -> ModalityBundle:
    # Load fully into memory (writable) so __getitem__ row slices require no per-sample copy.
    data = np.load(data_npy_path)
    with open(pid_to_idx_json_path, "r", encoding="utf-8") as f:
        pid_to_idx = json.load(f)
    return ModalityBundle(data=data, pid_to_idx=pid_to_idx)


def _modality_mask_key(row: pd.Series, modalities: Tuple[str, ...], flag_columns: Dict[str, str]) -> str:
    """Build a short string key like 'IGCB' from a label row's modality flags."""
    abbreviations = {"image": "I", "genomic": "G", "clinical": "C", "biospecimen": "B"}
    return "".join(abbreviations[m] for m in modalities if int(row[flag_columns[m]]) == 1) or "none"


def modality_mask_counts(
    adni_root: str,
    partition_name: str = "stratified",
    split_name: str = "train",
    cfg: ADNIConfig | None = None,
) -> Dict[str, Dict[str, int]]:
    """Count samples per modality-mask combination for each silo.

    Returns:
        {silo_key: {mask_string: count, ...}, ...}
        e.g. {"silo_1": {"IGCB": 120, "GCB": 15, "CB": 3, ...}, ...}
    """
    cfg = cfg or build_default_config()
    paths = _resolve_paths(adni_root, partition_name)
    labels_df = _load_labels(paths["labels_csv"], cfg)
    labels_indexed = labels_df.set_index(cfg.id_column)

    silos = list_silos(adni_root, partition_name)
    result: Dict[str, Dict[str, int]] = {}
    for silo_key in silos:
        ptids = get_split_ptids(adni_root, silo_key, split_name, partition_name)
        silo_rows = labels_indexed.loc[[pid for pid in ptids if pid in labels_indexed.index]]
        masks = silo_rows.apply(
            lambda row: _modality_mask_key(row, cfg.modalities, cfg.modality_flag_columns), axis=1
        )
        result[silo_key] = dict(Counter(masks).most_common())
    return result


class ADNINativeDataset(Dataset):
    """Dataset yielding (data_dict_by_modality, label, observed_vector)."""

    def __init__(
        self,
        adni_root: str,
        silo_key: str,
        split_name: str,
        partition_name: str = "stratified",
        cfg: ADNIConfig | None = None,
    ):
        if split_name not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split_name: {split_name}")

        self.cfg = cfg or build_default_config()
        self.modalities: Tuple[str, ...] = tuple(self.cfg.modalities)
        self.tabular_modalities: Tuple[str, ...] = tuple(self.cfg.tabular_modalities)
        self.encoder_input_shapes: Dict[str, List[int]] = self.cfg.encoder_input_shapes
        self.silo_key = silo_key
        self.split_name = split_name

        paths = _resolve_paths(adni_root, partition_name)
        self.image_root = paths["image_root"]
        self._split_dir = paths["split_dir"]
        labels_df = _load_labels(paths["labels_csv"], self.cfg)

        self.ptids = get_split_ptids(adni_root, silo_key, split_name, partition_name)
        split_rows = labels_df.set_index(self.cfg.id_column)
        missing = [pid for pid in self.ptids if pid not in split_rows.index]
        if missing:
            raise ValueError(f"{len(missing)} PTIDs missing in labels.csv. Examples: {missing[:5]}")
        split_rows = split_rows.loc[self.ptids].reset_index()

        self.labels = torch.tensor(
            split_rows[self.cfg.label_column].astype(int).values, dtype=torch.long
        )

        observed_cols = [self.cfg.modality_flag_columns[m] for m in self.modalities]
        self.observed = torch.tensor(split_rows[observed_cols].astype(int).values, dtype=torch.bool)

        self.tabular_bundles: Dict[str, ModalityBundle] = {}
        for modality in self.tabular_modalities:
            base = self._split_dir / "federated" / silo_key / modality
            npy_path = base / f"{split_name}_data.npy"
            pid_json = base / f"{split_name}_pid_to_idx.json"
            if npy_path.exists() and pid_json.exists():
                self.tabular_bundles[modality] = _load_tabular_bundle(str(npy_path), str(pid_json))
            else:
                self.tabular_bundles[modality] = ModalityBundle(data=None, pid_to_idx={})

        # Pre-compute row indices for tabular modalities for O(1) getitem.
        self.row_indices: Dict[str, np.ndarray] = {}
        for modality in self.tabular_modalities:
            pid_map = self.tabular_bundles[modality].pid_to_idx
            mod_idx = self.modalities.index(modality)
            indices = np.full(len(self.ptids), -1, dtype=np.int64)
            for i, pid in enumerate(self.ptids):
                if self.observed[i, mod_idx]:
                    if pid not in pid_map:
                        raise ValueError(
                            f"PTID '{pid}' marked observed for {modality}, but missing from pid_to_idx map."
                        )
                    indices[i] = int(pid_map[pid])
            self.row_indices[modality] = indices

    def __len__(self) -> int:
        return len(self.ptids)

    def __getitem__(self, idx: int):
        pid = self.ptids[idx]
        label = self.labels[idx]
        observed_vec = self.observed[idx]

        data_dict = {}
        for mod_i, modality in enumerate(self.modalities):
            if not bool(observed_vec[mod_i]):
                shape = tuple(self.encoder_input_shapes[modality])
                data_dict[modality] = torch.zeros(shape, dtype=torch.float32)
                continue

            if modality == "image":
                data_dict[modality] = self._load_image(pid)
            else:
                row_idx = self.row_indices[modality][idx]
                arr = self.tabular_bundles[modality].data[row_idx]
                data_dict[modality] = torch.from_numpy(arr).to(dtype=torch.float32)

        return data_dict, label, observed_vec

    @lru_cache(maxsize=256)
    def _load_image(self, pid: str) -> torch.Tensor:
        image_path = self.image_root / pid / f"{pid}.pt"
        if not image_path.exists():
            raise FileNotFoundError(f"Image tensor not found for PTID '{pid}': {image_path}")
        image_tensor = torch.load(image_path, map_location="cpu", weights_only=True)
        return image_tensor.to(dtype=torch.float32)

