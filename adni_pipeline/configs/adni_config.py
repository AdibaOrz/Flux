"""ADNI pipeline configuration."""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field



@dataclass(frozen=True)
class ADNIConfig:
    dataset_name: str = "adni"
    modalities: Tuple[str, ...] = ("image", "genomic", "clinical", "biospecimen")
    modality_flag_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "image": "Image",
            "genomic": "Genomic",
            "clinical": "Clinical",
            "biospecimen": "Biospecimen",
        }
    )
    label_column: str = "label"
    id_column: str = "PTID"
    required_label_columns: Tuple[str, ...] = (
        "PTID",
        "label",
        "Image",
        "Genomic",
        "Clinical",
        "Biospecimen",
    )
    tabular_modalities: Tuple[str, ...] = ("genomic", "clinical", "biospecimen")
    # Used for zero placeholders when a modality is absent.
    encoder_input_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "image": [30, 90, 90],
            "genomic": [317858],
            "clinical": [2237],
            "biospecimen": [151],
        }
    )


def build_default_config() -> ADNIConfig:
    """Returns the default ADNI configuration."""
    return ADNIConfig()


ADNI_DATASET_CFG = {
    "dataset_name": "adni",
    "modalities": ["image", "genomic", "clinical", "biospecimen"],
    "num_classes": 3,
    "encoder_configs": {
        "image": {
            "type": "3dcnn_patch",
            "input_shape": (30, 90, 90),
            "hidden_dim": 128,
            "num_patches": 16,
        },
        "genomic": {
            "type": "patch",
            "input_shape": (317858,),
            "hidden_dim": 128,
            "num_patches": 16,
        },
        "clinical": {
            "type": "patch",
            "input_shape": (2237,),
            "hidden_dim": 128,
            "num_patches": 16,
        },
        "biospecimen": {
            "type": "patch",
            "input_shape": (151,),
            "hidden_dim": 128,
            "num_patches": 16,
        },
    },
    "enc_out_dim": 128,
    "num_patches": 16,
    "fusion_configs": {
        "hid_dim": 192,
        "num_heads": 4,
    },
    "classifier_configs": {
        "hid_dim": 256,
        "num_hid_layers": 2,
    },

}


# ======================================================================= #
# Partition (silo split) configuration                                    #
# ======================================================================= #

ADNI_PARTITION_MAP = {
    "stratified":      "stratified_5silos",
    "modality_unique": "modality_unique_5silos",
    "med_intra_0p0":   "stratified_5silos_sim_avail_med_intra0p0",
    "low_intra_0p0":   "stratified_5silos_sim_avail_low_intra0p0",
}

NATURAL_PARTITIONS = {"stratified", "modality_unique"}

# PEPSY contrastive loss weight by partition (mirrors wearable complete/medium vs low).
PEPSY_CONTRASTIVE_WEIGHT_BY_PARTITION = {
    "stratified":      0.1,
    "med_intra_0p0":   0.1,
    "low_intra_0p0":   0.2,
    "modality_unique": 0.2,
}


def resolve_pepsy_contrastive_weight(partition_name: str) -> float:
    """Return PEPSY contrastive weight for an ADNI partition name."""
    if partition_name not in PEPSY_CONTRASTIVE_WEIGHT_BY_PARTITION:
        raise ValueError(
            f"Unknown partition '{partition_name}' for PEPSY contrastive weight. "
            f"Available: {sorted(PEPSY_CONTRASTIVE_WEIGHT_BY_PARTITION.keys())}"
        )
    return PEPSY_CONTRASTIVE_WEIGHT_BY_PARTITION[partition_name]


# ======================================================================= #
# Federated learning configuration                                        #
# ======================================================================= #

# Common FL defaults shared across all methods
ADNI_FL_COMMON = {
    "partition_name": "stratified",
    "client_ids": ["silo_1", "silo_2", "silo_3", "silo_4", "silo_5"],
    "num_total_clients": 5,
    "client_selection_rate": 1.0,
    "global_rounds": 150,
    "local_epochs": 1,
}

# lr/bs grid: {4e-5, 1e-4} x {8, 16}; method-specific HPs per adni_pipeline/run/hps/*_adni.sh
ADNI_METHOD_CONFIGS = {
    "fedavg":   {"local_lr": 1e-4, "local_bs": 16},  # tuned
    "fedprox":  {"local_lr": 1e-4, "local_bs": 16, "fedprox_mu": 0.01},  # tuned
    "fedper":   {"local_lr": 1e-4, "local_bs": 8},  # tuned
    "moon":     {"local_lr": 1e-4, "local_bs": 8, "moon_mu": 0.1, "moon_temp": 1.0},  # tuned
    "fedrod":   {"local_lr": 1e-4, "local_bs": 8},  # tuned
    "pmcmfl":   {"local_lr": 1e-4, "local_bs": 8, "pmcmfl_clip_loss_weight": 0.5},  # tuned
    "pepsy":    {"local_lr": 4e-5, "local_bs": 16,  # tuned (contrastive_weight by partition)
                 "pepsy_pool_size": 10, "pepsy_top_k": 3, "pepsy_num_outer_loops": 10},
    "flux":     {"local_lr": 1e-4, "local_bs": 8, "flux_hidden_dim": 128},  # tuned
}


def load_adni_fl_config(method: str) -> dict:
    """Build a complete config dict for a given FL method.

    Merges ADNI_DATASET_CFG + ADNI_FL_COMMON + method-specific overrides.
    """
    cfg = dict(ADNI_DATASET_CFG)
    cfg.update(ADNI_FL_COMMON)

    if method not in ADNI_METHOD_CONFIGS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {sorted(ADNI_METHOD_CONFIGS.keys())}"
        )
    cfg.update(ADNI_METHOD_CONFIGS[method])
    cfg["method"] = method
    if method == "pepsy":
        cfg["pepsy_contrastive_weight"] = resolve_pepsy_contrastive_weight(cfg["partition_name"])
    return cfg

