"""Path configuration for ADNI dataset preprocessing."""

import sys
from pathlib import Path

ADNI_PREP_DIR = Path(__file__).resolve().parents[1]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths


def get_config_dict(modality, config_path=None):
    assert modality in ["MRI",  "biospecimen", "clinical", "genomic"], f"Invalid modality {modality}!"

    paths = load_adni_paths(config_path)
    parent_dir = paths["adni_root"]
    downloaded_date = paths["downloaded_date"]

    config = {
        "genomic": {
            "adni_1_path": parent_dir / "raw/genomic/ADNI_cluster_01_forward_757LONI",
            "adni_2_1_path": parent_dir / "raw/genomic/ADNI_GO_2_Forward_Bin",
            "adni_2_2_path": parent_dir / "raw/genomic/ADNI_GO2_GWAS_2nd_orig_BIN",
            "adni_3_1_path": parent_dir / "raw/genomic/ADNI3_PLINK_Final",
            "adni_3_2_path": parent_dir / "raw/genomic/ADNI3_PLINK_FINAL_2nd",
            "label_path": paths["diagnosis_file"],
            "save_dir": parent_dir / "processed/genomic",
        },

        "MRI": {
            "data_path": paths["mri_data_dir"],
            "save_dir": paths["mri_processed_dir"],
            "reference_atlas_location": paths["mri_reference_atlas"],
            "re-process": False,
            "axial_size": 90,
            "save_2d": False,
            "save_numpy": False,
            "save_torch": True,
            "wandb_log": True,
            "compact_save_path": True
        },

        "biospecimen": {
            "bio_1_path": parent_dir / f"raw/biospecimen/APOERES_{downloaded_date}.csv",
            "bio_2_path": parent_dir / f"raw/biospecimen/UPENNBIOMK_ROCHE_ELECSYS_{downloaded_date}.csv",
            "label_path": paths["diagnosis_file"],
            "save_dir": parent_dir / "processed/biospecimen",
            "wandb_log": True,
        },

        "clinical": {
            "clinical_1_path": parent_dir / f"raw/clinical/MEDHIST_{downloaded_date}.csv",
            "clinical_2_path": parent_dir / f"raw/clinical/NEUROEXM_{downloaded_date}.csv",
            "clinical_3_path": parent_dir / f"raw/clinical/PTDEMOG_{downloaded_date}.csv",
            "clinical_4_path": parent_dir / f"raw/clinical/RECCMEDS_{downloaded_date}.csv",
            "clinical_5_path": parent_dir / f"raw/clinical/VITALS_{downloaded_date}.csv",
            "label_path": paths["diagnosis_file"],
            "save_dir": parent_dir / "processed/clinical",
            "wandb_log": True,
        }
    }

    return config[modality]
