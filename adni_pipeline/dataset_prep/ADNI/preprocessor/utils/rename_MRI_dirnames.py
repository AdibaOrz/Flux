"""Rename each folder in the MRI directory from {subject_id} to {subject_id}_{scan_date}"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path

ADNI_PREP_DIR = Path(__file__).resolve().parents[2]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths

def extract_and_move_by_timestamp(root_dir):
    cnt_dict = {}

    for idx, level_1_name in enumerate(os.listdir(root_dir)):
        level_1_path = os.path.join(root_dir, level_1_name)
        if not os.path.isdir(level_1_path):
            continue

        try:
            level_2 = next(os.scandir(level_1_path))
            level_2_path = level_2.path
        except (StopIteration, FileNotFoundError):
            print(f"Skipping '{level_1_name}' – no level 2 folder.")
            continue

        moved_any = False  # Track if anything was moved

        cnt_dict[idx] = len(os.listdir(level_2_path))
        for level_3 in os.scandir(level_2_path):
            if not level_3.is_dir():
                continue

            timestamp_name = level_3.name
            if not re.match(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d+", timestamp_name):
                continue

            new_folder_name = f"{level_1_name}_{timestamp_name}"
            new_folder_path = os.path.join(root_dir, new_folder_name)

            if not os.path.exists(new_folder_path):
                print(f"Moving to: {new_folder_name}")
                shutil.move(level_3.path, new_folder_path)
                moved_any = True
            else:
                print(f"Skipped existing: {new_folder_name}")

        # If we moved everything out, remove the original level_1 folder
        if moved_any:
            print(f"Removing original folder: {level_1_name}")
            shutil.rmtree(level_1_path)

    for idx, count in cnt_dict.items():
        if count == 1:
            continue
        print(f"Folder {idx} has {count} items.")



def parse_args():
    parser = argparse.ArgumentParser(description="Flatten timestamped ADNI MRI directories")
    parser.add_argument('--path_config', type=str, default=None,
                        help='Path to adni_paths.json (default: ADNI_PATH_CONFIG or ../../adni_paths.json)')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Override mri_rename_source_dir from the path config')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = load_adni_paths(args.path_config)
    root_dir = Path(args.root_dir).expanduser() if args.root_dir else paths["mri_rename_source_dir"]
    extract_and_move_by_timestamp(root_dir)

