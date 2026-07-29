import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

ADNI_PREP_DIR = Path(__file__).resolve().parents[1]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Validate processed ADNI MRI tensors")
    parser.add_argument('--path_config', type=str, default=None,
                        help='Path to adni_paths.json (default: ADNI_PATH_CONFIG or ../adni_paths.json)')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Override the processed MRI directory from the path config')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = load_adni_paths(args.path_config)
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else paths["mri_processed_dir"]

    pid_dirs = os.listdir(base_dir)
    nan_pids = []
    for pid in tqdm(pid_dirs):
        processed_image_path = base_dir / pid / f'{pid}.pt'
        # load torch tensor and check if it contains nan values
        tensor = torch.load(processed_image_path)
        if torch.isnan(tensor).any():
            print(f'Found NaN values in {processed_image_path}')
            nan_pids.append(pid)
    print(f'Total PIDs with NaN values: {len(nan_pids)}/{len(pid_dirs)}')

    for pid in nan_pids:
        print(pid)
        # We will delete those directories since they contain NaN values
        pid_dir = base_dir / pid
        if pid_dir.exists():
            print(f'Deleting directory: {pid_dir}')
            shutil.rmtree(pid_dir)



if __name__ == "__main__":
    main()