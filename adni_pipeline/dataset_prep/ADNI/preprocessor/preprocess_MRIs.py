"""
References go to: https://github.com/SANCHES-Pedro/adni_preprocessing
Some parts have been modified to fit the current project
"""

import os
import time
import torch
import wandb
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import SimpleITK as sitk
from datetime import datetime
from collections import defaultdict
from preprocessing_configs import get_config_dict


def get_unique_img_files(img_folder):
    """Get unique the most preprocessed image files from a list of image files."""
    found_img_ids = np.array([img_file.parents[0].name for img_file in img_folder])
    inverse_found_img_ids = found_img_ids[::-1]
    img_ids, indices = np.unique(inverse_found_img_ids, return_index=True)
    unique_img_ids_paths = np.array(img_folder)[::-1][indices]

    assert len(img_folder) >= len(unique_img_ids_paths) > 0
    return unique_img_ids_paths


def run_fsl(img_path, save_dir, ref):
    """Run FSL preprocessing on an image file."""
    os.system(f'fslreorient2std {img_path} {save_dir}')
    os.system(f'robustfov -i {save_dir} -r {save_dir}')
    os.system(f'bet {save_dir} {save_dir} -R')
    os.system(f'flirt -in {save_dir} -ref {ref} -out {save_dir}')
    # "fast" saves output file as {file_name}_restore.nii.gz
    os.system(f'fast --nopve -B -o {save_dir} {save_dir}')
    fsl_img_path = Path(str(save_dir).replace(".nii", "_restore.nii"))
    return fsl_img_path


def load_numpy_img(fsl_img_path):
    img_sitk = sitk.ReadImage(str(fsl_img_path))
    img_np = sitk.GetArrayFromImage(img_sitk)
    return img_np


def normalize_intensity(volume: np.array, clip_ratio: float = 99.5):
    # Normalize the pixel values of volumes (our numpy img) to (-1, 1)
    assert np.min(volume) == 0.0
    volume_max = np.percentile(volume, clip_ratio)
    volume = np.clip(volume / volume_max, 0, 1) * 2 - 1
    return volume


def crop_numpy_img(image: np.array, axial_size: int = 90, central_crop_along_z: bool = True):
    init_shape = image.shape
    cropped_image = image
    if axial_size is not None:
        cropped_image = cropped_image[:,
                        init_shape[1] // 2 - axial_size // 2:init_shape[1] // 2 + axial_size // 2,
                        init_shape[2] // 2 - axial_size // 2:init_shape[2] // 2 + axial_size // 2]
    if central_crop_along_z:
        cropped_image = cropped_image[30:60, ...]
    return cropped_image


def save_numpy_img(image_np, save_dir):
    save_dir = str(save_dir).replace(".nii.gz", "")
    data_dict = {"image": image_np}
    np.savez_compressed(save_dir, data_dict)  # saving into .npz



def save_2d_img(image_np, save_dir):
    save_dir = str(save_dir).replace(".nii.gz", "")
    for slice in range(image_np.shape[0]):
        Image.fromarray(image_np[slice]).save(save_dir + f"_slice{slice}.tiff")


def save_torch_tensor(image_np, save_dir):
    save_dir = str(save_dir).replace(".nii.gz", "")
    tensor = torch.tensor(image_np, dtype=torch.float32)
    torch.save(tensor, save_dir + '.pt')


def remove_nii_files(path: Path):
    for file in path.parent.glob('**/*.nii*'):
        file.unlink()


def get_latest_recorded_folders(paths):
    # Group paths by user ID
    groups = defaultdict(list)
    for p in paths:
        # Extract user ID and date/time from filename
        parts = p.stem.split('_')
        user_id = '_'.join(parts[:3])  # e.g. "002_S_0295"
        date_str = parts[3]  # e.g. "2006-04-18"
        time_str = ':'.join(parts[4:7])  # e.g. "08:20:30"
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        groups[user_id].append((p, dt))

    # Filter to keep only the latest file per user
    latest_paths = []
    for user_id, entries in groups.items():
        # Verbose output
        # print(f"User {user_id} had {len(entries)} files with dates:")
        # for _, dt in sorted(entries, key=lambda x: x[1]):
        #     print(f"  - {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        latest_entry = max(entries, key=lambda x: x[1])
        # print(f"Keeping only the latest: {latest_entry[1].strftime('%Y-%m-%d %H:%M:%S')} ({latest_entry[0]})\n")
        latest_paths.append(latest_entry[0])

    # Resulting list of latest paths
    # print("Filtered list of latest paths:")
    # for p in latest_paths:
    #     print(p)
    return latest_paths


def get_compact_save_dir(longer_save_dir):
    parts = longer_save_dir.parts
    try:
        data_idx = len(parts) - 1 - parts[::-1].index('data')
    except ValueError:
        raise ValueError(f"Cannot find 'data' in {longer_save_dir}")

    id_with_date = parts[data_idx + 1]
    ptid = '_'.join(id_with_date.split('_')[:3])
    # print('ptid:', ptid)
    # print('longer_save_dir:', longer_save_dir)

    # make compact save_dir upto "data" then ptid
    compact_save_dir = Path(*parts[:data_idx], ptid, f'{ptid}.nii.gz')
    return compact_save_dir



def main():
    tic = time.time()
    config = get_config_dict(modality='MRI')
    log_wandb = config['wandb_log']
    if log_wandb:
        wandb.init(
            project='Preprocess_ADNI',
            config=config)

    data_path = config['data_path']
    subject_folder_list = sorted(list(data_path.glob('*')))
    subject_folder_list = get_latest_recorded_folders(subject_folder_list)
    print(f'Unique subject folders: {len(subject_folder_list)}')


    success_count = 0
    for subject_num, subject_folder in tqdm(enumerate(subject_folder_list)):
        print(f'Processing subject {subject_num + 1}/{len(subject_folder_list)}')
        # print(f'--> Subject folder: {subject_folder}')
        img_files = sorted(subject_folder.glob('**/*.nii'))
        unq_img_files = get_unique_img_files(img_files)

        ## Process current subject's image files
        for img_path in unq_img_files:
            start_time = time.time()
            relative_img_path = img_path.relative_to(data_path)
            save_dir = (
                config["save_dir"] / "data" / relative_img_path
            ).with_suffix(".nii.gz")
            if config['compact_save_path']:
                save_dir = get_compact_save_dir(save_dir)
            save_dir.parent.mkdir(parents=True, exist_ok=True)  # w/o filename

            ## Check if preprocessed images already exist
            if len(list(save_dir.parent.glob("**/*.tiff"))) > 0 and not config["re_process"]:
                continue  # Skip if preprocessed images already exist
            try:
                fsl_img_path = run_fsl(img_path, save_dir, config["reference_atlas_location"])
                numpy_img = load_numpy_img(fsl_img_path)
            except:
                with open('load_error_list.txt', 'a') as f:
                    f.write(f'{img_path}\n')
                continue

            # Normalize intensity values
            norm_numpy_img = normalize_intensity(numpy_img)

            # Crop the normalized numpy image
            if config["axial_size"] is not None:
                cropped_norm_numpy_img = crop_numpy_img(norm_numpy_img, axial_size=config["axial_size"])

            # Save in the required format(s)
            if config["save_2d"]:
                save_2d_img(cropped_norm_numpy_img, save_dir)
            if config["save_numpy"]:
                save_numpy_img(cropped_norm_numpy_img, save_dir)
            if config["save_torch"]:
                save_torch_tensor(cropped_norm_numpy_img, save_dir)

            remove_nii_files(save_dir) # -> Uncomment to remove .nii files
            print(f'--> Processed {img_path.name} and saved to {save_dir}'
                  f'\nTook {time.time() - start_time:.2f} seconds to process a single image')
            success_count += 1

    toc = time.time()
    print(f"Took {toc - tic:.2f} seconds to process all subjects")
    print(f"Successfully processed {success_count} images")
    if log_wandb:
        wandb.finish()



if __name__ == "__main__":
    main()