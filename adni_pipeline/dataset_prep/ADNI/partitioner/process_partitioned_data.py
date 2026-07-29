"""
    Impute, scale, and save partitioned data w/o leaking information
Note: Any imputation, scaling should be applied after train-test split.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

ADNI_PREP_DIR = Path(__file__).resolve().parents[1]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser to process partitioned data")
    parser.add_argument('--path_config', type=str, default=None,
                        help='Path to adni_paths.json (default: ADNI_PATH_CONFIG or ../adni_paths.json)')
    parser.add_argument('--mode', type=str, default='centralized', choices=['centralized', 'federated'],
                        help='Mode to process data: centralized or federated')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='Override partition directory (default: partitioned/stratified_5silos)')
    parser.add_argument('--input_data_dir', type=str, default=None,
                        help='Override processed input directory from the path config')
    parser.add_argument('--modality', type=str, default='genomic', choices=['genomic', 'clinical', 'biospecimen'],
                        help='Modality to process: genomic, clinical, or biospecimen')
    return parser.parse_args()


def read_patient_splits(split_dir, mode):
    path_to_json = f'{split_dir}/split_metadata.json'
    if not os.path.exists(path_to_json):
        raise FileNotFoundError(f"Metadata file not found at {path_to_json}")
    with open(path_to_json, 'r') as f:
        split_data = json.load(f)

    return split_data['data'][mode]


def _are_all_splits_non_empty(df_train, df_val, df_test):
    """Check if each split is non-empty"""
    return (len(df_train) > 0) and (len(df_val) > 0) and (len(df_test) > 0)


def process_genomic_data(train_patient_ids, test_patient_ids, val_patient_ids, df, split_data_dir, mode, silo_id, modality_name):
    """Apply mode imputation (since genotype values are 0, 1, 2..) and MinMax scaling to genomic data"""
    # 1) Extract dataframes for each split (train, valid, test)
    df_train = df.loc[df.index.intersection(train_patient_ids)].copy()
    df_val   = df.loc[df.index.intersection(val_patient_ids)].copy()
    df_test  = df.loc[df.index.intersection(test_patient_ids)].copy()
    print(f'Len of train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}')

    # Check if any split is empty
    if not _are_all_splits_non_empty(df_train, df_val, df_test):
        print("One or more splits are empty. Exiting processing genomic data.")
        return

    # 2) Mode-imputation. If no values in a column, fill with 0
    fill_vals = {}
    for col in tqdm(df_train.columns, desc='Computing mode for each column and fill for train'):
        non_na = df_train[col].dropna()
        if not non_na.empty:
            fill_val = non_na.mode().iloc[0]
        else:
            fill_val = 0 # fallback value if all rows in this column are NaN
        fill_vals[col] = fill_val
        df_train[col].fillna(fill_val, inplace=True)

    # Fill missing values in val and test using train mode values
    for df_split, name in zip([df_val, df_test], ['val', 'test']):
        for col in tqdm(df_split.columns, desc=f'Filling missing values in {name} using train mode values'):
            df_split[col].fillna(fill_vals[col], inplace=True)


    # 3) MinMax scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_arr = scaler.fit_transform(df_train.values)
    val_arr   = scaler.transform(df_val.values)
    test_arr  = scaler.transform(df_test.values)
    print(f'Shape of train_arr: {train_arr.shape}, val_arr: {val_arr.shape}, test_arr: {test_arr.shape}')


    # 4) Save each split: .npy + index map .json
    for arr, df_split, split_name in [
        (train_arr, df_train, 'train'),
        (val_arr,   df_val,   'val'),
        (test_arr,  df_test,  'test'),
    ]:
        # a) file paths
        save_dir = f'{split_data_dir}/{mode}'  # 'split_data_dir/centralized' or split_data_dir/federated
        if mode == 'federated':
            save_dir += f'/silo_{silo_id}'     # split_data_dir/federated/silo_1
        save_dir += f'/{modality_name}'        # split_data_dir/federated/silo_1/genomic
                                               # or split_data_dir/centralized/genomic
        os.makedirs(save_dir, exist_ok=True)
        npy_path  = os.path.join(save_dir, f'{split_name}_data.npy')
        json_path = os.path.join(save_dir, f'{split_name}_pid_to_idx.json')

        np.save(npy_path, arr)

        # c) build & save mapping patient_id -> row_index
        patient_ids = df_split.index.tolist()
        id_to_idx   = {pid: idx for idx, pid in enumerate(patient_ids)}
        with open(json_path, 'w') as fp:
            json.dump(id_to_idx, fp)

        print(f"Saved {split_name}:  array → {npy_path},  index → {json_path}")


def process_clinical_data(train_patients, test_patients, val_patients, df, split_data_dir, mode, silo_id, modality_name):
    # 1) Exclude columns that might have information about labels
    columns_to_exclude = [col for col in df.columns if
                          col.startswith('PTCOGBEG') or col.startswith('PTADDX') or col.startswith('PTADBEG')]
    if len(columns_to_exclude) > 0:
        print(f'Will exclude {len(columns_to_exclude)} columns')
        df = df.drop(columns_to_exclude, axis=1)

    # 2) Identify continuous vs. binary columns
    feats = [c for c in df.columns if c != 'PTID']
    # continuous = >2 distinct values in TRAIN
    cont_cols = [c for c in feats
                 if df[c].dtype.kind in ('i', 'f')
                 and df[c].nunique(dropna=True) > 2]
    # everything else is a “binary” dummy (either {-1,1} or {0,1})
    bin_cols = [c for c in feats if c not in cont_cols]

    # 3) Extract dataframes for each split (train, valid, test)
    df_train = df.loc[df.index.intersection(train_patients)].copy()
    df_val   = df.loc[df.index.intersection(val_patients)].copy()
    df_test  = df.loc[df.index.intersection(test_patients)].copy()
    print(f'Len of train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}')

    # Check if any split is empty
    if not _are_all_splits_non_empty(df_train, df_val, df_test):
        print("One or more splits are empty. Exiting processing clinical data.")
        return

    # 4) Compute imputation values
    train_fill = {}
    # 4a) continuous → mean
    for c in tqdm(cont_cols, desc="Train: computing continuous means"):
        non_na = df_train[c].dropna()
        train_fill[c] = non_na.mean() if not non_na.empty else 0.0
    # 4b) binary → mode
    for c in tqdm(bin_cols, desc="Train: computing binary modes"):
        non_na = df_train[c].dropna()
        train_fill[c] = non_na.mode().iloc[0] if not non_na.empty else 0

    # 5) Apply imputes
    for split in (df_train, df_val, df_test):
        for c, fillv in train_fill.items():
            split[c].fillna(fillv, inplace=True)

    # 6) Apply scaling for continuous values [-1,1]
    if cont_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_train[cont_cols])
        df_train[cont_cols] = scaler.transform(df_train[cont_cols])
        df_val[cont_cols] = scaler.transform(df_val[cont_cols])
        df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    print(f"[After processing] shapes → train: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}")

    train_arr = df_train.to_numpy()
    val_arr = df_val.to_numpy()
    test_arr = df_test.to_numpy()

    # 7) Save each split: .npy + index map .json
    for arr, df_split, split_name in [
        (train_arr, df_train, 'train'),
        (val_arr,   df_val,   'val'),
        (test_arr,  df_test,  'test'),
    ]:
        # a) file paths
        save_dir = f'{split_data_dir}/{mode}'  # 'split_data_dir/centralized' or split_data_dir/federated
        if mode == 'federated':
            save_dir += f'/silo_{silo_id}'     # split_data_dir/federated/silo_1
        save_dir += f'/{modality_name}'        # split_data_dir/federated/silo_1/genomic
                                               # or split_data_dir/centralized/genomic
        os.makedirs(save_dir, exist_ok=True)
        npy_path  = os.path.join(save_dir, f'{split_name}_data.npy')
        json_path = os.path.join(save_dir, f'{split_name}_pid_to_idx.json')

        np.save(npy_path, arr)

        # c) build & save mapping patient_id -> row_index
        patient_ids = df_split.index.tolist()
        id_to_idx   = {pid: idx for idx, pid in enumerate(patient_ids)}
        with open(json_path, 'w') as fp:
            json.dump(id_to_idx, fp)

        print(f"Saved {split_name}:  array → {npy_path},  index → {json_path}")


def process_biospecimen_data(train_patients, test_patients, val_patients, df, split_data_dir, mode, silo_id, modality_name):
    # 1) Identify continuous vs. binary columns
    feats = [c for c in df.columns if c != 'PTID']
    # continuous = >2 distinct values in TRAIN
    cont_cols = [c for c in feats
                 if df[c].dtype.kind in ('i', 'f')
                 and df[c].nunique(dropna=True) > 2]
    # everything else is a “binary” dummy (either {-1,1} or {0,1})
    bin_cols = [c for c in feats if c not in cont_cols]

    df_train = df.loc[df.index.intersection(train_patients)].copy()
    df_val   = df.loc[df.index.intersection(val_patients)].copy()
    df_test  = df.loc[df.index.intersection(test_patients)].copy()
    print(f'Len of train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}')

    # Check if any split is empty
    if not _are_all_splits_non_empty(df_train, df_val, df_test):
        print("One or more splits are empty. Exiting processing biospecimen data.")
        return

    # 2) Compute imputation values
    train_fill = {}
    # 2a) continuous → mean
    for c in tqdm(cont_cols, desc="Train: computing continuous means"):
        non_na = df_train[c].dropna()
        train_fill[c] = non_na.mean() if not non_na.empty else 0.0
    # 2b) binary → mode
    for c in tqdm(bin_cols, desc="Train: computing binary modes"):
        non_na = df_train[c].dropna()
        train_fill[c] = non_na.mode().iloc[0] if not non_na.empty else 0

    # 3) Apply imputation
    for split in (df_train, df_val, df_test):
        for c, fillv in train_fill.items():
            split[c].fillna(fillv, inplace=True)

    # 4) Scale based on train data → [-1,1]
    if cont_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_train[cont_cols])
        df_train[cont_cols] = scaler.transform(df_train[cont_cols])
        df_val[cont_cols] = scaler.transform(df_val[cont_cols])
        df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    print(f"[After processing] shapes → train: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}")

    train_arr = df_train.to_numpy()
    val_arr = df_val.to_numpy()
    test_arr = df_test.to_numpy()

    # 7) Save each split: .npy + index map .json
    for arr, df_split, split_name in [
        (train_arr, df_train, 'train'),
        (val_arr,   df_val,   'val'),
        (test_arr,  df_test,  'test'),
    ]:
        # a) file paths
        save_dir = f'{split_data_dir}/{mode}'  # 'split_data_dir/centralized' or split_data_dir/federated
        if mode == 'federated':
            save_dir += f'/silo_{silo_id}'     # split_data_dir/federated/silo_1
        save_dir += f'/{modality_name}'        # split_data_dir/federated/silo_1/genomic
                                               # or split_data_dir/centralized/genomic
        os.makedirs(save_dir, exist_ok=True)
        npy_path  = os.path.join(save_dir, f'{split_name}_data.npy')
        json_path = os.path.join(save_dir, f'{split_name}_pid_to_idx.json')

        np.save(npy_path, arr)

        # c) build & save mapping patient_id -> row_index
        patient_ids = df_split.index.tolist()
        id_to_idx   = {pid: idx for idx, pid in enumerate(patient_ids)}
        with open(json_path, 'w') as fp:
            json.dump(id_to_idx, fp)

        print(f"Saved {split_name}:  array → {npy_path},  index → {json_path}")


def process_data(modality, train_patients, test_patients, val_patients, df,
                 split_data_dir, mode, silo_id):
    # silo_id is None for centralized mode
    print(f'Processing in {mode} mode', f' with silo_id: {silo_id}...' if silo_id else '...')

    # Need to handle for each modality
    # For MRI, we already applied intensity normalization, so no need to apply again. Do it for other modalities
    if modality == 'genomic':
        process_genomic_data(train_patients, test_patients, val_patients, df, split_data_dir, mode, silo_id, modality)
    elif modality == 'clinical':
        process_clinical_data(train_patients, test_patients, val_patients, df, split_data_dir, mode, silo_id, modality)
    elif modality == 'biospecimen':
        process_biospecimen_data(train_patients, test_patients, val_patients, df, split_data_dir, mode, silo_id, modality)
    else:
        raise ValueError(f"Unknown modality: {modality}")




def main():
    args = parse_args()
    paths = load_adni_paths(args.path_config)
    split_dir = args.split_dir or str(paths["partitioned_dir"] / "stratified_5silos")
    input_data_dir = args.input_data_dir or str(paths["processed_dir"])
    patient_splits = read_patient_splits(split_dir, args.mode)

    # Load dataframes for genomic, clinical, and biospecimen modalities
    tic = time.time()
    if args.modality == 'genomic':
        df = sc.read_h5ad(f'{input_data_dir}/genomic/genomic_merged.h5ad').to_df()
    elif args.modality == 'clinical':
        df = pd.read_csv(f'{input_data_dir}/clinical/clinical_merged.csv', index_col=0)
    elif args.modality == 'biospecimen':
        df = pd.read_csv(f'{input_data_dir}/biospecimen/biospecimen_merged.csv', index_col=0)
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    print(f'Loaded {args.modality} in {time.time() - tic:.2f} seconds')


    if args.mode == 'centralized':
        train_patients = patient_splits['train']
        test_patients = patient_splits['test']
        val_patients = patient_splits['val']
        process_data(args.modality, train_patients, test_patients, val_patients, df,
                     split_dir, args.mode, silo_id=None)

    elif args.mode == 'federated':
        num_silos = len(patient_splits.keys())  # {keys: silo_id, values: {train: [], test: [], val: []}}
        for silo_id in tqdm(range(1, num_silos + 1)):
            train_patients = patient_splits[f'silo_{silo_id}']['train']
            test_patients = patient_splits[f'silo_{silo_id}']['test']
            val_patients = patient_splits[f'silo_{silo_id}']['val']
            process_data(args.modality, train_patients, test_patients, val_patients, df,
                         split_dir, args.mode, silo_id=silo_id)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'centralized' or 'federated'.")



if __name__ == '__main__':
    main()