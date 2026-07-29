"""Script to create splits for ADNI dataset
   First, we create silos each with train/val/test splits.
   Then, based on the split, we create centralized train/val/test by collecting corresponding silo data splits.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

ADNI_PREP_DIR = Path(__file__).resolve().parents[1]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths

MODAL_COLS = ['Image', 'Genomic', 'Clinical', 'Biospecimen']  # Columns for modalities in the stat_df


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser to create splits for ADNI dataset")
    parser.add_argument('--path_config', type=str, default=None,
                        help='Path to adni_paths.json (default: ADNI_PATH_CONFIG or ../adni_paths.json)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_silos', type=int, default=5, help='Number of silos to create')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override the ADNI dataset root from the path config')
    parser.add_argument('--downloaded_date', type=str, default=None,
                        help='Override downloaded_date from the path config')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument('--split', type=str, default='stratified',
                        choices=['stratified', 'modality_unique'],
                        help='stratified      => balanced across labels + modalities\n'
                             'modality_unique => one unique modality combination per silo')
    parser.add_argument('--min_silo_size', type=int, default=30,
                        help='Minimum total num of samples per silo for all split types')

    return parser.parse_args()


def load_labels(data_path, downloaded_date):
    _tag = '[Loading labels]'
    label_path = f'{data_path}/raw/diagnosis/DXSUM_{downloaded_date}.csv'
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    initial_label_df = pd.read_csv(label_path, parse_dates=['EXAMDATE'], index_col=None)
    initial_label_df = initial_label_df.sort_values(['PTID', 'EXAMDATE'])

    # Keep rows with the latest exam date
    label_df = (
        initial_label_df.groupby('PTID', as_index=False)
            .tail(1)
            .set_index('PTID')
    )

    # Start label from 0 (0, 1, 2) instead of (1, 2, 3)
    label_df['DIAGNOSIS'] -= 1
    print(f'{_tag} Before removing duplicate patients:', len(initial_label_df))
    print(f'{_tag} After removing duplicate patients (keep only the latest diagnosis result): ', len(label_df))

    label_df = label_df[['DIAGNOSIS', 'EXAMDATE']]
    label_df.dropna(inplace=True)  # Drop rows without diagnosis values
    print(f'{_tag} After removing nan values: ', len(label_df))
    label_df['DIAGNOSIS'] = label_df['DIAGNOSIS'].astype(int)
    label_df = label_df.reset_index()

    return label_df


def _assert_unique_ptids(ptid_list, modality):
    """
    Assert that the list of PTIDs is unique.
    """
    assert len(ptid_list) == len(list(set(ptid_list))), f"Duplicate PTIDs found in {modality} modality data."


def collect_ptids_from_modalities(data_path):
    _tag = '[Collecting PTIDs from modalities]'
    modality_ptids = {}

    ## Check for images
    image_path = f'{data_path}/processed/MRI'
    genomic_path = f'{data_path}/processed/genomic/genomic_merged.h5ad'
    clinical_path = f'{data_path}/processed/clinical/clinical_merged.csv'
    biospecimen_path = f'{data_path}/processed/biospecimen/biospecimen_merged.csv'

    # Assert for all paths
    assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
    assert os.path.exists(genomic_path), f"Genomic path does not exist: {genomic_path}"
    assert os.path.exists(clinical_path), f"Clinical path does not exist: {clinical_path}"
    assert os.path.exists(biospecimen_path), f"Biospecimen path does not exist: {biospecimen_path}"


    ### 1. Collect unique PTIDs from image (MRI) modality data
    image_ptids = os.listdir(image_path)
    _assert_unique_ptids(image_ptids, 'image')
    modality_ptids['image'] = image_ptids
    print(f'{_tag} {len(image_ptids)} unique PTIDs found in image data, such as: {image_ptids[:2]}, ...')


    ### 2. Collect unique PTIDs from genomic modality data
    genomic_df = sc.read_h5ad(genomic_path).to_df()
    genomic_ptids = list(genomic_df.index)
    _assert_unique_ptids(genomic_ptids, 'genomic')
    modality_ptids['genomic'] = genomic_ptids
    print(f'{_tag} {len(genomic_ptids)} unique PTIDs found in genomic data, such as: {genomic_ptids[:2]}, ...')


    ### 3. Collect unique PTIDs from clinical modality data
    clinical_df = pd.read_csv(clinical_path)
    clinical_ptids = list(clinical_df['PTID'].unique())
    _assert_unique_ptids(clinical_ptids, 'clinical')
    modality_ptids['clinical'] = clinical_ptids
    print(f'{_tag} {len(clinical_ptids)} unique PTIDs found in clinical data, such as: {clinical_ptids[:2]}, ...')

    ### 4. Collect unique PTIDs from biospecimen modality data
    biospecimen_df = pd.read_csv(biospecimen_path)
    biospecimen_ptids = list(biospecimen_df['PTID'].unique())
    _assert_unique_ptids(biospecimen_ptids, 'biospecimen')
    modality_ptids['biospecimen'] = biospecimen_ptids
    print(f'{_tag} {len(biospecimen_ptids)} unique PTIDs found in biospecimen data, such as: {biospecimen_ptids[:2]}, ...')

    return modality_ptids


def create_and_save_stat_df(label_df, ptids_by_modality, data_path):
    _tag = '[Creating stat_df]'

    all_ptids = sorted(set(ptids_by_modality['image']) | set(ptids_by_modality['genomic'])
                     | set(ptids_by_modality['clinical']) | set(ptids_by_modality['biospecimen']))
    stat_df = pd.DataFrame({'PTID': all_ptids})

    stat_df['Image'] = stat_df['PTID'].isin(ptids_by_modality['image']).astype(int)
    stat_df['Genomic'] = stat_df['PTID'].isin(ptids_by_modality['genomic']).astype(int)
    stat_df['Clinical'] = stat_df['PTID'].isin(ptids_by_modality['clinical']).astype(int)
    stat_df['Biospecimen'] = stat_df['PTID'].isin(ptids_by_modality['biospecimen']).astype(int)

    # Merge with labels
    stat_df = stat_df.merge(label_df[['PTID', 'DIAGNOSIS']], how='left')

    # Drop if no diagnosis is available
    print(f'{_tag} Before dropping rows w/o diagnosis: {len(stat_df)}')
    stat_df.dropna(inplace=True)
    print(f'{_tag} After dropping rows w/o diagnosis: {len(stat_df)}')
    stat_df['DIAGNOSIS'] = stat_df['DIAGNOSIS'].astype(int)

    stat_df.rename(columns={'DIAGNOSIS': 'label'}, inplace=True)
    stat_df.to_csv(f'{data_path}/processed/labels.csv', index=False)
    print(f'{_tag} Saved stat_df to {data_path}/processed/labels.csv')

    return stat_df


def _train_val_test_split(df, train_size, seed):
    """
    Split a single silo DataFrame into train/val/test (70/15/15 by default),
    stratifying on modalities+label when possible, else falling back to random.
    Returns (df_train, df_val, df_test).
    """
    y = np.hstack([df[MODAL_COLS].values, pd.get_dummies(df['label'], prefix='lbl').values])
    try:
        msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-train_size, random_state=seed)
        tr_idx, temp_idx = next(msss1.split(np.zeros((len(df), 1)), y))
        df_tr, df_temp = df.iloc[tr_idx], df.iloc[temp_idx]

        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = next(msss2.split(np.zeros((len(df_temp), 1)), y[temp_idx]))
        df_val, df_test = df_temp.iloc[val_idx], df_temp.iloc[test_idx]

    except ValueError:
        # Fallback = naive random split
        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        tr_end = int(train_size*n)
        val_end = tr_end + int((1-train_size)*n/2)
        df_tr, df_val, df_test = np.split(shuffled, [tr_end, val_end])

    return map(lambda x: x.reset_index(drop=True), (df_tr, df_val, df_test))


def _check_min_size(silos, min_size):
    for k, v in silos.items():
        n = len(v['train']) + len(v['val']) + len(v['test'])
        if n < min_size:
            raise ValueError(
                f"{k} has {n} samples < min_silo_size {min_size}.\n"
                "Try increasing data, adjusting parameters, "
                "or lowering --min_silo_size.")


def _finalise_and_save(silos, save_dir, meta, min_silo_size):
    _check_min_size(silos, min_silo_size)

    cat = lambda s: pd.concat([v[s] for v in silos.values()])
    centered = {s: cat(s)['PTID'].tolist() for s in ['train', 'val', 'test']}
    silo_dict = {k: {s: v[s]['PTID'].tolist() for s in ['train', 'val', 'test']}
                 for k, v in silos.items()}

    payload = {'data': {'centralized': centered, 'federated': silo_dict},
               'metadata': meta | {'generated_on': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/split_metadata.json', 'w') as f:
        json.dump(payload, f, indent=4)
    print(f'[SAVE] -> {save_dir}/split_metadata.json')


def split_stratified(df, num_silos, train_size, data_path, seed, min_silo_size):
    y = np.hstack([df[MODAL_COLS].values, pd.get_dummies(df['label']).values])
    mskf = MultilabelStratifiedKFold(n_splits=num_silos, shuffle=True, random_state=seed)

    silos = {}
    for i, (_, idx) in enumerate(mskf.split(np.zeros((len(df), 1)), y), 1):
        silo_df = df.iloc[idx]
        silos[f'silo_{i}'] = dict(zip(['train', 'val', 'test'],
                                      _train_val_test_split(silo_df, train_size, seed+i)))
    _finalise_and_save(
        silos,
        f'{data_path}/partitioned/stratified_{num_silos}silos',
        {'split_type': 'stratified',
         'num_silos': num_silos,
         'train_size': train_size,
         'seed': seed},
        min_silo_size
    )


def split_by_modality_unique(df, train_size, data_path, seed, min_silo_size):
    df = df.copy()
    df['combo'] = df[MODAL_COLS].apply(lambda r: ''.join(c[0] for c, v in zip(MODAL_COLS, r) if v), axis=1)
    valid = df['combo'].value_counts()
    valid = valid[valid >= min_silo_size].index.tolist()
    if not valid:
        raise ValueError(f'No combo ≥ {min_silo_size} samples.')

    silos = {}
    for i, c in enumerate(valid, 1):
        silo_df = df[df.combo == c].drop(columns='combo')
        silos[f'silo_{i}'] = dict(zip(['train', 'val', 'test'],
                                      _train_val_test_split(silo_df, train_size, seed+i)))
    _finalise_and_save(
        silos,
        f'{data_path}/partitioned/modality_unique_{len(valid)}silos',
        {'split_type': 'modality_unique',
         'num_silos': len(valid),
         'train_size': train_size,
         'seed': seed,
         'min_silo_size': min_silo_size},
        min_silo_size
    )


def main():
    args = parse_args()
    paths = load_adni_paths(args.path_config)
    data_path = args.data_path or str(paths["adni_root"])
    downloaded_date = args.downloaded_date or paths["downloaded_date"]

    label_df = load_labels(data_path, downloaded_date)
    ptids_by_modality = collect_ptids_from_modalities(data_path)
    stat_df = create_and_save_stat_df(label_df, ptids_by_modality, data_path)

    # Partition data to federated silos & centralized dataset
    if args.split == 'stratified':
        split_stratified(stat_df, args.num_silos, args.train_size, data_path, args.seed, args.min_silo_size)
    elif args.split == 'modality_unique':
        split_by_modality_unique(stat_df, args.train_size, data_path, args.seed, args.min_silo_size)
    else:
        raise ValueError(f"Unknown split type: {args.split}")


if __name__ == "__main__":
    main()