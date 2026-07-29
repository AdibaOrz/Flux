import os
import sys
import time
import json
import torch
import logging
import argparse
import warnings
import pandas as pd
from torch.utils.data import TensorDataset

sys.path.append('../')
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

from configs import load_dataset_configs



def parse_arguments():
    parser = argparse.ArgumentParser(description="Create tensor dataset from dataframe")
    parser.add_argument('--dataset', type=str, default='pamap2',
                        choices=['pamap2', 'realworldhar', 'sleepedf20'],
                        help='Name of the dataset')
    parser.add_argument('--availability', type=str, nargs='+', default=['complete', 'medium', 'low'],
                        choices=['complete', 'medium', 'low'],
                        help='Availability levels of the dataset (can specify multiple)')
    parser.add_argument('--stability', type=str, nargs='+', default=['complete', 'medium', 'low'],
                        choices=['complete', 'medium', 'low'],
                        help='Stability levels of the dataset (can specify multiple)')
    parser.add_argument('--local_imp', type=str, default='zero', help='Local imputation method (e.g., zero)')
    parser.add_argument('--scaling_method', type=str, default='standard', choices=['standard', 'minmax'],
                        help='Scaling method used in simulate_incompleteness.py (must match simulation step)')
    return parser.parse_args()


def create_windowed_dataset(df, dataset_configs, local_imp):
    win_len = dataset_configs['seq_len']  # rows per window (seq_len from configs)
    input_data, exists_mask, class_labels = [], [], []

    label_list = list(df['label'].unique())
    for label in label_list:
        label_df = df[df['label']==label]
        for win_start in range(0, len(label_df), win_len):
            win_end = win_start + win_len
            if win_end > len(label_df):
                break
            _input_data = label_df.iloc[win_start:win_end][dataset_configs['feature_columns']].values
            _input_data = _input_data.T
            _exists_mask = (~torch.isnan(torch.tensor(_input_data))).int()
            _class_label = torch.tensor(label, dtype=torch.int)

            # Fill NaN values
            _input_data = torch.tensor(_input_data, dtype=torch.float)
            # Fill NaN values with zero
            if local_imp == 'zero':
                _input_data[torch.isnan(_input_data)] = 0.0
            else:
                raise NotImplementedError(f'local_imp: {local_imp} is not implemented yet')

            input_data.append(_input_data)
            exists_mask.append(_exists_mask)
            class_labels.append(_class_label)

    # Convert to pytorch tensors
    input_data = torch.stack(input_data) # shape: (N, features, seq_len)
    exists_mask = torch.stack(exists_mask) # shape: (N, features, seq_len)
    class_labels = torch.stack(class_labels) # shape: (N,)

    torch_dataset = TensorDataset(input_data, exists_mask, class_labels)
    del input_data, exists_mask, class_labels
    return torch_dataset


def load_df(dataset_name, simulated_dir, split_type, scaling_method):
    split_path = os.path.join(simulated_dir, f"{dataset_name}_{split_type}_{scaling_method}_scaled.csv")
    df = pd.read_csv(split_path)
    missing_cnt = df.isnull().sum().sum()
    logging.info(f"Missing values in {split_type} set: {missing_cnt}")
    return df


def get_dataset_path_and_configs(dataset_name, availability, stability,
                                 json_dataset_path_configs="../data_preprocessor/dataset_paths.json"):
    # Get the path to the simulated dataset and dataset configurations
    with open(json_dataset_path_configs) as f:
        dataset_path_configs = json.load(f)
    simulated_data_dir = os.path.join(dataset_path_configs[dataset_name]['simulated'], f"avail:{availability}/stab:{stability}")
    dataset_configs = load_dataset_configs(dataset_name)
    return simulated_data_dir, dataset_configs


def process_dataset(dataset_name, availability, stability, local_imp, scaling_method):
    simulated_data_dir, dataset_configs = get_dataset_path_and_configs(dataset_name, availability, stability)
    for split_type in ['train', 'val', 'test']:
        df = load_df(dataset_name, simulated_data_dir, split_type, scaling_method)
        client_ids = dataset_configs['client_ids']
        for client_id in client_ids:
            client_df = df[df['user_id'] == client_id]
            # Create a windowed dataset
            torch_dataset = create_windowed_dataset(client_df, dataset_configs, local_imp)

            # Save the tensor dataset
            save_dir = f'{simulated_data_dir}/client_data/{client_id}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = f'{save_dir}/{split_type}_tensors.pt'
            torch.save(torch_dataset, save_path)
            logging.info(f"Saved {split_type} tensor dataset to {save_path}")


def main():
    args = parse_arguments()
    dataset_name = args.dataset
    avail_levels, stab_levels, local_imp = args.availability, args.stability, args.local_imp
    scaling_method = args.scaling_method

    for availability in avail_levels:
        for stability in stab_levels:
            start_time = time.time()
            logging.info(
                f"Processing dataset: {dataset_name}, Availability: {availability}, "
                f"Stability: {stability}, scaling: {scaling_method}..."
            )
            process_dataset(dataset_name, availability, stability, local_imp, scaling_method)
            logging.info(f'Took {time.time() - start_time:.2f} seconds to process dataset: {dataset_name},  with availability: {availability}, stability: {stability}')

if __name__ == "__main__":
    main()