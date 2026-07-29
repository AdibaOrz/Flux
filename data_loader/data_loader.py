# data_loader/data_loader.py

import os
import time
import json
import torch
import pandas as pd



def load_client_tensor_data(dataset_name, client_id, args, split_type,
                            json_dataset_path_configs="./data_preprocessor/dataset_paths.json"):
    """ Load tensor data for a specific client and split type (train/val/test) """
    with open(json_dataset_path_configs, 'r') as f:
        dataset_path_configs = json.load(f)

    simulated_dir = os.path.join(dataset_path_configs[dataset_name]['simulated'],
                                 f"avail:{args.availability}/stab:{args.stability}")
    full_path = f'{simulated_dir}/client_data/{client_id}/{split_type}_tensors.pt'
    load_tic = time.time()
    try:
        tensor_data = torch.load(full_path)
        # print(f"Loaded {split_type} tensors for client {client_id} from {full_path} in {time.time() - load_tic:.2f} seconds.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {full_path}")
    return tensor_data


## Used in simulate_incompleteness.py
def load_unscaled_entire_dataset(dataset_name, json_dataset_path_configs="./data_preprocessor/dataset_paths.json"):
    """ Load entire (includes all users) dataset for the given dataset name """
    start_load_time = time.time()
    with open(json_dataset_path_configs, 'r') as f:
        dataset_path_configs = json.load(f)

    dir_to_processed_file = dataset_path_configs[dataset_name]['processed']
    # pamap2_all_data_unscaled_100Hz.csv
    unified_sampling_rates = {'pamap2': 100, 'realworldhar': 50, 'sleepedf20': 100}
    dataset_sampling_rate = unified_sampling_rates[dataset_name]
    full_path = f"{dir_to_processed_file}/{dataset_name}_all_data_unscaled_{dataset_sampling_rate}Hz.csv"
    try:
        full_df = pd.read_csv(full_path)
    except Exception as e:
        print(f"No such file at path: {full_path}.\nDetailed error: {e}")
        exit(1)
    print(f"Took {time.time()-start_load_time:.2f} seconds to load entire unscaled dataset for {dataset_name}.")

    return full_df, dataset_sampling_rate, dataset_path_configs[dataset_name]


