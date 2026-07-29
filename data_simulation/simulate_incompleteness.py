"""
Simulate multidimensional sensor modality incompleteness.
- Sensor availability: whether a sensor is available or not. (static missingness)
- Sensor stability: whether a sensor is stable or not. (dynamic missingness)
"""

import os
import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.append('../')
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

from utils.utils import set_seed
from data_loader.data_loader import load_unscaled_entire_dataset
from configs import load_dataset_configs, get_sensor_to_featureColumn_mapping



def parse_arguments():
    parser = argparse.ArgumentParser(description='Missingness modeling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset_name', type=str, default='pamap2',
                        choices=['pamap2', 'realworldhar', 'sleepedf20'], help='Dataset name.')
    parser.add_argument('--availability', type=str, default='low', choices=['low', 'medium', 'complete'],
                        help='Sensor availability level. `complete` means all sensors are available.')
    parser.add_argument('--stability', type=str, default='low', choices=['low', 'medium', 'complete'],
                        help='Sensor stability level. `complete` means all sensors are stable.')
    parser.add_argument('--scaling_method', type=str, default='standard', choices=['standard', 'minmax'],
                        help='Scaling method for the dataset, default is standard scaling ')
    args = parser.parse_args()
    return args


def get_beta_parameters(level):
    """Extract (alpha, beta) parameters for Beta distribution based on the given level."""
    assert level in ['low', 'medium'], f'Unknown level: {level}'
    level_to_beta_params = {
        'low': {'alpha': 45, 'beta': 45},
        'medium': {'alpha': 45, 'beta': 20},
    }
    return level_to_beta_params[level]


def get_stability_parameters(dataset_name, level):
    """Extract stability parameters for the given dataset and level."""
    assert dataset_name in ['pamap2', 'realworldhar', 'sleepedf20'], f'Unknown dataset: {dataset_name}'
    assert level in ['low', 'medium'], f'Unknown level: {level}'
    stability_params = {
        'pamap2': {
                'medium': {'p_11': 0.9999, 'p_00': 0.9997 },
                'low':    {'p_11': 0.9999, 'p_00': 0.9999 },
        },
        'realworldhar': {
            'medium': {'p_11': 0.9999, 'p_00': 0.9997},  # P(present) ≈ 75%
            'low':    {'p_11': 0.9999, 'p_00': 0.9999},  # P(present) ≈ 50%
        },
        'sleepedf20': {
                'medium': {'p_11': 0.99999, 'p_00': 0.99998 },
                'low':    {'p_11': 0.99999, 'p_00': 0.99999 },
        }
    }
    p_11 = stability_params[dataset_name][level]['p_11']
    p_00 = stability_params[dataset_name][level]['p_00']
    return p_11, p_00


def order_columns(df, dataset_configs):
    """Order columns based on the dataset name + Leave only feature columns, user_id and label columns.
    Re-order dataframe columns so that:
        - 0~num_feature_cols-1 are sensor feature columns.
        - num_feature_cols is the 'user_id' column.
        - num_feature_cols+1 is the 'label' column.
    """
    feature_cols = dataset_configs['feature_columns']
    df = df[feature_cols + ['user_id', 'label']]
    return df


def generate_markov_missingness_control_both(num_rows, p_present_to_present, p_missing_to_missing, verbose=False):
    p_present_to_missing = 1 - p_present_to_present
    p_missing_to_present = 1 - p_missing_to_missing

    stability_mask = np.ones(num_rows, int)
    for i in range(1, len(stability_mask)):
        if stability_mask[i - 1]:  # if previous value is present
            stability_mask[i] = np.random.rand() > p_present_to_missing
        else:  # if previous value is missing
            stability_mask[i] = np.random.rand() < p_missing_to_present

    if verbose:
        logging.info('Data availability (1: available, 0: missing)\n',
              ''.join([str(int(x)) for x in stability_mask.astype(int)]))
    return stability_mask


def simulate_client_modality_incompleteness(client_df, sensor_availability_list, sensor_to_feature_col_mapping, p_11, p_00):
    """Simulate modality incompleteness for a given client."""
    incomplete_client_df = client_df.copy()

    ## 1. Static missingness
    available_sensor_list = []
    # sensor_availability_list = client_metadata['sensor_availability_list']
    for sensor_idx, sensor_availability in enumerate(sensor_availability_list):
        if sensor_availability == 0:
            feature_col_indices = sensor_to_feature_col_mapping[sensor_idx]
            incomplete_client_df.iloc[:, feature_col_indices] = np.nan
        else:
            available_sensor_list.append(sensor_idx)
    # Debugging
    # print(f"When sensor availability is {sensor_availability_list}")
    # print(f"client_df after static missingness:\n{incomplete_client_df.head(5)}")

    ## 2. Dynamic missingness
    if (p_11 is None) and (p_00 is None):
        logging.info('No dynamic missingness modeling.')
        return incomplete_client_df

    num_rows = len(incomplete_client_df)
    for sensor_idx in available_sensor_list:
        feature_col_indices = sensor_to_feature_col_mapping[sensor_idx]
        stability_mask = generate_markov_missingness_control_both(num_rows,
                                                                p_present_to_present=p_11, p_missing_to_missing=p_00)
        if len(feature_col_indices) == 1:
            stability_mask = stability_mask[:, np.newaxis]
        else:
            stability_mask = np.tile(stability_mask[:, np.newaxis], (1, len(feature_col_indices)))

        # where replaces the values with np.nan if the condition is False (i.e., 0)
        incomplete_client_df.iloc[:, feature_col_indices] = incomplete_client_df.iloc[:,
                                                feature_col_indices].where(stability_mask == 1, # if 1, keep the value
                                                                           np.nan) # else, replace with np.nan

    return incomplete_client_df


def split_train_val_test(client_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the dataframe into train, validation, and test based on the ratios while maintaining class balance."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    train_df = pd.DataFrame(columns=client_df.columns.tolist())
    val_df = pd.DataFrame(columns=client_df.columns.tolist())
    test_df = pd.DataFrame(columns=client_df.columns.tolist())

    unique_labels = client_df['label'].unique()
    for label in unique_labels:
        curr_label_df = client_df[client_df['label'] == label]
        num_samples = len(curr_label_df)

        # Calculate split indices
        num_train_samples = int(train_ratio * num_samples)
        num_val_samples = int(val_ratio * num_samples)
        # Remaining samples go to test to ensure we use all data
        num_test_samples = num_samples - num_train_samples - num_val_samples

        # Split the data
        train_df = pd.concat([train_df, curr_label_df.iloc[:num_train_samples, :]], axis=0)
        val_df = pd.concat([val_df, curr_label_df.iloc[num_train_samples:num_train_samples + num_val_samples, :]],
                           axis=0)
        test_df = pd.concat([test_df, curr_label_df.iloc[num_train_samples + num_val_samples:, :]], axis=0)

    return train_df, val_df, test_df


def scale_client_train_val_test_data(train_df, val_df, test_df, scaling_method, dataset_configs):
    """Scale the train, validation, and test dataframes based on the scaling_method.
    Validation and test sets are scaled using statistics from the training set only."""
    if scaling_method == 'minmax':
        scaler = MinMaxScaler(range(-1, 1))
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}")

    scaled_train_df, scaled_val_df, scaled_test_df = train_df.copy(), val_df.copy(), test_df.copy()
    feature_columns = dataset_configs['feature_columns']

    for idx, feature_col in enumerate(feature_columns):
        # Check if every value in this column is nan in training set
        if scaled_train_df[feature_col].isnull().all():
            logging.info(f"Feature column {feature_col} is all missing values in training set. Skipping scaling.")
            continue

        # Fit scaler on training data only (non-NaN values)
        train_mask = scaled_train_df[feature_col].notna()
        scaler.fit(scaled_train_df.loc[train_mask, feature_col].values.reshape(-1, 1))

        # Transform training data
        scaled_values_train = scaler.transform(scaled_train_df.loc[train_mask, feature_col].values.reshape(-1, 1))
        scaled_train_df.loc[train_mask, feature_col] = scaled_values_train

        # Transform validation data using training statistics
        val_mask = scaled_val_df[feature_col].notna()
        if np.sum(val_mask) > 0:
            scaled_values_val = scaler.transform(scaled_val_df.loc[val_mask, feature_col].values.reshape(-1, 1))
            scaled_val_df.loc[val_mask, feature_col] = scaled_values_val
        else:
            logging.info(f"Feature column {feature_col} is all missing values in validation set. Skipping scaling.")

        # Transform test data using training statistics
        test_mask = scaled_test_df[feature_col].notna()
        if np.sum(test_mask) > 0:
            scaled_values_test = scaler.transform(scaled_test_df.loc[test_mask, feature_col].values.reshape(-1, 1))
            scaled_test_df.loc[test_mask, feature_col] = scaled_values_test
        else:
            logging.info(f"Feature column {feature_col} is all missing values in test set. Skipping scaling.")

    return scaled_train_df, scaled_val_df, scaled_test_df


def save_processed_data(scaled_df, data_type, args, path_configs):
    simulated_dir = path_configs['simulated'] + f'/avail:{args.availability}/stab:{args.stability}'
    if not os.path.exists(simulated_dir):
        os.makedirs(simulated_dir)
        logging.info(f"Created directory: {simulated_dir}")

    save_filename = f"{args.dataset_name}_{data_type}_{args.scaling_method}_scaled.csv"
    save_path = os.path.join(simulated_dir, save_filename)
    scaled_df.to_csv(save_path, index=False)
    logging.info(f"Saved {data_type} data to: {save_path}")


def main():
    args = parse_arguments()
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    logging.info(f'Setting seed to : {args.seed}')
    set_seed(args.seed) # set seed for reproducibility

    ## Load full dataset
    full_df, sampling_rate, path_configs = load_unscaled_entire_dataset(args.dataset_name,
                                json_dataset_path_configs="../data_preprocessor/dataset_paths.json")

    ## Dataset configs & column re-ordering
    dataset_configs = load_dataset_configs(args.dataset_name)
    full_df = order_columns(full_df, dataset_configs)

    ## Client IDs and sensor list
    client_ids = dataset_configs['client_ids']
    sensor_list = dataset_configs['sensor_list']
    num_clients = len(client_ids)
    num_sensors = len(sensor_list)
    sensor_to_feature_col_mapping = get_sensor_to_featureColumn_mapping(args.dataset_name)

    ## Sensor availability and stability
    meta_dict = {client_id: {} for client_id in client_ids}
    p_availability_list = [None] * num_clients
    p_11, p_00 = None, None
    if args.availability != 'complete':  # low, medium
        availability_beta_params = get_beta_parameters(args.availability)
        p_availability_list = np.random.beta(availability_beta_params['alpha'], availability_beta_params['beta'],
                                             num_clients)

    if args.stability != 'complete':  # low, medium
        p_11, p_00 = get_stability_parameters(args.dataset_name, args.stability)

    logging.info(f"p_availability_list: {p_availability_list}")
    logging.info(f"client_ids: {client_ids}")


    for client_id, p_availability in zip(client_ids, p_availability_list):
        if p_availability is None:
            sensor_availability_list = np.ones(num_sensors) # all available
        else:
            sensor_availability_list = np.random.binomial(1, p_availability, num_sensors)
        # (1: available, 0: missing)
        meta_dict[client_id]['p_availability'] = p_availability
        meta_dict[client_id]['sensor_availability_list'] = sensor_availability_list.tolist()
        meta_dict[client_id]['num_available_sensors'] = np.sum(sensor_availability_list)

        logging.info(f"client {client_id}: "
              f"p_availability: {p_availability} "
              f"sensor_availability_list: {sensor_availability_list}")


    ### Process data for each client
    scaled_train_data, scaled_val_data, scaled_test_data = [], [], []
    client_data_map = {client_id: group for client_id, group in full_df.groupby('user_id')}
    for client_id in client_ids:
        client_df = client_data_map[client_id]
        logging.info(f'Processing client {client_id} with ({len(client_df) // sampling_rate // 60}) minutes of data.')
        incomplete_client_df = simulate_client_modality_incompleteness(client_df, meta_dict[client_id]['sensor_availability_list'],
                                                                       sensor_to_feature_col_mapping, p_11, p_00)
        # split current client data to train and test, scale it
        client_train_df, client_val_df, client_test_df = split_train_val_test(incomplete_client_df)
        logging.info(f"Client {client_id} - Train: {len(client_train_df)}, Val: {len(client_val_df)}, Test: {len(client_test_df)}")
        client_train_df, client_val_df, client_test_df = scale_client_train_val_test_data(client_train_df,
                                                    client_val_df, client_test_df, args.scaling_method, dataset_configs)

        scaled_train_data.append(client_train_df)
        scaled_val_data.append(client_val_df)
        scaled_test_data.append(client_test_df)

    scaled_train_data = pd.concat(scaled_train_data, ignore_index=True)
    scaled_val_data = pd.concat(scaled_val_data, ignore_index=True)
    scaled_test_data = pd.concat(scaled_test_data, ignore_index=True)

    ## Save the scaled data
    save_processed_data(scaled_train_data, 'train', args, path_configs)
    save_processed_data(scaled_val_data, 'val', args, path_configs)
    save_processed_data(scaled_test_data, 'test', args, path_configs)


if __name__ == '__main__':
    main()

