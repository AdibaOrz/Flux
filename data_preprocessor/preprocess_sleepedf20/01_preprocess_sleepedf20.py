"""
Is dependent on the merged.csv from the sleep-edfx (entire) dataset.
"""
import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


TWENTY_USER_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
SAMPLING_RATE = 100  # Hz


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess the sleep-edf-20 dataset.')
    parser.add_argument("--scaling_method", type=str, default='standard',
                        choices=['standard', 'minmax'], help='Type of scaler to use.')
    parser.add_argument('--silent', action='store_true', help='Do not show verbose messages')
    return parser.parse_args()


def read_merged_all_user_csv(dataset_paths):
    # Read the merged.csv from sleepedfx
    merged_all_user_csv_path = os.path.join(dataset_paths['sleepedfx']['processed'],
                                            'sleepedfx_all_data_unscaled_100Hz.csv')
    merged_df = pd.read_csv(merged_all_user_csv_path)
    user_list = merged_df['user_id'].unique()
    print(f"User list [{len(user_list)}]:\n{user_list}")
    return merged_df


def train_test_split_and_scale(merged_df, scaling_method, dataset_paths, train_ratio=0.7):
    feature_columns = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental']

    all_train_df = pd.DataFrame(columns=merged_df.columns)
    all_test_df = pd.DataFrame(columns=merged_df.columns)

    single_data_point_len = 30 * SAMPLING_RATE  # 30 * 100 sampling_rate => 30 seconds
    unique_client_ids = merged_df['user_id'].unique()

    for user_id in tqdm(unique_client_ids, desc="User loop"):
        print(f"Processing user: {user_id}")
        curr_user_df = merged_df[merged_df['user_id'] == user_id]
        unique_labels = curr_user_df['label'].unique()
        for label in unique_labels:
            curr_user_and_label = curr_user_df[curr_user_df['label'] == label]
            total_datapoint_size = len(curr_user_and_label) / single_data_point_len
            num_train_datapoints = int(train_ratio * total_datapoint_size)
            num_train_rows = num_train_datapoints * single_data_point_len
            curr_train_df = curr_user_and_label[:num_train_rows]
            curr_test_df = curr_user_and_label[num_train_rows:]
            print(f"len(curr_train_df): {len(curr_train_df)}, len(curr_test_df): {len(curr_test_df)}")
            print(f"len(curr_train_df) in sec: {len(curr_train_df) / single_data_point_len}, len(curr_test_df) in sec: {len(curr_test_df) / single_data_point_len}")
            if len(all_train_df) == 0:
                all_train_df = curr_train_df.copy()
            else:
                all_train_df = pd.concat([all_train_df, curr_train_df.copy()], axis=0, ignore_index=True)

            if len(all_test_df) == 0:
                all_test_df = curr_test_df.copy()
            else:
                all_test_df = pd.concat([all_test_df, curr_test_df.copy()], axis=0, ignore_index=True)

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError("Invalid scaling method")

    start_scaling = time.time()
    all_train_df[feature_columns] = scaler.fit_transform(all_train_df[feature_columns])
    all_test_df[feature_columns] = scaler.transform(all_test_df[feature_columns])

    print(f"Finished scaling data. Took: {time.time() - start_scaling}")

    # Save train and test data
    start_saving = time.time()
    path_to_processed_dir = dataset_paths['sleepedf20']['processed']
    if not os.path.exists(path_to_processed_dir):
        os.makedirs(path_to_processed_dir)

    path_to_save_train = os.path.join(path_to_processed_dir, f'sleepedf20_{scaling_method}_scaled_train.csv')
    path_to_save_test = os.path.join(path_to_processed_dir, f'sleepedf20_{scaling_method}_scaled_test.csv')

    all_train_df.to_csv(path_to_save_train, index=False)
    all_test_df.to_csv(path_to_save_test, index=False)
    print(f"Finished saving train and test data. Took: {time.time() - start_saving}")


def save_unscaled_merged_data(merged_df, path_to_save):
    path_to_save_unscaled = os.path.join(path_to_save, 'sleepedf20_all_data_unscaled_100Hz.csv')
    merged_df.to_csv(path_to_save_unscaled, index=False)
    print(f'Saved unscaled data to {path_to_save_unscaled}')

def main():
    args = parse_arguments()

    # 1. Load dataset paths (both sleepedfx and sleepedf20 can be accessed)
    json_path = "../dataset_paths.json"
    with open(json_path, 'r') as f:
        dataset_paths = json.load(f)

    # 2. Read the merged csv file (as a result of preprocessing of sleepedfx dataset)
    merged_df = read_merged_all_user_csv(dataset_paths)

    # 3. Select only the 20 users
    selected_20user_df = merged_df[merged_df['user_id'].isin(TWENTY_USER_IDS)]
    print(f"merged_df has {len(merged_df)} rows, selected_20user_df has {len(selected_20user_df)} rows")

    # Save unscaled data
    save_unscaled_merged_data(selected_20user_df.copy(), dataset_paths['sleepedf20']['processed'])

    # 4. Split to train and test, scale and save to the path of sleepedf20
    train_test_split_and_scale(selected_20user_df, args.scaling_method, dataset_paths)



if __name__ == "__main__":
    main()


