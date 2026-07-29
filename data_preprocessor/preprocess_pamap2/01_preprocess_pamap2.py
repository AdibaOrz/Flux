import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Constants
DATASET_NAME = "pamap2"
SUBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 8] # 8 subjects, 9 is excluded (following other papers) as it has only single label
ACTIVITY_LIST = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess PAMAP2 dataset')
    parser.add_argument('--path_config_file', type=str, default='../dataset_paths.json')
    parser.add_argument('--scaling_method', type=str, default='standard', choices=['standard', 'minmax'])
    parser.add_argument('--silent', action='store_true', help='Do not show verbose messages')
    return parser.parse_args()


def load_all_user_data(path_to_raw_data, silent):
    user_files = os.listdir(f'{path_to_raw_data}/PAMAP2_Dataset/Protocol')
    colNames = ["timestamp", "activityID", "heartrate"]
    IMU_hand = ['handTemperature',
               'hand_acc_x', 'hand_acc_y', 'hand_acc_z',
               'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
               'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
               'handMagne1', 'handMagne2', 'handMagne3',
               'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

    IMU_chest = ['chestTemperature',
                'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
                'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                'chestMagne1', 'chestMagne2', 'chestMagne3',
                'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

    IMU_ankle = ['ankleTemperature',
                'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
                'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
                'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

    all_column_list = colNames + IMU_hand + IMU_chest + IMU_ankle
    if not silent:
        print(f'all_column_list with len ({len(all_column_list)}): {all_column_list}')

    target_columns = [0, 1, 4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]
    target_classes = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    df = pd.DataFrame()
    for user_file in user_files:
        full_path = f"{path_to_raw_data}/PAMAP2_Dataset/Protocol/{user_file}"
        cur_df = pd.read_table(full_path, header=None, sep='\s+')
        cur_df.columns = all_column_list
        cur_df = cur_df.iloc[:, target_columns]
        cur_df['user_idx'] = int(user_file[-5])
        cur_df = cur_df[cur_df['activityID'].isin(target_classes)]
        cur_df = cur_df.dropna()

        if len(df) == 0:
            df = cur_df.copy()
        else:
            df = pd.concat([df, cur_df], axis=0)

    df = df[df['user_idx'].isin(SUBJECT_LIST)] # Keep users in the subject list, user 9 will be filtered
    return df


def preprocess_data(df, silent):
    if not silent:
        for user_idx in SUBJECT_LIST:
            cur_user_df = df[df["user_idx"] == user_idx]
            unique_activity_ids = cur_user_df['activityID'].unique()
            if not silent:
                print(f"As for the user {user_idx}, unique activities available: {unique_activity_ids}")

    # Rename columns
    df = df.rename(columns={'activityID': 'label', 'user_idx': 'user_id'})

    # Let label start from 0
    unique_labels = np.unique(df["label"])
    mapping_dict = {val: idx for idx, val in enumerate(unique_labels)}
    df['label'] = df['label'].map(mapping_dict)
    return df


def save_unscaled_data(df, path_to_save_processed_data, silent):
    save_path = f"{path_to_save_processed_data}/pamap2_all_data_unscaled_100Hz.csv"
    df.to_csv(save_path, index=False)

    if not silent:
        print(f"All data (unscaled) is saved to {save_path}")


def divide_data_to_train_and_test(df, train_ratio, silent):
    """ Divide the data to train and test dataframes with class balance in mind """
    train_df = pd.DataFrame(columns=df.columns.tolist())
    test_df = pd.DataFrame(columns=df.columns.tolist())

    for uid in SUBJECT_LIST:
        curr_user_df = df[df['user_id'] == uid]
        unique_labels = curr_user_df['label'].unique()

        for label in unique_labels:
            curr_user_label_df = curr_user_df[curr_user_df['label'] == label]
            total_num_rows_cur_label = len(curr_user_label_df)
            num_train_rows = int(total_num_rows_cur_label * train_ratio)
            curr_train_df = curr_user_label_df.iloc[:num_train_rows]
            curr_test_df = curr_user_label_df.iloc[num_train_rows:]

            if len(train_df) == 0:
                train_df = curr_train_df.copy()
            else:
                train_df = pd.concat([train_df, curr_train_df], axis=0)
            if len(test_df) == 0:
                test_df = curr_test_df.copy()
            else:
                test_df = pd.concat([test_df, curr_test_df], axis=0)

            if not silent:
                print(f"User {uid}, label {label}: #train: {len(curr_train_df)}, #test: {len(curr_test_df)}")

    return train_df, test_df



def scale_train_and_test_data(train_df, test_df, scaling_method):
    """
    Scale the train and test dataframes
    """
    non_feature_cols = ['timestamp', 'label', 'user_id']
    feature_columns = [col for col in train_df.columns.tolist() if col not in non_feature_cols]
    print(f"Feature columns: {feature_columns}")

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}")

    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    return train_df, test_df


def save_train_and_test_data(train_df, test_df, path_to_save_processed_data, scaling_method, silent):
    train_save_path = f"{path_to_save_processed_data}/pamap2_train_{scaling_method}_scaled_100Hz.csv"
    test_save_path = f"{path_to_save_processed_data}/pamap2_test_{scaling_method}_scaled_100Hz.csv"

    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    if not silent:
        print(f"Train data saved to {train_save_path}")
        print(f"Test data saved to {test_save_path}")


def main():
    start_time = time.time()

    # 1. Get dataset-specific options (path, etc.)
    args = parse_arguments()
    path_config_file = args.path_config_file
    scaling_method = args.scaling_method
    silent = args.silent
    with open(path_config_file, 'r') as f:
        path_config_json = json.load(f)

    path_to_read_raw_data = path_config_json[DATASET_NAME]['raw']
    path_to_save_processed_data = path_config_json[DATASET_NAME]['processed']
    if not os.path.exists(path_to_save_processed_data):
        os.makedirs(path_to_save_processed_data)

    df = load_all_user_data(path_to_read_raw_data, silent)
    df = preprocess_data(df.copy(), silent)
    save_unscaled_data(df, path_to_save_processed_data, silent)

    all_train_df, all_test_df = divide_data_to_train_and_test(df.copy(), train_ratio=0.7, silent=silent)
    all_train_df, all_test_df = scale_train_and_test_data(all_train_df.copy(), all_test_df.copy(), scaling_method)
    save_train_and_test_data(all_train_df, all_test_df, path_to_save_processed_data, scaling_method, silent)

    print(f">>> All done.\nElapsed time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
