import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Constants
DATASET_NAME = 'realworldhar'
USER_LIST = [i + 1 for i in range(15)]
ACTIVITIES = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
ORIENTATIONS = ['chest', 'head', 'shin', 'upperarm', 'waist']
MODALITIES = ['acc', 'gyr', ]
SAMPLING_RATE = 50
second_level_modality = {
    'acc': 'acc',
    'gyr': 'Gyroscope',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess RealworldHAR dataset")
    parser.add_argument("--path_config_file", type=str, default="../dataset_paths.json")
    parser.add_argument("--scaling_method", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use (standard, minmax)")
    parser.add_argument("--silent", action="store_true", help="Do not show verbose messages")
    return parser.parse_args()


def isReadableFile(file_name):
    try:
        if not os.path.exists(file_name):
            return False
        elif not os.path.isfile(file_name):
            return False
        elif not os.access(file_name, os.R_OK):
            return False
        else:
            return True
    except IOError as ex:
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    return False


def load_group(filenames):
    loaded = []
    for name in filenames:
        data = pd.read_csv(name, header=0, usecols=[2, 3, 4]).values  # 2,3,4 -> x,y,z axis data
        data = np.asarray(data)
        loaded.append(data)
    return loaded


def load_dataset(raw_dir, user_id, modal, activity, orientation):
    file_path = raw_dir + f'/proband{user_id}/data/'
    filenames = []

    candidate1 = file_path + f'{second_level_modality[modal]}_{activity}_{orientation}.csv'
    candidate2 = file_path + f'{second_level_modality[modal]}_{activity}_2_{orientation}.csv'
    candidate3 = file_path + f'{second_level_modality[modal]}_{activity}_3_{orientation}.csv'

    if isReadableFile(candidate1):
        filenames += [candidate1]
    if isReadableFile(candidate2):
        filenames += [candidate2]
    if isReadableFile(candidate3):
        filenames += [candidate3]

    X = load_group(filenames)
    return X, filenames


def merge_dataset(X_acc_all_clients, X_gyro_all_clients, y_all_clients):
    acc_columns, gyro_columns = [], []
    for orientation in ORIENTATIONS:
        acc_columns.append(f'{orientation}_acc_x')
        acc_columns.append(f'{orientation}_acc_y')
        acc_columns.append(f'{orientation}_acc_z')

        gyro_columns.append(f'{orientation}_gyro_x')
        gyro_columns.append(f'{orientation}_gyro_y')
        gyro_columns.append(f'{orientation}_gyro_z')

    all_modality_client_data = []
    num_users = len(USER_LIST)

    for user_id in range(num_users):
        try:
            acc_data = X_acc_all_clients[user_id]
            gyro_data = X_gyro_all_clients[user_id]
            label = np.reshape(y_all_clients[user_id], (-1, 1)).astype(int)
            user_id = np.full((len(acc_data), 1), user_id).astype(int)

            # Create DataFrame for accelerometer and gyro data
            client_df_acc = pd.DataFrame(acc_data, columns=acc_columns)
            client_df_gyro = pd.DataFrame(gyro_data, columns=gyro_columns)

            # Ensure the gyroscope DataFrame is indexed correctly before join
            client_df_gyro.index = client_df_acc.index

            # Join the DataFrames on the index
            merged_df = client_df_acc.join(client_df_gyro, lsuffix='_acc', rsuffix='_gyro')
            merged_df['label'] = label
            merged_df['user_id'] = user_id

            all_modality_client_data.append(merged_df)
        except Exception as e:
            print(user_id, e)

    all_modality_all_client_data = pd.concat(all_modality_client_data, ignore_index=True)
    return all_modality_all_client_data


def divide_data_to_train_and_test(given_df, train_ratio=0.7):
    train_df = pd.DataFrame(columns=given_df.columns.tolist())
    test_df = pd.DataFrame(columns=given_df.columns.tolist())

    for uid in range(len(USER_LIST)):
        print(f"====User {uid}===")
        for lbl in range(len(ACTIVITIES)):
            curr_user_label_df = given_df[(given_df['user_id'] == uid) & (given_df['label'] == lbl)]
            total_num_seconds = int(len(curr_user_label_df) // SAMPLING_RATE)
            train_seconds = int(train_ratio * total_num_seconds)
            train_rows = int(train_seconds * SAMPLING_RATE)

            curr_train_df = curr_user_label_df.iloc[:train_rows, :]
            curr_test_df = curr_user_label_df.iloc[train_rows:, :]

            if len(train_df) == 0:
                train_df = curr_train_df.copy()
            else:
                train_df = pd.concat([train_df, curr_train_df], axis=0)

            if len(test_df) == 0:
                test_df = curr_test_df.copy()
            else:
                test_df = pd.concat([test_df, curr_test_df], axis=0)

            print(f'\tLabel:{lbl}. '
                  f'Train: {len(curr_train_df) // SAMPLING_RATE}, '
                  f'Test: {len(curr_test_df) // SAMPLING_RATE}')
    return train_df, test_df


def scale_train_and_test(train_df, test_df, scaling_method):
    # Scale
    feature_cols = [col for col in train_df.columns.tolist() if col not in ['label', 'user_id']]
    print(f'=====> Feature columns <=====\n{feature_cols}')

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown scaler: {scaling_method}")

    print(f'===== {scaling_method} scaling please wait... =====')
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df


def process_single_user(path_to_read_raw_data, uid):
    X_acc_curr_user, X_gyr_curr_user, y_curr_user = [], [], []
    for activity in ACTIVITIES:
        x_acc_by_orientation_dict = {orientation: list() for orientation in ORIENTATIONS}
        x_gyro_by_orientation_dict = {orientation: list() for orientation in ORIENTATIONS}

        for ori in ORIENTATIONS:
            # print(f"preprocessing {activity} at {ori}")
            x_acc_curr_ori, x_gyro_curr_ori = [], []
            # xAccList, xGyrList = [], []
            # Some users contain more than one file for the same activity, so we need to concatenate them
            loaded_acc_data_list, acc_filenames = load_dataset(path_to_read_raw_data, uid, 'acc', activity, ori)
            loaded_gyro_data_list, gyro_filenames = load_dataset(path_to_read_raw_data, uid, 'gyr', activity, ori)

            assert len(loaded_acc_data_list) != 0
            assert len(loaded_gyro_data_list) != 0

            for i in range(len(loaded_acc_data_list)):
                min_length_curr_ori = min([len(loaded_acc_data_list[i]), len(loaded_gyro_data_list[i])])
                loaded_acc_df = pd.DataFrame(loaded_acc_data_list[i])
                loaded_gyro_df = pd.DataFrame(loaded_gyro_data_list[i])
                loaded_acc_np = loaded_acc_df.to_numpy()
                loaded_gyro_np = loaded_gyro_df.to_numpy()

                x_acc_curr_ori.append(loaded_acc_np[:min_length_curr_ori])
                x_gyro_curr_ori.append(loaded_gyro_np[:min_length_curr_ori])

            x_acc_by_orientation_dict[ori].append(np.vstack(x_acc_curr_ori))
            x_gyro_by_orientation_dict[ori].append(np.vstack(x_gyro_curr_ori))

        min_length_curr_activity = min([len(length[0]) for length in list(x_acc_by_orientation_dict.values())])

        for orientation in ORIENTATIONS:
            x_acc_by_orientation_dict[orientation] = x_acc_by_orientation_dict[orientation][0][:min_length_curr_activity]
            x_gyro_by_orientation_dict[orientation] = x_gyro_by_orientation_dict[orientation][0][:min_length_curr_activity]

        X_acc_curr_user.append(np.hstack(list(x_acc_by_orientation_dict.values())))
        X_gyr_curr_user.append(np.hstack(list(x_gyro_by_orientation_dict.values())))
        y_curr_user.append(np.full(min_length_curr_activity, ACTIVITIES.index(activity)))

    return X_acc_curr_user, X_gyr_curr_user, y_curr_user


def process_all_users(path_to_read_raw_data):
    X_acc_all_clients = []
    X_gyro_all_clients = []
    y_all_clients = []

    for uid in USER_LIST:
        print(f"=============== Starting preprocessing for user {uid} =====================")
        x_acc_curr_client, x_gyro_curr_client, y_curr_client = process_single_user(path_to_read_raw_data, uid)
        X_acc_all_clients.append(np.vstack(x_acc_curr_client))
        X_gyro_all_clients.append(np.vstack(x_gyro_curr_client))
        y_all_clients.append(np.hstack(y_curr_client))

    return X_acc_all_clients, X_gyro_all_clients, y_all_clients

def save_unscaled_data(df, path_to_save_processed_data):
    save_path = f'{path_to_save_processed_data}/realworldhar_all_data_unscaled_{SAMPLING_RATE}Hz.csv'
    df.to_csv(save_path, index=False)
    print(f"Unscaled data saved to {save_path}")


def main():
    args = parse_arguments()

    path_config_file = args.path_config_file
    with open(path_config_file, 'r') as f:
        path_config_json = json.load(f)

    path_to_read_raw_data = path_config_json[DATASET_NAME]['raw']
    path_to_save_processed_data = path_config_json[DATASET_NAME]['processed']
    if not os.path.exists(path_to_save_processed_data):
        os.makedirs(path_to_save_processed_data)

    X_acc_all_clients, X_gyro_all_clients, y_all_clients = process_all_users(path_to_read_raw_data)
    combined_df = merge_dataset(X_acc_all_clients, X_gyro_all_clients, y_all_clients)
    save_unscaled_data(combined_df, path_to_save_processed_data)

    train_df, test_df = divide_data_to_train_and_test(combined_df.copy())
    train_df, test_df = scale_train_and_test(train_df.copy(), test_df.copy(), args.scaling_method)

    train_save_path = f"{path_to_save_processed_data}/realworldhar_train_{args.scaling_method}_scaled_{SAMPLING_RATE}Hz.csv"
    test_save_path = f"{path_to_save_processed_data}/realworldhar_test_{args.scaling_method}_scaled_{SAMPLING_RATE}Hz.csv"
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f'>>> Saved train file to: {train_save_path}')
    print(f'>>> Saved test file to: {test_save_path}')


if __name__ == '__main__':
    main()
