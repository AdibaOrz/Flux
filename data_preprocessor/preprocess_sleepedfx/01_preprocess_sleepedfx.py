"""
Credits to: https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py
"""

import os
import glob
import json
import math
import time
import argparse
import warnings
import dhedfreader
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess Sleep-EDFX data')
    parser.add_argument("--path_config_file", type=str, default="../dataset_paths.json")
    parser.add_argument("--scaling_method", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use (standard, minmax)")
    parser.add_argument("--steps", type=list,
                        # default=["each_user", "merge", "train_test_split"])
                        default=["merge"])
                        # default=["each_user", "merge", "train_test_split"])
    parser.add_argument("--silent", action="store_true", help="Do not show verbose messages")
    return parser.parse_args()


def process_each_user(path_to_raw_data, path_to_processed_data):
    path_to_processed_data = f"{path_to_processed_data}/each_user"
    if not os.path.exists(path_to_processed_data):
        print(f"Creating directory to save processed user files: {path_to_processed_data}")
        os.makedirs(path_to_processed_data)

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(path_to_raw_data, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(path_to_raw_data, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    print(f"Total users to be handled: {len(psg_fnames)}")

    for i in tqdm(range(len(psg_fnames))):
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']

        file_identifier = psg_fnames[i].split('/')[-1].split('-')[0]
        curr_user_id = psg_fnames[i].split('/')[-1].split('-')[0][3:5]

        raw_ch_df = raw.to_data_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        raw_ch_df.drop(columns=['time', 'Temp rectal', 'Event marker'], inplace=True)
        column_list = raw_ch_df.columns.tolist()
        # Get raw header
        f = open(psg_fnames[i], 'r', encoding='iso-8859-1')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fnames[i], 'r', encoding='iso-8859-1')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = list(zip(*reader_ann.records()))
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []  # indices of the data that will be removed
        labels = []  # indices of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            ann_str = ann_str.strip('b\'"')
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                label_idx.append(idx)

                print("Include onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                remove_idx.append(idx)

                print("Remove onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
        labels = np.hstack(labels)

        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                select_idx = select_idx[:-n_trims]
                labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            if len(labels) == 0:
                print("!!! Cannot proceed with this data as the len(labels) = 0")
                continue

        raw_ch = raw_ch_df.values[select_idx]
        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print(("Data before selection: {}, {}".format(x.shape, y.shape)))
        x = x[select_idx]
        y = y[select_idx]
        print(("Data after selection: {}, {}".format(x.shape, y.shape)))

        reshaped_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x_df = pd.DataFrame(reshaped_x, columns=column_list)
        y_repeated = np.repeat(y, x.shape[1])
        x_df['label'] = y_repeated
        x_df['user'] = curr_user_id

        # Save the data
        save_path = f"{path_to_processed_data}/{file_identifier}.csv"
        x_df.to_csv(save_path, index=False)

        print(f"\n\n=>=> Finished processing user {curr_user_id} / FileID: {file_identifier} <=<=\n\n")


def merge_all_files(path_to_processed_data):
    path_to_processed_individual_files = f"{path_to_processed_data}/each_user"
    all_available_csvs = os.listdir(path_to_processed_individual_files)
    print(len(all_available_csvs))

    df_list = []
    start_time = time.time()
    for csv_filename in all_available_csvs:
        df = pd.read_csv(f"{path_to_processed_individual_files}/{csv_filename}")
        df_list.append(df.copy())

    print(f"Took {time.time() - start_time} seconds to read all data")
    # Reading all files takes around 5 minutes


    start_merge = time.time()
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Took {time.time() - start_merge} seconds to merge all data")
    # Merging all dataframes takes around 12 seconds

    # Change the column "user" to "user_id"
    merged_df.rename(columns={'user': 'user_id'}, inplace=True)

    start_saving = time.time()
    path_to_save_merged_file = f"{path_to_processed_data}/sleepedfx_all_data_unscaled_100Hz.csv"
    merged_df.to_csv(path_to_save_merged_file, index=False)
    print(f"Saved the merged file to {path_to_save_merged_file}.\nTook {time.time() - start_saving} seconds to save the merged file")



def train_test_split_and_scale(path_to_processed_data, scaling_method, train_ratio=0.7):
    start_time = time.time()
    path_to_merged_csv = f"{path_to_processed_data}/merged.csv"
    merged_df = pd.read_csv(path_to_merged_csv)
    print("Took {} seconds to read the merged file".format(time.time() - start_time))

    feature_columns = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental']

    all_train_df = pd.DataFrame(columns=merged_df.columns)
    all_test_df = pd.DataFrame(columns=merged_df.columns)

    single_data_point_len = 30 * 100  # 30 * 100 sampling_rate => 30 seconds
    unique_client_ids = merged_df['user_id'].unique()

    for user_id in unique_client_ids:
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
            print(
                f"len(curr_train_df) in sec: {len(curr_train_df) / single_data_point_len}, len(curr_test_df) in sec: {len(curr_test_df) / single_data_point_len}")
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

    # Save the train and test data
    start_saving = time.time()
    path_to_save_train = f"{path_to_processed_data}/sleepedfx_{scaling_method}_scaled_train.csv"
    path_to_save_test = f"{path_to_processed_data}/sleepedfx_{scaling_method}_scaled_test.csv"
    all_train_df.to_csv(path_to_save_train, index=False)
    all_test_df.to_csv(path_to_save_test, index=False)

    print(f"Saved scaled train and test data. Took {time.time() - start_saving} seconds")



def main():
    args = parse_arguments()
    steps = args.steps
    print(f"Steps to do: {steps}")
    with open(args.path_config_file, "r") as f:
        paths = json.load(f)
    dataset_path = paths["sleepedfx"]

    path_to_raw_data = dataset_path["raw"]
    path_to_raw_data = f"{path_to_raw_data}/sleep-edf-database-expanded-1.0.0/sleep-cassette"

    path_to_processed_data = dataset_path["processed"]
    if not os.path.exists(path_to_processed_data):
        os.makedirs(path_to_processed_data)

    # 1. Process for each user
    if "each_user" in steps:
        start_time = time.time()
        process_each_user(path_to_raw_data, path_to_processed_data)
        print(f"Took {time.time() - start_time} seconds to process all users.")

    if "merge" in steps:
        start_time = time.time()
        merge_all_files(path_to_processed_data)
        print(f"Took {time.time() - start_time} seconds to merge all files.")

    if "train_test_split" in steps:
        start_time = time.time()
        train_test_split_and_scale(path_to_processed_data, args.scaling_method)
        print(f"Took {time.time() - start_time} seconds to split and scale data.")



if __name__ == "__main__":
    main()

