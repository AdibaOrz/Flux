import os
import io
import json
import zipfile
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Unzip RealWorldHAR dataset')
    parser.add_argument('--dataset_paths_file', type=str, default='../dataset_paths.json', help='Path to the dataset config file')
    return parser.parse_args()


def unzip_first_level(zip_file_content, user_zip_filename, output_dir):
    # Directory to unzip the user data ({output_path}/proband{proband_id}/data)
    unzipped_user_data_dir = os.path.join(output_dir, '/'.join(user_zip_filename.split('/')[:-1]))
    if not os.path.exists(unzipped_user_data_dir):
        os.makedirs(unzipped_user_data_dir)

    # Unzip files inside the user zip file (contains sensor position data)
    with zipfile.ZipFile(io.BytesIO(zip_file_content), 'r') as zip_ref:
        for sensor_position_filename in zip_ref.namelist():
            extracted_path = os.path.join(unzipped_user_data_dir, sensor_position_filename)
            with zip_ref.open(sensor_position_filename) as source, open(extracted_path, 'wb') as target:
                target.write(source.read())

    return unzipped_user_data_dir


def unzip_second_level(dir_to_unzipped_user_files, curr_filename):
    full_path_to_second_level_zip_file = os.path.join(dir_to_unzipped_user_files, curr_filename)
    with zipfile.ZipFile(full_path_to_second_level_zip_file, 'r') as zip_ref:
        for sensor_position_filename in zip_ref.namelist():
            if sensor_position_filename.endswith('csv'):
                zip_ref.extract(sensor_position_filename, dir_to_unzipped_user_files)
    os.remove(full_path_to_second_level_zip_file)



def unzip_csv_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_to_org_zip_file = os.path.join(input_dir, 'realworld2016_dataset.zip')
    with zipfile.ZipFile(path_to_org_zip_file, 'r') as parent_zip_ref:
        for user_zip_filename in parent_zip_ref.namelist(): # list of all files in the realworld2016_dataset.zip
            # We select only files with acc or gyr and that are csv.zip files (there are sqlite extensions too)
            if user_zip_filename.lower().endswith('csv.zip') and any(substr in user_zip_filename.lower() for substr in ['acc', 'gyr']):
                user_zip_file_content = parent_zip_ref.read(user_zip_filename)
                dir_to_unzipped_user_files = unzip_first_level(user_zip_file_content, user_zip_filename, output_dir)

                # For users that have a second level of zipped files (need to unzip & delete the second level zip file)
                extracted_filenames = os.listdir(dir_to_unzipped_user_files) # Extracted files; some are inner-level new zip files
                for curr_filename in extracted_filenames:
                    if curr_filename.endswith('.zip'): # Second level zip file
                        unzip_second_level(dir_to_unzipped_user_files, curr_filename)


def main():
    args = parse_arguments()
    dataset_paths_file = args.dataset_paths_file
    with open(dataset_paths_file, 'r') as f:
        dataset_paths = json.load(f)

    input_dir = dataset_paths['realworldhar']['raw_unzipped']
    output_dir = dataset_paths['realworldhar']['raw']
    unzip_csv_files(input_dir, output_dir)


if __name__ == "__main__":
    main()