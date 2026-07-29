# ml_federated/client/Base_Client.py

import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from data_loader.General_Dataset import General_Dataset
from data_loader.data_loader import load_client_tensor_data



class Base_Client(ABC):
    def __init__(self, client_id, dataset_configs, device, args):
        self.client_id = client_id
        self.dataset_configs = dataset_configs
        self.device = device
        self.args = args
        self.num_modalities = self.dataset_configs['num_sensors']
        self.local_epochs = self.dataset_configs['local_epochs']

        # Set up local train and test data loaders
        # Load train, val, test tensors for this client
        dataset_name = dataset_configs['name']
        self.sensor_to_feat_cols = dataset_configs['sensor_to_featureColumn']
        train_set = load_client_tensor_data(dataset_name, client_id, args, 'train')
        val_set = load_client_tensor_data(dataset_name, client_id, args, 'val')
        test_set = load_client_tensor_data(dataset_name, client_id, args, 'test')

        # Filter datasets and calculate statistics
        train_set, val_set, test_set = self.filter_and_analyze_datasets(train_set, val_set, test_set)

        train_set = General_Dataset(train_set)
        val_set = General_Dataset(val_set)
        test_set = General_Dataset(test_set)

        self.train_loader = DataLoader(train_set, batch_size=dataset_configs['local_bs'], shuffle=True) if len(train_set) > 0 else None
        self.val_loader   = DataLoader(val_set,   batch_size=dataset_configs['local_bs'], shuffle=False) if len(val_set) > 0 else None
        self.test_loader  = DataLoader(test_set,  batch_size=dataset_configs['local_bs'], shuffle=False) if len(test_set) > 0 else None

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


    @abstractmethod
    def local_train(self, passed_model):
        """Abstract method for local training.
        Should be implemented in subclasses."""
        pass


    @abstractmethod
    def local_test(self, model, eval_on):
        """Abstract method for local testing.
        Should be implemented in subclasses."""
        assert eval_on in ['val', 'test'], "eval_on must be either 'val' or 'test'"
        pass


    @abstractmethod
    def _set_model_mode(self, model, mode):
        """Set the model to train or eval mode."""
        """Abstract method for setting model mode.
        Should be implemented in subclasses."""
        assert mode in ['train', 'eval'], "mode must be either 'train' or 'eval'"
        pass


    def get_data_size(self, data_type):
        assert data_type in ['train', 'val', 'test'], "data_type must be either 'train', 'val', or 'test'"
        return self.data_sizes[data_type]


    def get_modality_data_size(self, data_type):
        """Get the number of samples for a specific modality in the given data type."""
        assert data_type in ['train', 'val', 'test'], "data_type must be either 'train', 'val', or 'test'"
        return self.modality_data_sizes[data_type]



    def _get_sample_modality_info(self, mask):
        modality_presence = []
        num_modals_in_sample = 0

        for m in range(self.num_modalities):
            feat_cols = self.sensor_to_feat_cols[m]
            curr_modality_mask = mask[feat_cols, :]
            # Mark modality as absent if all values are are absent in a given window
            is_m_present = 0 if torch.all(curr_modality_mask == 0) else 1
            modality_presence.append(is_m_present)
            num_modals_in_sample += is_m_present

        return modality_presence, num_modals_in_sample


    def filter_and_analyze_datasets(self, train_set, val_set, test_set):
        self.modality_data_sizes = {'train': [], 'val': [], 'test': []}
        self.data_sizes = {'train': 0, 'val': 0, 'test': 0}

        filtered_datasets = []
        for dataset, key in zip([train_set, val_set, test_set], ['train', 'val', 'test']):
            filtered_dataset = []
            modality_counts = [0] * self.num_modalities
            for sample_idx, (x, mask, label) in enumerate(dataset):
                modality_presence, num_modals_in_sample = self._get_sample_modality_info(mask)

                # Exclude the samples where all modalities are absent
                if num_modals_in_sample > 0:
                    filtered_dataset.append((x, mask, label))

                    # Update modality counts
                    for m in range(self.num_modalities):
                        modality_counts[m] += modality_presence[m]

                    # Update the total sample count
                    self.data_sizes[key] += 1
            # Store modality stats
            self.modality_data_sizes[key] = modality_counts
            filtered_datasets.append(filtered_dataset)

            # Uncomment the following lines to print dataset size statistics
            original_size, filtered_size = len(dataset), len(filtered_dataset)
            # print(f"Dataset {key} - Original size: {original_size}, Filtered size: {filtered_size}")

        return tuple(filtered_datasets)

