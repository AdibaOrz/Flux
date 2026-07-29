# data_loader/General_Dataset.py

from torch.utils.data import Dataset


class General_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        input_data, exists_mask, class_label = self.dataset[idx]
        return input_data, exists_mask, class_label