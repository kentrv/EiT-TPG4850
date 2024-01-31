from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class Custom3DDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __getitem__(self, index):
        data = np.load(self.data_paths[index])  # Adjust this based on your data format
        data = torch.tensor(data, dtype=torch.float32)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_paths)
