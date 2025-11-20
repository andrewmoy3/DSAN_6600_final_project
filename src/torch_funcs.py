import torch
from torch.utils.data import Dataset, DataLoader
# Instantiates all PyTorch related code

# have to define our own dataset class that inherits from PyTorch Dataset
class ImageDataset(Dataset):
    # data = list of tuples (filename, multi-hot label)
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        return img_path, torch_funcs.tensor(label, dtype=torch_funcs.float32)

    def __len__(self):
        return len(self.data)