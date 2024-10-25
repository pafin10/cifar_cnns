# utils/data_loader.py
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        # Load data from CIFAR batches
        data, labels = [], []
        for i in range(1, 6):
            with open(f"{self.data_dir}/data_batch_{i}", "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                data.append(batch[b"data"])
                labels += batch[b"labels"]
        data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        return data, np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(data_dir, batch_size, transform=None):
    dataset = CIFAR10Dataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
