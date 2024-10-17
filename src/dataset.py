from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.dataset)
