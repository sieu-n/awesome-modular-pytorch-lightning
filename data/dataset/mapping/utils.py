from torch.utils.data import Dataset
import random


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices=None, size=None, seed=None):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        """
        self.base_dataset = base_dataset

        if indices is not None:
            self.indices = indices
        else:
            if seed:
                random_sampler = random.Random(seed).sample
            else:
                random_sampler = random.sample
            self.indices = random_sampler(range(len(base_dataset)), size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.base_dataset[idx]
