from torch.utils.data import Dataset
from tqdm import tqdm


class PrecomputeDataset(Dataset):
    """
    Precompute the entire dataset and cache it. This is useful for speeding up
    training when the entire dataset can be cached. Be aware to apply data
    augmentations after applying this dataset.
    """

    def __init__(self, base_dataset, const_cfg=None):
        print("`PrecomputeDataset` is loading memory.")
        self.memory = [base_dataset[x] for x in tqdm(range(len(base_dataset)))]

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)
