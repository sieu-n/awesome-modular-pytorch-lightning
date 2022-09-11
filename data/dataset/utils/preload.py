import catalog
from torch.utils.data import Dataset
from tqdm import tqdm


class PreloadedDataset(Dataset):
    """
    Precompute the entire dataset and cache it. This is useful for speeding up
    training when the entire dataset can be cached. This can be used as a wrapper around
    the base dataset in the config file.
    """

    def __init__(self, name, args={}):
        base_dataset = catalog.dataset.build(
            name=name,
            args=args,
        )
        print("`PrecomputeDataset` is loading memory.")
        self.memory = [base_dataset[x] for x in tqdm(range(len(base_dataset)))]

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)
