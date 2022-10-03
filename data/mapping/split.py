import random

from torch.utils.data import Dataset
from tqdm import tqdm

"""
Datasets that simply remap indices and sample data in a different order.
"""


class RemapIndices(Dataset):
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.base_dataset[idx]


class ClassBalanceTrainValSplit(RemapIndices):
    """
    Split base dataset into two subsets and use one of them. The validation set
    is chosen to have the same number of samples on each class.
    """

    def __init__(self, base_dataset, subset, count, seed=42, const_cfg=None):
        rng = random.Random(seed)
        self.base_dataset = base_dataset
        dataset_size = len(base_dataset)

        print("Loading dataset for splitting validation set based on class type.")
        samples_of_class = {}
        for idx, sample in tqdm(enumerate(base_dataset)):
            c = sample["labels"]
            if c not in samples_of_class:
                samples_of_class[c] = []
            samples_of_class[c].append(idx)

        # select validation set.
        val_indices = []
        for c in samples_of_class.keys():
            val_indices += rng.sample(samples_of_class[c], count)

        if subset in ["trn", "train"]:
            # train indices is the complement of val indices.
            self.indices = list(set(dataset_size) - set(val_indices))
        elif subset in ["val"]:
            self.indices = val_indices
        else:
            raise ValueError(f"Invalid subset type: {subset} was given.")

    def select_random_indices(self, dataset_size, count, seed=None):
        indices = list(range(dataset_size))
        if seed is None:
            r = random
        else:
            r = random.Random(seed)
        return r.sample(indices, count)


class TrainValSplit(RemapIndices):
    """
    Split base dataset into two subsets and use one of them based on the `subset` value.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        subset: str,
        seed: int = 42,
        count=None,
        ratio=None,
        indices=None,
        const_cfg=None,
    ):
        self.base_dataset = base_dataset
        if indices is not None:
            self.indices = indices
            return

        assert (
            count is not None or ratio is not None
        ), f"val_count: {count}, val_ratio: {ratio}. Only \
            one should be given."
        assert (
            count is None and ratio is None
        ), f"val_count: {count}, val_ratio: {ratio}. Only \
            one should be given."

        dataset_size = len(base_dataset)
        if ratio is not None:
            count = round(ratio * dataset_size)

        val_indices = self.select_random_indices(
            dataset_size=dataset_size, count=count, seed=seed
        )

        if subset in ["trn", "train"]:
            # train indices is the complement of val indices.
            self.indices = list(set(range(dataset_size)) - set(val_indices))
        elif subset in ["val"]:
            self.indices = val_indices
        else:
            raise ValueError(f"Invalid subset type: {subset} was given.")

    def select_random_indices(self, dataset_size, count, seed=None):
        indices = list(range(dataset_size))
        if seed is None:
            r = random
        else:
            r = random.Random(seed)
        return r.sample(indices, count)


class KFoldValidation(RemapIndices):
    def __init__(
        self,
        base_dataset,
        subset: str,
        fold_idx: int,
        max_folds: int,
        seed=42,
        const_cfg=None,
    ):
        assert 0 <= fold_idx < max_folds
        self.base_dataset = base_dataset
        dataset_size = len(base_dataset)
        val_indices = self.select_random_indices(
            dataset_size, fold_idx, max_folds, seed=seed
        )

        if subset in ["trn", "train"]:
            # train indices is the complement of val indices.
            self.indices = list(set(range(dataset_size)) - set(val_indices))
        elif subset in ["val"]:
            self.indices = val_indices
        else:
            raise ValueError(f"Invalid subset type: {subset} was given.")

    def select_random_indices(self, dataset_size, fold_idx, max_folds, seed):
        # shuffle and slice
        indices = list(range(dataset_size))

        random.Random(seed).shuffle(indices)
        idx_start = int(dataset_size * fold_idx / max_folds)
        idx_end = int(dataset_size * (fold_idx + 1) / max_folds)
        return indices[idx_start:idx_end]


class SubsetDataset(RemapIndices):
    def __init__(
        self, base_dataset, indices=None, size=None, seed=None, const_cfg=None
    ):
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
