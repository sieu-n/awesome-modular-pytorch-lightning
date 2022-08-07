import numpy as np
from torch.utils.data import Dataset


class InstanceIndependantNoise(Dataset):
    def __init__(
        self,
        base_dataset,
        num_classes,
        method="custom",
        transition_matrix=None,
        transition_prob=None,
        *args,
        **kwargs,
    ):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        transition_matrix: list[list[float]]
            When g.t. class is i, the label is corrupted to j with a probability of transition_matrix[i][j].
        """
        self.base_dataset = base_dataset

        if method == "custom":
            assert transition_matrix is not None
            transition_matrix = np.array(transition_matrix)
            assert transition_matrix.shape == (num_classes, num_classes)
        elif method == "symmetric":
            assert transition_prob is not None
            normalized_prob = transition_prob / (num_classes - 1)

            transition_matrix = np.full((num_classes, num_classes), normalized_prob)
            for c in range(num_classes):
                transition_matrix[c][c] = 0
        elif method == "asymmetric":
            assert transition_prob is not None

            transition_matrix = np.zeros((num_classes, num_classes))
            for c in range(num_classes):
                next_class = (c + 1) % num_classes
                transition_matrix[c][next_class] = transition_prob
        else:
            raise ValueError(
                f"Invalid method '{method}' was provided to `InstanceIndependantNoise`."
            )
        # fill in diagonal indices so sum of transition_matrix[c] 1.0
        for c in range(num_classes):
            # sum of transition_matrix excluding p[c][c]
            s = sum(transition_matrix[c]) - transition_matrix[c][c]
            assert 0.0 <= s <= 1.0
            transition_matrix[c][c] = 1.0 - s
        print(
            f"Transitioning data based on transition matrix: \n{np.array(transition_matrix)}"
        )

        self.corruption_prob = np.cumsum(transition_matrix, axis=1)
        self.random_score = np.random.uniform(size=len(self.base_dataset))

    def corrupt_label(self, label, random_score):
        cumsum = self.corruption_prob[label]
        for idx, cum in enumerate(cumsum):
            if cum >= random_score:
                return idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        data["labels"] = self.corrupt_label(data["labels"], self.random_score[idx])
        return data
