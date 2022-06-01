from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_features, num_channels):
        """
        Standard MLP model.
        Parameters
        ----------
        in_features: int
            Number of input channels.
        num_channels: list[int]
            Number of nodes in each intermediate layer.
        """
        super().__init__()

        self.FC = []
        prev_features = in_features
        for channels in num_channels:
            self.FC.append(nn.Linear(prev_features, channels))
            prev_features = channels
        self.FC = nn.ModuleList(self.FC)

        self.activation = nn.ReLU()

    def forward(self, feature):
        for idx, layer in enumerate(self.FC):
            feature = self.activation(layer(feature))

        return feature
