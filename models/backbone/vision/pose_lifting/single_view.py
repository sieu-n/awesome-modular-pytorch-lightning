from torch import nn


class LBRD(nn.Module):
    """
    Linear - BN - ReLU - Dropout
    """
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResMLPBlock(nn.Module):
    def __init__(self, features=1024, dropout=0.5, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            LBRD(features, features, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        skip_con = x
        for layer in self.layers:
            x = layer(x)
        return skip_con + x


class PoseLiftingSingleFrameMLP(nn.Module):
    """
    Learn mapping from a single 2d pose to 3d pose using a residual MLP model
    described in the paper:
        - A simple yet effective baseline for 3d human pose estimation, ICCV 2017

    Parameters
    ----------
    num_features : int
        Width of intermediate layer.
    dropout : float
        Dropout factor.
    num_layers : int
        Number of layers per residual block.
    num_blocks : int
        Number of residual blocks.
    num_joints : int
        Number of joints in the pose model(Human 3.6M has 17 joints). The input
        of the model is 2 * num_joints 2D coordinates and output is 3 * num_joints
        3D coordinates.
    """
    def __init__(self, num_features=1024, dropout=0.5, num_layers=2, num_blocks=2, num_joints=17):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_joints * 2, num_features)
        )
        self.res_blocks = nn.ModuleList([
            ResMLPBlock(num_features, dropout, num_layers) for _ in range(num_blocks)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(num_features, num_joints * 3),
            nn.Unflatten(1, (num_joints, 3))
        )

    def forward(self, x):
        x = self.encoder(x)
        for block in self.res_blocks:
            x = block(x)
        return self.decoder(x)


class PoseLiftingTemporalConv(nn.Module):
    """
    Learn mapping from stream of continuous 2d poses to 3d pose using a dialated convolution model
    described in the paper:
        - 3D human pose estimation in video with temporal convolutions and semi-supervised training

    Parameters
    ----------
    num_features : int
        Width of intermediate layer.
    dropout : float
        Dropout factor.
    num_layers : int
        Number of layers per residual block.
    num_blocks : int
        Number of residual blocks.
    num_joints : int
        Number of joints in the pose model(Human 3.6M has 17 joints). The input
        of the model is 2 * num_joints 2D coordinates and output is 3 * num_joints
        3D coordinates.
    """
    def __init__(
        self,
        receptive_field=243,
        dialations=[3, 3, 3, 3],
        num_features=1024,
        dropout=0.25,
        num_layers=2,
        num_blocks=4,
        num_joints=17
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_joints * 2, num_features)
        )
        self.res_blocks = nn.ModuleList([
            ResMLPBlock(num_features, dropout, num_layers) for _ in range(num_blocks)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(num_features, num_joints * 3),
            nn.Unflatten(1, (num_joints, 3))
        )

    def forward(self, x):
        x = self.encoder(x)
        for block in self.res_blocks:
            x = block(x)
        return self.decoder(x)
