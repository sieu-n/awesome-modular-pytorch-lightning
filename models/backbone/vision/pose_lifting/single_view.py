import numpy as np
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
        self.layers = nn.ModuleList(
            [LBRD(features, features, dropout) for _ in range(num_layers)]
        )

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

    def __init__(
        self, num_features=1024, dropout=0.5, num_layers=2, num_blocks=2, num_joints=17
    ):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Flatten(), nn.Linear(num_joints * 2, num_features)
        )
        self.res_blocks = nn.ModuleList(
            [ResMLPBlock(num_features, dropout, num_layers) for _ in range(num_blocks)]
        )
        self.shrink = nn.Sequential(
            nn.Linear(num_features, num_joints * 3), nn.Unflatten(1, (num_joints, 3))
        )

    def forward(self, x):
        x = self.expand(x)
        for block in self.res_blocks:
            x = block(x)
        return self.shrink(x)


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        features=1024,
        kernel_size=3,
        dilation=1,
        dropout=0.25,
        num_layers=2,
        residual=True,
        slice_pad=2,
    ):
        super().__init__()
        self.residual = residual
        self.slice_pad = slice_pad

        self.init_layer = nn.Sequential(
            nn.Conv1d(features, features, kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(features, features, kernel_size, bias=False),
                    nn.BatchNorm1d(features),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
                for _ in range(num_layers - 1)
            ]
        )

    def forward(self, x):
        if self.residual:
            skip_con = x

        x = self.init_layer(x)
        for layer in self.layers:
            x = layer(x)

        if self.residual:
            # Due to valid convolutions, we slice the residuals (left and right, symmetrically) to
            # match the shape of subsequent tensors.
            skip_con = skip_con[self.slice_pad : -self.slice_pad]
            x = skip_con + x
        return x


class PoseLiftingTemporalConv(nn.Module):
    """
    Learn mapping from stream of continuous 2d poses to 3d pose using a dialated convolution model
    described in the paper:
        - 3D human pose estimation in video with temporal convolutions and semi-supervised training

    TODO
    - write docstring
    - bn momentum
    - dense model conversion
    - metric
    - memory leaking
    - tta
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
        kernel_sizes=[3, 3, 3, 3],
        num_features=1024,
        dropout=0.25,
        num_layers=2,
        num_joints=17,
    ):
        super().__init__()
        self.receptive_field = np.prod(kernel_sizes)
        self.num_blocks = len(kernel_sizes)
        # e.g. if kernel_sizes=[3, 3, 3, 3] then sparse_dilations=[3, 9, 27, 81]
        self.sparse_dilations = []
        temp = 1
        for k in kernel_sizes:
            temp *= k
            self.sparse_dilations.append(temp)
        self.is_dense = False

        self.expand = nn.Sequential(
            nn.Conv1d(num_joints * 2, num_features, kernel_sizes[0], bias=False),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        res_blocks = []
        for i in range(self.num_blocks):
            res_blocks.append(
                TemporalConvBlock(
                    features=num_features,
                    kernel_size=kernel_sizes[i],
                    dilation=self.sparse_dilations[i],
                    dropout=dropout,
                    num_layers=num_layers,
                    residual=True,
                    slice_pad=(kernel_sizes[i] - 1) * self.sparse_dilations[i] // 2,
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)

        self.shrink = nn.Conv1d(num_features, num_joints * 3, 1)

    def forward(self, x):
        if self.is_dense:
            assert x.ndim == 4
        else:
            assert x.ndim == 4
            assert (
                x.size(1) == self.receptive_field
            ), f"Sparse mode expected sequence of len \
                {self.receptive_field}, got shape {x.shape}."
        x = self.expand(x)
        for block in self.res_blocks:
            x = block(x)
        return self.shrink(x).reshape(x.size(0), 17, 3)
