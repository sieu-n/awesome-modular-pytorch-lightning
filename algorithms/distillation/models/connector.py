import math

from torch import nn


def build_feature_connector(
    s_channel, t_channel, depth=1, bn=True, bias=False, kernel_size=1
):
    """
    Build feature connector that matches the dimenstion of the smaller student model and
    larger teacher model.
    """
    C = [
        nn.Conv2d(
            s_channel,
            t_channel,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=bias,
        )
    ]
    if bn:
        C += [nn.BatchNorm2d(t_channel)]

    for idx in range(depth - 1):
        C += [
            nn.ReLU(t_channel),
            nn.Conv2d(
                t_channel,
                t_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
        ]
        if bn:
            C += [nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)
