"""ResNet in PyTorch. Support `low_res` argument for CIFAR10.

Reference: 
    - https://github.com/kuangliu/pytorch-cifar
    - https://pytorch.org/vision/0.12/_modules/torchvision/models/resnet.html
"""
import torch.nn as nn
from torchvision.models.resnet import ResNet as _ResNet


def ResNet18(preact=False, rn_d=False, **kwargs):
    block = PreActBasicBlock if preact else BasicBlock
    rn_model = ResNetD if rn_d else ResNet
    return rn_model(block=block, layers=[2, 2, 2, 2], **kwargs)


def ResNet34(preact=False, rn_d=False, **kwargs):
    block = PreActBasicBlock if preact else BasicBlock
    rn_model = ResNetD if rn_d else ResNet
    return rn_model(block=block, layers=[3, 4, 6, 3], **kwargs)


def ResNet50(preact=False, rn_d=False, **kwargs):
    block = PreActBottleneck if preact else Bottleneck
    rn_model = ResNetD if rn_d else ResNet
    return rn_model(block=block, layers=[3, 4, 6, 3], **kwargs)


def ResNet101(preact=False, rn_d=False, **kwargs):
    block = PreActBottleneck if preact else Bottleneck
    rn_model = ResNetD if rn_d else ResNet
    return rn_model(block=block, layers=[3, 4, 23, 3], **kwargs)


"""
Includes building block variants from the following papers
Classic ResNet:
    [1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE
        conference on computer vision and pattern recognition. 2016.
PreActResNet:
    [2] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on
        computer vision. Springer, Cham, 2016.
ResNetD:
    [3] He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
"""


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)    # ResNet-B is already implemented.
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class PreActBasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class PreActBottleneck(Bottleneck):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)    # ResNet-B is already implemented.
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        return out


class ResNet(_ResNet):
    def __init__(self, low_res=False, **kwargs):
        """
        Adapt torchvision.models.resnet.ResNet to support small dataset(CIFAR10).

        Additional / Modified Parameters
        --------------------------------
        low_res: bool
            Adapting to small images such as CIFAR10

        Returns
        -------
        (a) torch.tensor, list[torch.tensor] of len==4
            When return_middle_features==True, we return some
            intermediate features that are used for Loss Prediction.
        (b) torch.tensor
            Typical output of Resnet when return_middle_features==False
        """
        super(ResNet, self).__init__(**kwargs)
        if low_res:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()

    def _forward_impl(self, x):
        # Override: https://pytorch.org/vision/0.12/_modules/torchvision/models/resnet.html
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

'''
TODO: fix bug in resnet-d
class ResNetD(_ResNet):
    def __init__(self, low_res=False, **kwargs):
        """
        Adapt torchvision.models.resnet.ResNet for ResNetD.

        Additional / Modified Parameters
        --------------------------------
        low_res: bool
            Adapting to small images such as CIFAR10

        Returns
        -------
        (a) torch.tensor, list[torch.tensor] of len==4
            When return_middle_features==True, we return some
            intermediate features that are used for Loss Prediction.
        (b) torch.tensor
            Typical output of Resnet when return_middle_features==False
        """
        super(ResNetD, self).__init__(**kwargs)
        # ResNet-C
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            self._norm_layer(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self._norm_layer(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if low_res:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride), -> replace to AvgPool2d.
                nn.AvgPool2d(2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
'''

