"""ResNet in PyTorch. Support `low_res` argument for CIFAR10.

Reference: https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as _ResNet


def ResNet18(**kwargs):
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)


class ResNet(_ResNet):
    def __init__(self, low_res=False, **kwargs):
        """
        Adapt torchvision.models.resnet.ResNet to extract features only.

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
