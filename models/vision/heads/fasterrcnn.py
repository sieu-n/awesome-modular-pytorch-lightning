from torch import nn
from torchvision.ops import RoIPool


class FasterRCNNBaserpn(nn.Module):
    def __init__(self, n=3, d=256, in_channels=2048, k=9):
        self.sliding_window = nn.Conv2d(in_channels, d, kernel_size=n, padding=1)
        self.objectness = nn.Conv2d(d, 2 * k, kernel_size=1)
        self.bboxreg = nn.Conv2d(d, 4 * k, kernel_size=1)

        self.activation = nn.ReLU()

        for layer in [self.sliding_window, self.objectness, self.bboxreg]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature):
        """
        predict the roi from feature in the form of objectness score and bbox refinement.
        Objectness score describes the probability of object being at k anchor boxes centered at each feature. This
        bbox can be refined with the bbox refinement which is also predicted by the model.

        Parameters
        ----------
        feature : torch.Tensor(bs, C, W, H)
            Intermediate representations of neural network.
        Returns
        -------
        torch.Tensor(bs, 2 * k, W, H)
            objectness score.
        torch.Tensor(bs, 4 * k, W, H)
            predictions for bbox refinement.
        """
        d = self.activation(self.sliding_window(feature))

        objectness = self.objectness(d)
        bbox_pred = self.bboxreg(d)

        rois = None
        return {"roi": rois, "objectness": objectness, "bbox_refinement": bbox_pred}


class ROIPooler(nn.Module):
    def __init__(self, output_size=7, spatial_scale=1.0):
        """ Construct a ROIPooling module.
        reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/poolers.py
        """
        self.pool = RoIPool(output_size, spatial_scale=spatial_scale)

    def forward(self, feature, bbox):
        return self.pool(feature, bbox)