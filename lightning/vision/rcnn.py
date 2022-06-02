from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from lightning.common import _BaseLightningTrainer
from models.heads import FastRCNNPredictor, MLPHead, ROIPooler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from utils.bbox import check_isvalid_boxes, get_anchor_shapes
from utils.image.batcher import batch_images


class FasterRCNNBaseTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        # build models and heads defined in `model_cfg`.
        raise NotImplementedError()
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # training mode and hyperparameters.
        self.lambda_reg = training_cfg["lambda_reg"]
        self.bbox_dims = get_anchor_shapes(
            training_cfg["roi"]["anchor_size"], training_cfg["roi"]["aspect_ratio"]
        )

        self.computeProposalLoss = True
        self.computeFineTuningLoss = True

    def get_anchor_list(self, feature_shape, image_shape, bbox_shapes):
        """
        Returns the coordinates of anchors, in relative coordinates to the image.
        Parameters
        ----------
        feature_shape : tuple, list
            dimenstions of feature in (w, h, ...)
        image_shape : tuple, list
            dimenstions of image in (w, h, ...)
        Returns
        -------
        list[list[x, y, w, h]], len = w * h
            objectness score.
        """
        feature_w, feature_h = feature_shape[0], feature_shape[1]
        image_w, image_h = image_shape[0], image_shape[1]
        anchors = []
        for x in range(feature_w):
            for y in range(feature_h):
                xPos, yPos = (x + 0.5) / feature_w, (y + 0.5) / feature_h
                for bbox_shape in bbox_shapes:
                    anchors.append(
                        [xPos, yPos, bbox_shape[0] / image_w, bbox_shape[1] / image_h]
                    )
        return anchors

    def _get_roibatch(
        gt_bbox,
        anchors,
        roi_threshold=(0.7, 0.3),
        roi_num=(128, 128),
        ensure_one_positive=True,
    ):
        """
        Given a single image, Faster R-CNN creates a batch of ROIs that is used to compute the RPN objectness
        classification, bounding box regression, and classification loss for training. `_get_roibatch` returns a batch
        of labels for each task. Specifically, the function recieves the possible anchors location and g.t. bbox labels
        and creates a balanced batch of ROI where the number of positive and negative ROIs can be specified, whereas
        randomly selecting ROI will return much more negative ROI. The values returned can be directly compared with
        the model predictions to compute loss.
        Parameters
        ----------
        anchors: list
        roi_threshold: int, int, (positive_roi_threshold, negative_roi_threshold)
            roi threshold for defining positive and negative roi. ROI is classified as positive when IoU overlap with
            closest G.T bbox is larger than `positive_roi_threshold` and negative when overlap is smaller than
            `negative_roi_threshold`.
        roi_num: int, int, (positive_roi_count, negative_roi_threshold)
            number of positive and negative rois to sample.
        ensure_one_positive: bool
            If true, ensure that each object has at least one positive anchor assigned.
        Returns
        -------
        list[(k, w, h), ...], anchors_selected
        torch.Tensor(bs, k, 256), is_selected
        cls_label
        gt_bbox
        """

        # for x
        return None, None, None

    def _objectness_classification_loss(objectness_pred, is_object):
        return 0

    def _bbox_regression_loss(anchors, bbox_pred, gt_bbox):
        return 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        image_w, image_h = x.size(3), x.size(2)
        assert x.size(0) == 1, "batch size must be 1 for fasterrcnn training."

        loss = 0

        feature = self.backbone(x)

        rpn_feature = self.get_hook("rpn", device=device)
        # proposals: torch.Tensor()
        rpn_pred = self.rpn(rpn_feature)
        rois = rpn_pred["roi"]

        # In the first step, we train the RPN as described in Section 3.1.3 of the paper.
        # train using objectness classification and bounding box regression loss.
        # anchors: list[ list[x, y, w, h] ], len: w * h * len(self.bbox_dims)
        anchors = self.get_anchor_list(
            feature_shape=(rois.size(3), rois.size(2)),
            image_shape=(image_w, image_h),
            bbox_shapes=self.bbox_dims,
        )
        is_selected, cls_label, gt_bbox = self._get_roibatch(y, anchors)

        if self.computeProposalLoss:
            objectness, bbox_pred = rpn_pred["objectness"], rpn_pred["bbox_refinement"]
            # compute rpn classification loss
            rpn_loss_cls = self._objectness_classification_loss(objectness, cls_label)

            # compute bounding box regression loss
            rpn_loss_reg = self._bbox_regression_loss(anchors, bbox_pred, gt_bbox)

            self.log("step/rpn_loss_cls", rpn_loss_cls)
            self.log("step/rpn_loss_reg", rpn_loss_reg)
            proposal_loss = rpn_loss_cls + rpn_loss_reg * self.lambda_reg
            loss += proposal_loss

        # todo finetune.
        # x, f = self.pooler(feature, rois)
        # loss = self.loss_fn(pred, y)

        self.log("step/train_loss", loss)
        return loss

    def get_rpn_loss():
        return 0

    def get_finetune_loss():
        return 0

    def evaluate(self, batch, stage=None):
        x, y = batch
        # todo: mAP
        return 0, 0


################################################################
# lightning wrappers for torchvision rcnn models.
################################################################
class BaseFasterRCNN(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        # build models and heads defined in `model_cfg`.
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # rpn
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        rpn_head = RPNHead(  # TODO
            in_channels=1024,
            num_anchors=rpn_anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(  # TODO
            rpn_anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
            score_thresh=0.0,
        )
        # roi head
        roi_pooler = ROIPooler(output_size=7, spatial_scale=1.0)  # TODO
        box_head = MLPHead(in_features=1024 * 49, num_channels=[1024, 1024])
        box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=21)
        self.roi_heads = RoIHeads(  # TODO
            # Box
            roi_pooler,
            box_head,
            box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def training_step(self, batch, batch_idx):
        assert "images" in batch
        assert "boxes" in batch
        assert "labels" in batch
        batch_size = len(batch["images"])
        img_w = [batch["images"][idx].size(2) for idx in range(batch_size)]
        img_h = [batch["images"][idx].size(1) for idx in range(batch_size)]
        check_isvalid_boxes(batch["boxes"], xywh=False, img_w=img_w, img_h=img_h)

        images = batch_images(batch["images"])
        targets = [
            {"boxes": batch["boxes"][idx], "labels": batch["labels"][idx]}
            for idx in range(batch_size)
        ]

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, None, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes) # TODO

        # compute loss
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        losses = sum(loss for loss in losses.values())
        return losses

    def evaluate(self, batch, stage=None):
        # x, y = batch
        # todo: mAP
        return 0, 0


class TorchVisionFasterRCNN(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        # build models and heads defined in `model_cfg`.
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        # roi_pooler = torchvision.ops.RoIPool(output_size=7, spatial_scale=1.0)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # training mode and hyperparameters.
        self.backbone.out_channels = 2048
        self.model = FasterRCNN(
            self.backbone,
            num_classes=21,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def training_step(self, batch, batch_idx):
        assert "images" in batch
        assert "boxes" in batch
        assert "labels" in batch
        batch_size = len(batch["images"])
        img_w = [batch["images"][idx].size(2) for idx in range(batch_size)]
        img_h = [batch["images"][idx].size(1) for idx in range(batch_size)]
        check_isvalid_boxes(batch["boxes"], xywh=False, img_w=img_w, img_h=img_h)

        loss_dict = self.model(
            batch["images"],
            [
                {"boxes": batch["boxes"][idx], "labels": batch["labels"][idx]}
                for idx in range(batch_size)
            ],
        )
        for k in loss_dict:
            self.log(k, loss_dict[k])

        losses = sum(loss for loss in loss_dict.values())
        return losses

    def evaluate(self, batch, stage=None):
        # x, y = batch
        # todo: mAP
        return 0, 0
