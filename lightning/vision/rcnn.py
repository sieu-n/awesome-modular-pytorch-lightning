import torch
import torch.nn as nn
from lightning.common import _BaseLightningTrainer
import models.heads as TorchHeads
from models.vision.backbone import build_backbone
from utils.bbox import get_bbox_shapes


class FasterRCNNBaseTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(training_cfg, *args, **kwargs)
        # define model function.
        self.backbone = build_backbone(model_cfg["backbone"])
        # setup hooks, TODO move this into `_BaseLightningTrainer`
        if "hooks" in model_cfg:
            for hook_name in model_cfg["hooks"]:
                self.setup_hooks(
                    self.backbone,
                    key=hook_name,
                    **model_cfg["hooks"][hook_name]
                )

        rpn_cfg = model_cfg["heads"]["rpn"]
        self.rpn = getattr(TorchHeads, rpn_cfg["ID"])(**rpn_cfg["cfg"])

        roipool_cfg = model_cfg["heads"]["roi_pooler"]
        self.pooler = getattr(TorchHeads, rpn_cfg[""])(
            **roipool_cfg,
        )

        classifier_cfg = model_cfg["heads"]["classifier"]
        self.classifier = getattr(TorchHeads, classifier_cfg["ID"])(
            in_features=model_cfg["backbone"]["out_features"],
            **classifier_cfg["cfg"],
        )
        # training mode and hyperparameters.
        self.mode = "rpn"
        self.lambda_reg = training_cfg["lambda_reg"]
        self.bbox_dims = get_bbox_shapes(training_cfg["roi"]["anchor_size"], training_cfg["roi"]["aspect_ratio"])

    def get_anchor_list(self, feature_shape, image_shape):
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
                for bbox_shape in self.bbox_dims:
                    anchors.append([xPos, yPos, bbox_shape[0] / image_w, bbox_shape[1] / image_h])
        return anchors

    def _get_roibatch(gt_bbox, anchors):
        # todo
        pass

    def _objectness_classification_loss(objectness_pred, is_object):
        return 0

    def _bbox_regression_loss(anchors, bbox_pred, gt_bbox):
        return 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device

        loss = 0

        feature = self.backbone(x)

        rpn_feature = self.get_hook("rpn", device=device)
        # proposals: torch.Tensor()
        rpn_pred = self.rpn(rpn_feature)
        rois = rpn_pred["roi"]

        if self.mode == "rpn":
            # train using objectness classification and bounding box regression loss.
            # anchors: list[ list[x, y, w, h] ], len: w * h * len(self.bbox_dims)
            anchors = self.propose_anchor(self.model)
            is_selected, cls_label, gt_bbox = self._get_roibatch(y, anchors)

            # compute rpn classification loss
            objectness, bbox_pred = rpn_pred["objectness"], rpn_pred["bbox_refinement"]
            rpn_loss_cls = self._objectness_classification_loss(objectness, cls_label) * is_selected 

            # compute bounding box regression loss
            rpn_loss_reg = self._bbox_regression_loss(anchors, bbox_pred, gt_bbox)

            self.log("step/rpn_loss_cls", rpn_loss_cls)
            self.log("step/rpn_loss_reg", rpn_loss_reg)
            loss = rpn_loss_cls + rpn_loss_reg * self.lambda_reg

        elif self.mode == "finetune":
            # todo
            x, f = self.pooler(feature, rois)
            loss = self.loss_fn(pred, y)

        self.log("step/train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        # todo: mAP
        return 0, 0
