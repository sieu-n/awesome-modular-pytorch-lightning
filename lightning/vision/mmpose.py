try:
    import mmpose
    from mmcv import ConfigDict
except ImportError:
    pass

from lightning.base import _BaseLightningTrainer
from utils.mmcv import send_datacontainers_to_device, unpack_datacontainers


class MMPoseTrainer(_BaseLightningTrainer):
    def init(self, model_cfg, training_cfg):
        """
        In essence, `MMDetectionTrainer` is a pytorch-lightning version of the runners in `mmcv`. The implementation
        was especially influenced by the implementation of `EpochBasedRunner`.
        https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/epoch_based_runner.html
        """
        assert "mm_model" in model_cfg
        self.MMPose_model = mmpose.models.build_posenet(
            ConfigDict(model_cfg["mm_model"])
        )
        # TODO backbone wrappers so we can use more generic feature extractors.

    def forward(self, x):
        """
        x: dict
            batch of data. For example,
            {
                "img":
                "img_metas":
                "gt_bboxes":
                "gt_labels":
            }
        """
        # out = self.MMDET_model.forward(x)
        pass

    def _predict_step(self, batch):
        # TODO: implement based on `simple_test`
        pass

    def _training_step(self, batch, batch_idx=0):
        # train dataloader.
        send_datacontainers_to_device(data=batch, device=self.device)
        batch = unpack_datacontainers(batch)

        total_loss, losses = self.compute_loss(**batch)

        # log step losses
        self.log_step_results(losses)
        res = total_loss

        return res, losses

    def evaluate(self, sample, stage=None):
        """
        sample: dict
            Single data sample.
        """
        # validation dataloader creates slightly different format from train dataloader.
        send_datacontainers_to_device(data=sample, device=self.device)
        sample = unpack_datacontainers(sample)
        # assert a bunch of stuff
        assert len(sample["img"]) == 1 and sample["img"][0].size(1) == 3
        pred = self.MMPose_model.forward_test(
            imgs=sample["img"],
            img_metas=sample["img_metas"],
        )
        return pred

    def compute_loss(self, *args, **kwargs):
        """
        Equivalent to `val_step` and `train_step` of `self.MMDET_model`.
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L221
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L256
        x: dict
            batch of data. For example,
            ```
            {
                "img": torch.Tensor, Size: [batch_size, C, W, H],
                "img_metas": [
                    {
                        'filename': 'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
                        'ori_filename': 'JPEGImages/000001.jpg',
                        'ori_shape': (500, 353, 3),
                        ...
                    },
                    ...(batch size times)
                ],
                "gt_bboxes": [
                    torch.Tensor, Size: [# objects in image, 4],
                    ...(batch size times)
                ],
                "gt_labels": [
                    torch.Tensor, Size: [# objects in image],
                    ...(batch size times)
                ],
            }
            ```
        """
        losses = self.MMPose_model.forward_train(
            *args,
            **kwargs,
        )
        return self.parse_losses(losses)

    def parse_losses(self, losses):
        # `_parse_losses`: https://github.com/open-mmlab/mmdetection/blob/
        # 56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L176
        loss, log_vars = self.MMPose_model._parse_losses(losses)
        return loss, log_vars

    def log_step_results(self, losses):
        map_loss_names = {# TODO
            "loss": "total_loss",
            "acc": "classification-accuracy",
            "loss_rpn_cls": "loss/rpn_cls",
            "loss_rpn_bbox": "loss/rpn_bbox",
            "loss_bbox": "loss/bbox_reg",
            "loss_cls": "loss/classification",
        }
        for key in losses:
            # map metric names same name with epoch-wise metrics defined in:
            # `configs/vision/object-detection/mmdet/mmdet-base.yaml`
            loss_name = map_loss_names.get(key, key)
            self.log(f"step/{loss_name}", losses[key])
