"""
Useful documentation for mmdetection:
    https://mmdetection.readthedocs.io/en/latest/api.html?

Originally, the `DataLoaders` in mmdetection handle multi-gpu by custom collate functions and importantly the
mmcv.parallel.MMDataParallel class which wrapps the dataloader which interpret the specifical format of input. Since
pytorch-lightning handles such multi-gpu training, we use a classic `torch.utils.data.DataLoader` object to batch data
but parse the output of collate function during `training_step` and `validation_step`.
Specifically, these include:
    - the logic that recieves `_sample` and converts it into `sample` in `MMDetectionTrainer::evaluate`
    - the logic that recieves `_batch` and converts it into `batch` in `MMDetectionTrainer::_training_step`
"""

try:
    import mmdet
    from mmcv import ConfigDict
except ImportError:
    pass

from lightning.base import _BaseLightningTrainer
from utils.mmcv import send_datacontainers_to_device, unpack_datacontainers


class MMDetectionTrainer(_BaseLightningTrainer):
    def init(self, model_cfg, training_cfg):
        """
        In essence, `MMDetectionTrainer` is a pytorch-lightning version of the runners in `mmcv`. The implementation
        was especially influenced by the implementation of `EpochBasedRunner`.
        https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/epoch_based_runner.html
        """
        assert "mm_model" in model_cfg
        self.MMDET_model = mmdet.models.builder.build_detector(
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
        out = self.MMDET_model.forward(x)

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
        return total_loss, losses

    def evaluate(self, sample, stage=None):
        """
        sample: dict
            Single data sample.
        """
        # validation dataloader creates slightly different format from train dataloader.
        send_datacontainers_to_device(data=sample, device=self.device)
        sample = unpack_datacontainers(sample)
        # assert a bunch of stuff
        assert sample["img"].size(0) == 1 and sample["img"].size(1) == 3

        _, losses = self.compute_loss(**sample)
        return losses

    def compute_loss(self, img, img_metas, gt_bboxes, gt_labels, *args, **kwargs):
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
        losses = self.MMDET_model.forward_train(
            img=img,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            *args,
            **kwargs,
        )
        return self.parse_losses(losses)

    def parse_losses(self, losses):
        # `_parse_losses`: https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L176
        loss, log_vars = self.MMDET_model._parse_losses(losses)
        return loss, log_vars

    def log_step_results(self, losses):
        map_loss_names = {
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
