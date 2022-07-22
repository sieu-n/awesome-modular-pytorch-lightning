from copy import deepcopy
try:
    import mmdet
    from mmcv import ConfigDict
except ImportError:
    pass

from lightning.base import _BaseLightningTrainer


class MMDetectionTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        """
        In essence, `MMDetectionTrainer` is a pytorch-lightning version of the runners in `mmcv`. The implementation
        was especially influenced by the implementation of `EpochBasedRunner`.
        https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/epoch_based_runner.html
        """
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        assert "mm_model" in model_cfg
        self.MMDET_model = mmdet.models.builder.build_detector(ConfigDict(model_cfg["mm_model"]))
        # TODO backbone wrappers so we can use more generic feature extractors.

    def forward(self, x):
        """
        x: batch of images
        """
        raise NotImplementedError()

    def _training_step(self, batch, batch_idx=0):
        return self.MMDET_model.train_step(batch, optimizer=None)

    def evaluate(self, batch, stage=None):
        return self.MMDET_model.val_step(batch)
