from timm.data import Mixup
from timm.data.mixup import mixup_target


class MixupCutmix(Mixup):
    def __init__(self, multilabel=True, allow_odd=True, **kwargs):
        """Mixup/Cutmix that applies different params to each element or whole batch
        https://github.com/rwightman/pytorch-image-models/blob/e4360e6125bb0bb4279785810c8eb33b40af3ebd/timm/data/mixup.py#L90
        Args:
            mixup_alpha (float): mixup alpha value, mixup is active if > 0.
            cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
            cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
            prob (float): probability of applying mixup or cutmix per batch or element
            switch_prob (float): probability of switching to cutmix instead of mixup when both are active
            mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
            correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
            label_smoothing (float): apply label smoothing to the mixed target tensor
            num_classes (int): number of classes for target
        """
        super().__init__(**kwargs)
        self.allow_odd = allow_odd
        self.multilabel = multilabel

    def __call__(self, x, target):
        if len(x) % 2 == 1:
            if self.allow_odd:
                print(f"Batch size should be even when applying mixup. Got batch of size: {x.shape}, trimming last image.")
                x = x[:-1]
                target = target[:-1]
            else:
                raise ValueError("Batch size should be even when using this")

        if self.mode == "elem":
            lam = self._mix_elem(x)
        elif self.mode == "pair":
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(
            target, self.num_classes, lam, self.label_smoothing, x.device
        )
        if self.multilabel is False:
            # returns mixup that sum to 1 instead of multi-label as proposed in timm.
            target = target / target.sum(dim=0)
        return x, target
