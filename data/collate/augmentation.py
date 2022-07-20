from timm.data import FastCollateMixup as _FastCollateMixup


class timm_collate_mixup(_FastCollateMixup):
    """
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
    Mixup/Cutmix that applies different params to each element or whole batch
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

    def __call__(self, batch, _=None):
        # {"image": }
        batch = [(x["images"], x["labels"]) for x in batch]
        images, labels = super().__call__(batch, _)
        return {"images": images, "labels": labels}
