from timm.data import FastCollateMixup as _FastCollateMixup
from torch.utils.data._utils.collate import default_collate


def build_collate_fn(name, kwargs):
    """
    Return a function that applies different collation functions to each key from the dataset.
    """
    return collate_fn_list[name](**kwargs)


def find_collate_fn_from_name(f_name):
    if type(f_name) == str:
        # find transform name that matches `name` from TRANSFORM_DECLARATIONS
        assert f_name in collate_fn_list
        return collate_fn_list[f_name]
    else:
        print(f"{f_name} might already be a function.")
        return f_name


################################################################
# Implement useful collate functions
################################################################
def KeepSequence(keys_to_apply=[]):
    """
    Keep sequences of different length
    """
    collate_fns = {}
    for key in keys_to_apply:
        collate_fn_builder = find_collate_fn_from_name(keys_to_apply[key]["name"])
        collate_fns[key] = collate_fn_builder(**keys_to_apply[key].get("args", {}))

    def _KeepSequence(batch):
        collated = {}
        for k in batch[0].keys():
            content = [sample[k] for sample in batch]
            if k in keys_to_apply:
                collated[k] = content
            else:
                collated[k] = default_collate(content)
        return collated

    return _KeepSequence

    return _KeepSequence


class FastCollateMixup(_FastCollateMixup):
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


collate_fn_list = {
    "KeepSequence": KeepSequence,
    "FastCollateMixup": FastCollateMixup,
}
