from torch.utils.data._utils.collate import default_collate


def build_collate_fn(collate_override={}):
    """
    Return a function that applies different collation functions to each key from the dataset.
    """
    to_override = list(collate_override.keys())
    collate_fns = {}
    for key in collate_override.keys():
        collate_fn_builder = find_collate_fn_from_name(collate_override[key]["name"])
        collate_fns[key] = collate_fn_builder(**collate_override[key].get("args", {}))

    def _collate_fn(batch):
        collated = {}
        for k in batch[0].keys():
            content = [sample[k] for sample in batch]
            if k in to_override:
                collated[k] = collate_fns[k](content)
            else:
                collated[k] = default_collate(content)
        return collated

    return _collate_fn


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
def KeepSequence():
    def _KeepSequence(batch):
        return batch

    return _KeepSequence


collate_fn_list = {
    "KeepSequence": KeepSequence,
}
