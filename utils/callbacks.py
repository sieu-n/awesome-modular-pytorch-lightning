from pytorch_lightning import callbacks as PytorchLightningCallbacks
from collections import OrderedDict


def build_callback(callback_cfg):
    if "name" in callback_cfg:
        return _build_callback(
            callback_class=callback_cfg["name"],
            file=callback_cfg.get("file", None),
            **callback_cfg.get("args", {}),
        )
    else:
        # NOTE assumes that callback is already baked.
        return callback_cfg


def _build_callback(callback_class=None, file=None, **kwargs):
    known_callbacks = {}
    callback_pool = OrderedDict({
        "lightning": PytorchLightningCallbacks,
    })
    # if library is specified
    if file:
        if type(callback_class) == str:
            callback_class = getattr(callback_pool[file], callback_class)
        else:
            raise ValueError("If `file` is specified, provide the name of the module as a string.")
    elif type(callback_class) == str:
        if callback_class in known_callbacks:
            callback_class = known_callbacks[callback_class]
        else:
            is_found = False
            for location in callback_pool.values():
                if hasattr(location, callback_class):
                    print(f"'{callback_class}' was found in `{location}.")
                    callback_class = getattr(location, callback_class)
                    is_found = True
                    break
            if not is_found:
                raise ValueError(f"{callback_class} was not found in the pool of modules: {list(callback_pool.values())}")
    callback = callback_class(**kwargs)
    return callback
