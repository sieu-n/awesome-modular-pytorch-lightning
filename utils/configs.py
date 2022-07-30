from copy import deepcopy

import yaml


def read_yaml(yaml_path):
    """Load configs from yaml file and return dictionary."""
    with open(yaml_path, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def merge_config(cfg_base, cfg_from):
    """
    Overwrite `cfg_base` with values of `cfg_from`. For example:
    cfg_base = {
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
        },
        "d": 3,
        "e": 1
    }
    cfg_from = {
        "optimizer": {
            "lr": 0.1,
            "momentum": 0.99,
        },
        "d": 1
    }
    --> cfg_from = {
        "optimizer": {
            "name": "adam",
            "lr": 0.1,
            "momentum": 0.99,
        },
        "d": 1,
        "e": 1
    }

    Parameters
    ----------
    cfg_base: dict
        Dictionary of config to merge into.
    cfg_from: dict
        Dictionary of config to merge keys from.
    Returns
    -------
    dict
        Dictionary combining the 2 configs.
    """

    def recursively_write(base, f):
        for k in f.keys():
            if isinstance(f[k], dict):
                if k in base:
                    base[k] = recursively_write(deepcopy(base[k]), f[k])
                else:
                    base[k] = f[k]
            # overwrite, like the case of "optimizer/lr" and "d" in the example.
            else:
                base[k] = f[k]
        return base

    return recursively_write(deepcopy(cfg_base), cfg_from)


def read_configs(yaml_paths, compile=True):
    """
    Load and combine multiple yaml files and return final config.

    Parameters
    ----------
    yaml_path: list[str]
        paths to yaml file in order of importance in merging.
    Returns
    -------
    dict
        Dictionary contining final configs from multiple `yaml` files.
    """
    cfg = {}
    assert len(yaml_paths) > 0
    # override last config.
    for yaml_path in yaml_paths[::-1]:
        new_cfg = read_yaml(yaml_path)
        cfg = merge_config(cfg, new_cfg)
    # complie links. This behaviour might be disabled when we would like to edit the config file later such as during
    # hyperparameter sweeping.
    if compile:
        cfg = compile_links(cfg)
    return cfg


def compile_links(cfg):
    """
    Computes `links` in config file that can be defined using curly brackets. For example,

    ```
    metrics:
        ConfusionMatrix:
            args:
                num_classes: "{config.const.num_classes}"   <-- This will reference cfg["const"]["num_classes"]
                ...
    const:
        num_classes: 10
    ```

    Since the python `eval` function computes the results, we can also apply simple arithmetics. For example, the
    weight decay value is often set proportional to the learning rate. The recursive search works when each config is a
    dictionary, list, tuple with index, or an object with the property.

    ```
    training:
        learning_rate: 0.01
        optimizer: "adamw"
        optimizer_cfg:
            weight_decay: "{training.learning_rate}*0.001"  <-- learning_rate * 0.001
    ```

    However as more complex objects might also be inside config and the logic of 1) converting them into string,
    2) using `eval` to original form, could destruct the data. Therefore as an exception if the entire string is
    defined as a link, that object is returned directly.
    """

    def recursive_search(s):
        _cfg = cfg
        for k in s.split("."):
            if type(_cfg) == dict:
                _cfg = _cfg[k]
            elif type(_cfg) == list or type(_cfg) == tuple:
                _cfg = _cfg[int(k)]
            else:
                _cfg = getattr(_cfg, k)
        return _cfg

    def compile_str(query):
        left, cursor = -1, 0
        is_compiled = False
        _query = deepcopy(query)
        while cursor < len(_query):
            if _query[cursor] == "{":
                assert (
                    left == -1
                ), f"Two consecutive opening brackets found. query: {query}"
                is_compiled = True
                left = cursor
            if _query[cursor] == "}":
                assert (
                    left != -1
                ), f"Closing bracket found but was never opened. query: {query}"
                obj = recursive_search(_query[left + 1 : cursor])
                if cursor == len(query) - 1 and left == 0:
                    print(f"Compiled {query} := {obj}")
                    return obj, True

                _query = _query.replace(_query[left : cursor + 1], str(obj))
                cursor = left
                left = -1
            cursor += 1
        if is_compiled:
            assert left == -1, f"Brackets was opened but not close. query: {query}"
            print(f"Compiled {query} := {_query}")
            return eval(_query), True
        else:
            return query, False

    def recurse_iter(parsed_cfg):
        if type(parsed_cfg) == dict:
            it = parsed_cfg.keys()
        elif type(parsed_cfg) == list:
            it = range(len(parsed_cfg))
        else:
            is_compiled = False
            if type(parsed_cfg) == str:
                parsed_cfg, is_compiled = compile_str(parsed_cfg)
            return parsed_cfg, is_compiled

        is_something_compiled = False
        for k in it:
            parsed_cfg[k], is_compiled = recurse_iter(parsed_cfg[k])
            if is_compiled:
                is_something_compiled = True
        return parsed_cfg, is_something_compiled

    is_compiled = True
    compiled_cfg = deepcopy(cfg)
    while is_compiled:
        compiled_cfg, is_compiled = recurse_iter(compiled_cfg)
    return compiled_cfg


class dict2obj(object):
    """
    Simple conversion of a recursive dictionary into an object. Note that lists
    are preserved.
    https://gist.github.com/byron2r/2657437
    """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)
