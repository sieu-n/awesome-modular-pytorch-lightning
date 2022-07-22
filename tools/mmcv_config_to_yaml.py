from argparse import ArgumentParser

import yaml
from mmcv import Config


def copy_to_dict(original_d):
    d = {}
    if not hasattr(original_d, "keys"):
        if type(original_d) == tuple:
            original_d = list(original_d)
        if type(original_d) == list:
            for k in range(len(original_d)):
                original_d[k] = copy_to_dict(original_d[k])

        return original_d
    for k in original_d.keys():
        d[k] = copy_to_dict(original_d[k])
    return d


if __name__ == "__main__":
    """
    Read MMdetection configuration files and convert them into .yaml files for use in MPL.
    """
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--to", type=str, default=None)

    args = parser.parse_args()

    mm_config = Config.fromfile(args.config)
    # recursively convert to python-dict
    dict_config = copy_to_dict(mm_config)

    if args.to is None:
        to = args.config.replace(".py", ".yaml").replace("/", "-")
        print("Saving to:", to)
    else:
        to = args.to
    with open(to, "w") as outfile:
        yaml.dump(dict_config, outfile, default_flow_style=False)
