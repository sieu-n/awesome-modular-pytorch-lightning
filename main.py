from argparse import ArgumentParser

# custom
from utils.configs import read_configs


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    config = read_configs(args.configs)
