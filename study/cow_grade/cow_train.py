from argparse import ArgumentParser
import os
import shutil

from torch.utils.data import Dataset
from main import Experiment
from utils.configs import read_configs

class CowDataset(Dataset):
    def __init__(self, csv_path, ):
        pass
    def 
if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # move data
    base_path = "/content/drive/data/cow_classification/"
    target_path = "./cow/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for subset in ("train", "test"):
        shutil.unpack_archive(f"{base_path}{subset}.zip", f"{target_path}{subset}", "zip")

    # train
    experiment = Experiment(cfg)
    experiment.setup_experiment_from_cfg(cfg, setup_dataset=False)
    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        epochs=cfg["training"]["epochs"],
    )
    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
