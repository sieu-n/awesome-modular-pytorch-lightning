import os
import random
import shutil
import torch
from argparse import ArgumentParser
from PIL import Image

import pandas as pd
from main import Experiment
from torch.utils.data import Dataset
from utils.configs import read_configs


class CowDataset(Dataset):
    def __init__(
        self,
        base_dir,
        csv_path,
        keep_order=False,
        has_labels=True,
        idx_min=0,
        idx_max=-1,
        shuffle_seed=42
    ):
        self.base_dir = base_dir
        self.has_labels = has_labels

        df = pd.read_csv(csv_path)
        indicies = df.index
        if has_labels and not keep_order:  # in order to randomly select validation set
            random.Random(shuffle_seed).shuffle(indicies)
        indicies = indicies[idx_min:idx_max]

        self.image_names = list(df.iloc[indicies]["imname"])
        self.grade = list(df.iloc[indicies]["grade"])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.base_path + self.image_names[idx])
        if self.has_labels:
            grade = self.grade[idx]
            return {"images": image, "labels": grade}
        else:
            return {"images": image}


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

    # make dataset
    train_data_dir = "./cow/train/images/"
    train_csv_path = "./cow/train/grade_labels.csv"
    train_dataset = CowDataset(
        base_dir=train_data_dir,
        csv_path=train_csv_path,
        idx_max=9000
    )
    val_dataset = CowDataset(
        base_dir=train_data_dir,
        csv_path=train_csv_path,
        idx_min=9000
    )
    test_dataset = CowDataset(
        base_dir="./cow/test/images/",
        csv_path="./cow/test/grade_labels.csv",
        has_labels=False,
    )
    # train
    experiment = Experiment(cfg)
    experiment.setup_dataset(train_dataset, val_dataset, cfg, dataloader=False)
    experiment.setup_experiment_from_cfg(cfg, setup_dataset=False)
    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        epochs=cfg["training"]["epochs"],
    )
    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
