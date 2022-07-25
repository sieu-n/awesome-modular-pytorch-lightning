import os
import random
import shutil
from argparse import ArgumentParser

import pandas as pd
import torch
from main import Experiment
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.configs import merge_config, read_configs
from utils.experiment import apply_transforms, build_transforms


class CowDataset(Dataset):
    def __init__(
        self,
        base_dir,
        csv_path,
        keep_order=False,
        has_labels=True,
        idx_min=0,
        idx_max=None,
        shuffle_seed=42,
        image_name_key="imname",
    ):
        self.base_dir = base_dir
        self.has_labels = has_labels

        df = pd.read_csv(csv_path)
        indicies = list(df.index)
        if has_labels and not keep_order:  # in order to randomly select validation set
            random.Random(shuffle_seed).shuffle(indicies)

        if idx_max is None:
            indicies = indicies[idx_min:]
        else:
            indicies = indicies[idx_min:idx_max]

        self.image_names = list(df.iloc[indicies][image_name_key])
        if self.has_labels:
            self.grade = list(df.iloc[indicies]["grade"])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.base_dir + self.image_names[idx])
        if self.has_labels:
            grade = self.grade[idx]
            return {"images": image, "labels": grade}
        else:
            return {"images": image}


def save_predictions_to_csv(image_names, pred, label_map, filename="prediction.csv"):
    grades = [label_map[x] for x in pred]
    df = pd.DataFrame(data={"id": image_names, "grade": grades})
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--root_dir", type=str, default="")

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    if args.name is not None:
        cfg["name"] = args.name
    if args.group:
        cfg["wandb"]["group"] = args.group
    if args.offline:
        cfg["wandb"]["offline"] = True

    # move data
    base_path = "/content/drive/MyDrive/data/cow_classification/"
    sample_submission = (
        "/content/drive/MyDrive/data/cow_classification/sample_submission.csv"
    )
    target_path = "./cow/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for subset in ("train", "test"):
        shutil.unpack_archive(
            f"{base_path}{subset}.zip", f"{target_path}{subset}", "zip"
        )
    shutil.copyfile(sample_submission, "./cow/test_order.csv")

    # make dataset
    train_data_dir = "./cow/train/images/"
    train_csv_path = "./cow/train/grade_labels.csv"
    train_dataset = CowDataset(
        base_dir=train_data_dir, csv_path=train_csv_path, idx_max=9000
    )
    val_dataset = CowDataset(
        base_dir=train_data_dir, csv_path=train_csv_path, idx_min=9000
    )
    # test dataset
    test_dataset = CowDataset(
        base_dir="./cow/test/images/",
        csv_path="./cow/test_order.csv",
        has_labels=False,
        image_name_key="id",
    )
    test_image_names = test_dataset.image_names
    transforms = build_transforms(
        transform_cfg=cfg["transform"],
        const_cfg=cfg["const"],
        subset_keys=["pred"],
    )["pred"]
    test_dataset = apply_transforms(test_dataset, None, transforms)
    # train
    experiment = Experiment(cfg)
    experiment.initialize_environment(cfg)
    experiment.setup_dataset(train_dataset, val_dataset, cfg, dataloader=False)
    experiment.setup_experiment_from_cfg(cfg, setup_env=False, setup_dataset=False)

    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        root_dir=args.root_dir,
        epochs=cfg["training"]["epochs"],
    )

    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
    val_dataloader_cfg = merge_config(
        cfg["dataloader"]["base_dataloader"], cfg["dataloader"]["val"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["validation"]["batch_size"],
        **val_dataloader_cfg,
    )
    predictions = experiment.predict(test_dataloader, trainer_cfg=cfg["trainer"])
    predictions = torch.argmax(torch.cat(predictions), dim=1)

    print("saving file under:", experiment.get_directory() + "prediction.csv")
    save_predictions_to_csv(
        image_names=test_image_names,
        pred=predictions,
        label_map=cfg["const"]["label_map"],
        filename=os.path.join(args.root_dir, "prediction.csv"),
    )
