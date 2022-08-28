# reference: https://github.com/open-mmlab/mmdetection/blob/master/tools/misc/download_dataset.py
import argparse
import os
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

try:
    import gdown
except ImportError:
    print("Gdown is not imported. Please install by `pip install gdown` if required.")

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument("--dataset-name", type=str, help="dataset name")
    parser.add_argument(
        "--save-dir", type=str, help="the dir to save dataset", default="data/coco"
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="whether unzip dataset or not, zipped files will be saved",
    )
    parser.add_argument(
        "--delete", action="store_true", help="delete the download zipped files"
    )
    parser.add_argument("--threads", type=int, help="number of threading", default=4)
    args = parser.parse_args()
    return args


def download(url, dir, download_from="url", unzip=True, delete=False, threads=1):
    def download_one(url, dir):
        if isinstance(url, tuple):
            url, output = url
            dir = dir / output
        os.makedirs(dir, exist_ok=True)
        f = dir / Path(url).name
        if download_from == "url":
            if Path(url).is_file():
                Path(url).rename(f)
            elif not f.exists():
                print("Downloading {} to {}".format(url, f))
                torch.hub.download_url_to_file(url, os.path.join(f), progress=True)
        elif download_from == "gdrive":
            f = gdown.download(id=url, output=str(dir) + "/", quiet=False)
            f = Path(f)
        else:
            raise ValueError("Invalid download type: {}".format(download_from))
        if unzip and f.suffix in (".zip", ".tar", ".gz"):
            print("Unzipping {}".format(f.name))
            if f.suffix == ".zip":
                ZipFile(f).extractall(path=dir)
            elif f.suffix == ".tar":
                TarFile(f).extractall(path=dir)
            elif f.suffix == ".gz":
                tarfile.open(f).extractall(dir)
            if delete:
                f.unlink()
                print("Delete {}".format(f))

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        # TODO: Support for downloading Panoptic Segmentation of COCO
        coco2017=[
            "http://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/zips/test2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ],
        mpii=[
            # images
            "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
            # mmpose annotations
            (
                "https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar",
                "annotations/",
            ),
        ],
        lvis=[
            "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",  # noqa
            "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",  # noqa
        ],
        voc2007=[
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",  # noqa
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",  # noqa
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar",  # noqa
        ],
        voc0712=[
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",  # noqa
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",  # noqa
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar",  # noqa
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        ],
        human36m_precomputed=[
            "https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz"
        ],
    )
    data2gdrive_id = dict(
        human36m_annotation=["1ztokDig-Ayi8EYipGE1lchg5XlAoLmwY"],
        human36m_images=[
            ("1AKpQOuRmsWgVJwHAPvUiXlnaXJdeBSYf", "images/"),
            ("1abMytHP_BdBaOMzkelRYf77jrTDZyfRZ", "images/"),
            ("1YvdnaMGqTcdgs4U4dYAFi2QxcmvSxCHc", "images/"),
            ("1alTStw-TWIhrtEEA3aJXqKzDkvl7Qes9", "images/"),
            ("1q69fYsAhlABXk_rFOUzDtNIEFWuYmJOP", "images/"),
            ("1gud5GEmFtlOwLabnIiE-s3EgFQjDppHH", "images/"),
            ("1hmvXEUYfqy8dhfZuPLRatXlLAk3lKrzR", "images/"),
        ],
    )

    if args.dataset_name in data2url:
        print("Dowloading from URLs")
        download_from = "url"
        download(
            data2url[args.dataset_name],
            dir=path,
            download_from=download_from,
            unzip=args.unzip,
            delete=args.delete,
            threads=args.threads,
        )

    if args.dataset_name in data2gdrive_id:
        print("Dowloading from Google Drive")
        download_from = "gdrive"
        download(
            data2gdrive_id[args.dataset_name],
            dir=path,
            download_from=download_from,
            unzip=args.unzip,
            delete=args.delete,
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
