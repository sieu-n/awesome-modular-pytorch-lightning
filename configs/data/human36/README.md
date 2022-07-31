# Human3.6M dataset for HPE

Download and extract dataset from [link](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK) under `awesome-modular-pytorch-lightning/human36m`. If the file is saved in another directory, modify `dataset.dataset_base_cfg.base_dir` in the config file to point to that directory.

Or use `tools/download_dataset.py` to download the dataset.
```shell
python tools/download_dataset.py --dataset-name human36m_annotation --unzip --save-dir human36m --delete --unzip
python tools/download_dataset.py --dataset-name human36m_images --unzip --save-dir human36m --delete --unzip
```