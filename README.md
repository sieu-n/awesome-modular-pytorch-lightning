# LightCollections⚡️: modular-pytorch-lightning-collections
[WIP] modular-pytorch-lightning, **WARNING: The repository is currently under development, and is unstable.**

If you are interested in participating, please contact `sieunpark77@gmail.com`!!!

What is `modular-pytorch-Lightning-Collections⚡`(LightCollections⚡️) for?
- LightCollection aims at extending the features of `pytorch-lightning`. We provide training procedures of various subtasks in Computer Vision with a collection of `LightningModule` and utilities for easily using model architecture, metrics, and dataset.
- One of the designing principles of `modular-pytorch-lightning` is to make it convinient to be able to use every component(function and classes) independently of the repository. To achive this, I avoided un-essential dependancies on most of the code, by avoiding too much abstraction and subclassing in particular.
- Also, I felt that there were some details in the code that were not so obvious but hard to find details such as how images are preprocessed to the same image size when using the `ImageNet` classification dataset or during object detection. We made these details transparant by defining everything in the config files. 

I felt that the design of `pytorch-lightning` is very effective in this terms.

This repository is designed to utilize many amazing and robust and open-source projects such as `timm`, `pytorch`, and more.

Notes(rules) for development
- The development is mainly focused in implementing various training procedures as `pl.LightningModule`. Other components such as model architecture is usually adopted from other repository, just for the purpose of testing.
- The development of this repository should stick towards the [PEP](https://peps.python.org/)(Python Enhancement Proposals) conventions and `flake8` linting.
- Function or class docstrings must follow Numpy [style guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
- The performance of every implementation has to be validated before being merged into `main`. The results and config used to reproduce the results should be presented in the relevant `README.MD` file.


## How to run experiments
1. Run experiments using `train.py`

- CIFAR10 image classification with `ResNet18`.
```
!python train.py --config configs/vision/training/resnet-cifar10.yaml configs/vision/models/resnet/resnet18-custom.yaml configs/vision/data/cifar10.yaml configs/utils/wandb.yaml configs/utils/train.yaml
```

2. Use the `Experiment` class to run complex experiments for research, hyperparameter sweep, ...:

For example, please have a look at the `study/dataset_size_experiment.py`.

### How does config files work?

Training involves many configs. `LightCollections` implements a cascading config system where we use multiple layers of config
files to define differnt parts of the experiment. For example, in the CIFAR10 example above, we use 5 config files.
```
configs/vision/training/resnet-cifar10.yaml
configs/vision/models/resnet/resnet18-custom.yaml
configs/vision/data/cifar10.yaml
configs/utils/wandb.yaml
configs/utils/train.yaml
```
here, if we want to log to `TensorBoard` instead of `wandb`, you may replace
```
configs/utils/wandb.yaml
-> configs/utils/tensorboard.yaml
```
these cascading config files are baked at the start of `train.py`, where configs are overriden in inverse order. These baked config files are logged under `configs/logs/{experiment_name}.(yaml/pkl/json)` for logging purpose.

## Overview

- If not implemented yet, you may take an instance of `main.py: Experiment` and override any part of it.
- Training procedure (`LightningModule`): List of available training procedures are listed in `lightning/trainers.py`
- Model architectures:
  - Backbone models implemented in `torchvision.models` can be used.
  - Backbone models implemented in `timm` can be used.
  - Although we highly recommend using `timm`, as it is throughly evaluated and managed, custom implementations of some architectures are listed in `models/backbone/__init__.py`.
- Dataset
  - Dataset: currently only `torchvision` datasets are supported by `Experiment`, however you could use `torchvision.datasets.ImageFolder` to load from custom dataset.
  - Transformations(data augmentation): Transforms must be listed in one in [`data/transforms/vision/__init__.py`]
- Other features
  - Optimizers
  - Metrics / loss

## Timeline

- 220517 | Create object detection dataset.
- 220517 | Implementation of ResNet-D and PreAct-ResNet.
- 220508 | Working version of the repository on `Image Classification`.
- 220504 | Create repository! Start of `awesome-modular-pytorch-lightning`.

### Tracking progress

- Project Dashboard: [\[Trello\]](https://trello.com/b/AnOjqk1F/awesome-modular-pytorch-lightning-development)

## About me

- Contact: sieunpark77@gmail.com / sp380@student.london.ac.uk / +82) 10-6651-6432
- @opentowork, internships oppertunities:)
