# LightCollections⚡️: modular-pytorch-lightning-collections
[WIP] modular-pytorch-lightning, **WARNING: The repository is currently under development, and is unstable.**

What is `modular-pytorch-Lightning-Collections⚡`(LightCollections⚡️) for?
- Ever wanted to train `tresnetm50` models and apply TTA(test-time augmentation) or SWA(stocahstic weight averaging) to enhance performance? Apply sharpness-aware minimization to semantic segmentation models and measure the difference in calibration? LightCollections is a framework that utilize and connects existing libraries so experiments can be run effortlessly.
- Although many popular repositories provide great implementations of algorithms, they are often fragmented and tedious to use in cooperation. LightCollection wraps many existing repositories into components of `pytorch-lightning`. We aim to provide training procedures of various subtasks in Computer Vision with a collection of `LightningModule` and utilities for easily using model architecture, metrics, dataset, and training algorithms.
- The components can be used through our system or simply imported from outside to be used in your `pytorch` or `pytorch-lightning` project. Currently, the following frameworks are integrated into `LightCollections`:
  - `torchvision.models` for models, `torchvision.transforms` for transforms, optimizers and learning rate schedules from `pytorch`.
  - Network architecture and weights from `timm`.
  - Object detection frameworks and techniques from [`mmdetection`](https://github.com/open-mmlab/mmdetection)
  - Keypoint detection(pose estimation) frameworks and techniques from [`mmpose`](https://github.com/open-mmlab/mmpose)
  - `inagenet21k` [pretrained weights](https://github.com/Alibaba-MIIL/ImageNet21K) and feature to load model weights from url / `.pth` file.
  - `TTAch` for test-time augmentation.
  - Metrics implemented in `torchmetrics`.
  - WIP & future TODO:
    - Data augmentation from `albumentations`
    - Semantic segmentation models and weights from `mmsegmentation`

A number of algorithms and research papers are also adopted into our framework. Please refer to the examples below for more information.

# Quickstart

```
%cd /content
!git clone https://github.com/krenerd/awesome-modular-pytorch-lightning
%cd awesome-modular-pytorch-lightning

!pip install -r requirements.txt -q 
# (optional) use `wandb` to log progress.
!wandb login
```

## Running experiments with `train.py`

After installing required packages, you can run the following experiments on COLAB.

- CIFAR10 image classification with ResNet18.
```shell
!python train.py --name DUMMY-CIFAR10-ResNet18 --configs \
    configs/vision/classification/resnet-cifar10.yaml \
    configs/vision/models/resnet/resnet18-custom.yaml \
    configs/data/cifar10-kuangliu.yaml \
    configs/device/gpu.yaml \
    configs/utils/wandb.yaml \
    configs/utils/train.yaml
```

- Transfer learning experiments on Stanford Dogs dataset using TResNet-M
```shell
# uncomment when using TResNet models, this takes quite long
!pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12 -q

!python train.py --name TResNetM-StanfordDogs --config \
    configs/data/transfer-learning/training/colab-modification.yaml \
    configs/data/transfer-learning/training/random-init.yaml \
    configs/data/transfer-learning/cifar10-224.yaml \
    configs/data/augmentation/randomresizecrop.yaml \
    configs/vision/models/tresnetm.yaml \
    configs/device/gpu.yaml \
    configs/utils/wandb.yaml \
    configs/utils/train.yaml
```

- Object detection based on `FasterRCNN-FPN` and `mmdetection` on `voc0712` dataset.
```shell
# refer to: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
# install dependencies: (use cu113+torch1.12 because colab has CUDA 11.3)
# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .

# clone MPL
%cd /content
!git clone https://github.com/krenerd/awesome-modular-pytorch-lightning
%cd awesome-modular-pytorch-lightning
!pip install -r requirements.txt -q 

# setup voc07+12 dataset
!python tools/download_dataset.py --dataset-name voc0712 --save-dir data --delete --unzip

# run experiment
!python train.py --name voc0712-FasterRCNN-FPN-ResNet50 --config \
    configs/vision/object-detection/mmdet/faster-rcnn-r50-fpn-voc0712.yaml \
    configs/vision/object-detection/mmdet/mmdet-base.yaml \
    configs/data/voc0712-mmdet-no-tta.yaml \
    configs/data/voc0712-mmdet.yaml \
    configs/device/gpu.yaml \
    configs/utils/wandb.yaml \
    configs/utils/train.yaml
```

- 2D Pose estimation based on `HRNet` and `mmdetection` in `MPII` dataset.
```shell
# train HRNet pose estimation on MPII
# refer to: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
# install dependencies: (use cu113+torch1.12 because colab has CUDA 11.3)
# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# Install mmdetection
!rm -rf mmpose
!git clone https://github.com/open-mmlab/mmpose.git
%cd mmpose
!pip install -e .

# clone MPL
%cd ..

# setup voc07+12 dataset
!python tools/download_dataset.py --dataset-name mpii --save-dir data/mpii --delete --unzip

# run experiment
!python train.py --name MPII-HRNet32 --config \
    configs/vision/pose/mmpose/hrnet_w32_256x256.yaml \
    configs/vision/pose/mmpose/mmpose-base.yaml \
    configs/data/pose-2d/mpii-hrnet.yaml \
    configs/device/gpu.yaml \
    configs/utils/wandb.yaml \
    configs/utils/train.yaml

```

- Supervised VideoPose3D 3D-pose estimation on `Human3.6M` dataset
```
!python tools/download_dataset.py --dataset-name human36m_annotation --unzip --save-dir human36m --delete --unzip

!python train.py --name Temporal-baseline-bs1024-lr0.001 --config \
    configs/vision/pose-lifting/temporal.yaml \
    configs/data/human36/temproal-videopose3d.yaml \
    configs/data/human36/normalization.yaml \
    configs/device/gpu.yaml \
    configs/utils/wandb.yaml \
    configs/utils/train.yaml
```

## `Experiment` and `catalog`

LightCollections can also be used as a library for extending your pytorch lightning code. `train.py` simply conveys the config file to the `Experiment` class defined in `main.py` to build components such as dataset, dataloaders, models, and callbacks, which in tern uses components defined in `catatlog`.

```Python
    ...
    experiment = Experiment(cfg)
    experiment.initialize_environment(cfg=cfg)
    datasets = experiment.setup_dataset(
        dataset_cfg=cfg["dataset"],
        transform_cfg=cfg["transform"],
    )
    dataloaders = experiment.setup_dataloader(
        datasets=datasets,
        dataloader_cfg=cfg["dataloader"],
    )
    train_dataloader, val_dataloader = dataloaders["trn"], dataloaders["val"]
    model = experiment.setup_model(model_cfg=cfg["model"], training_cfg=cfg["training"])
    logger_and_callbacks = experiment.setup_callbacks(cfg=cfg)
    ...
```

You don't neccessarily need to create every component using the `Experiment` class. For example, if you wish to use a custom dataset instead, you can skip `experiment.setup_dataset` and feed your custom dataset to `experiment.setup_dataloader`. The `Experiment` class simply manages constant global variables such as label map, normalization mean and standard deviation, and manages a common log directory to conveniently create the components.

In an example implemented in `tools/hyperparameter_sweep.py`, I was able to implement hyperparameter sweeping using the `Experiment` class.

## How does config files work?

Training involves many configs. `LightCollections` implements a cascading config system where we use multiple layers of config
files to define differnt parts of the experiment. For example, in the CIFAR10 example above, we combine 6 config files.
```shell
configs/vision/classification/resnet-cifar10.yaml \
configs/vision/models/resnet/resnet18-custom.yaml \
configs/data/cifar10-kuangliu.yaml \
configs/device/gpu.yaml \
configs/utils/wandb.yaml \
configs/utils/train.yaml
```
Each layer defines something different, such as the dataset, network architecture, or training procedure. If we wanted to use a `ResNet50` model instead, we may replace
```shell
configs/vision/models/resnet/resnet18-custom.yaml
-> configs/vision/models/resnet/resnet50-custom.yaml
```
these cascading config files are baked at the start of `train.py`. Configs in front have higher priority. These baked config files are logged under `configs/logs/{experiment_name}.(yaml/pkl/json)` for logging and reproduction purpose.

## Components from `catalog`

- If not implemented yet, you may take an instance of `main.py: Experiment` and override any part of it.
- Training procedure (`LightningModule`): List of available training procedures are listed in `lightning/trainers.py`
- Model architectures:
  - Backbone models implemented in `torchvision.models` can be used.
  - Backbone models implemented in `timm` can be used.
  - Although we highly recommend using `timm`, as they provide a large variaty of computer vision models and their models are throughly evaluated, custom implementations of some architectures are listed in `catalog/models/__init__.py`.
- Dataset
  - Dataset: currently only `torchvision` datasets are supported by `Experiment`, however `torchvision.datasets.ImageFolder` can be used to load from custom dataset. In addition, you may just use a custom dataset and combine it with the transforms, model and training feature of the repo.
  - Transformations(data augmentation): Transforms must be listed in one in [`data/transforms/vision/__init__.py`]
- Other features
  - Optimizers
  - Metrics / loss

### Tip for reproducing experiments

The results of experiments such as model checkpoints, logs, and the config file used to run the experiment is logged under `awesome-modular-pytorch-lightning/results/{exp_name}`. In particular, the `results/{exp_name}/configs/cfg.yaml` file which contains the config file can be useful when reproducing experiments or rechecking hyperparameters.

# List of algorithms implemented

## Network architecture

For computer vision models, we recommend borrowing architecture from `timm` as they provide robust implementations and pretrained weights for a wide variety of architectures. We remove the classification head from the original models.

### [timm](https://github.com/rwightman/pytorch-image-models)

[`timm`](https://github.com/rwightman/pytorch-image-models) provides a wide variety of architectures for computer vision. `timm.list_models()` returns a complete list of available models in timm. Models can be created with `timm.create_model()`.

An example of creating a `resnet50` model using `timm`:
```
model = timm.create_model("resnet50", pretrained=True)
```

To use `timm` models,
- set `model.backbone.name` to `TimmNetwork`.
- set `model.backbone.args.name` to the model name.
- set additional arguments in `model.backbone.cfg`.
- Refer to: `configs/vision/models/resnet/resnet50-timm.yaml`
```
model:
  backbone:
    name: "TimmNetwork"
    args:
      name: "resnet50"
      args:
        pretrained: True
    out_features: 2048
```


### torchvision.models

`torchvision.models` also provide a number of architectures for computer vision. The list of models can be found [here](https://pytorch.org/vision/stable/models.html).

An example of creating a `resnet50` model using `torchvision`:
```
model = torchvision.models.resnet50()
```

To use `timm` models,
- set `model.backbone.name` to `TorchvisionNetwork`.
- set `model.backbone.args.name` to the model name.
- set additional arguments in `models.backbone.args`.
- set `model.backbone.drop_after` to only use feature extractor.
- Refer to: `configs/vision/models/resnet/resnet50-torchvision.yaml`
```
model:
  backbone:
    name: "TorchvisionNetwork"
    args:
      name: "resnet50"
      args:
        pretrained: False
      drop_after: "avgpool"
    out_features: 2048
```

## Data Augmentation

### RandomResizeCrop(ImageNet augmentation)

- Paper: <https://arxiv.org/abs/1409.4842>
- Note: Common data augmentation strategy for ImageNet using RandomResizedCrop.
- Refer to: `configs/data/augmentation/randomresizecrop.yaml`

```yaml
transform: [
    [
      "trn",
      [
        {
          "name": "RandomResizedCropAndInterpolation",
          "args":
            {
              "size": 224,
              "scale": [0.08, 1.0],
              "ratio": [0.75, 1.3333],
              "interpolation": "random",
            },
        },
        {
          "name": "TorchvisionTransform",
          "args": { "name": "RandomHorizontalFlip" },
        },
        # more data augmentation (rand augment, auto augment, ...)
      ],
    ],
    [
      # standard approach to use images cropped to the central 87.5% for validation
      "val,test",
      [
        {
          "name": "Resize",
          "args": { "size": [256, 256], "interpolation": "bilinear" },
        },
        {
          "name": "TorchvisionTransform",
          "args": { "name": "CenterCrop", "args": { "size": [224, 224] } },
        },
        # more data augmentation (rand augment, auto augment, ...)
      ],
    ],
    [
      "trn,val,test",
      [
        { "name": "ImageToTensor", "args": {} },
        {
          "name": "Normalize",
          "args":
            {
              "mean": "{const.normalization_mean}",
              "std": "{const.normalization_std}",
            },
        },
      ],
    ],
  ]
```

### RandAugment

- Paper: <https://arxiv.org/abs/1909.13719>
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/data/augmentation/randaugment.yaml`

```yaml
transform:
...
        {
          "name": "TorchvisionTransform",
          "args":
            { 
              "name": "RandAugment",
              "args": { "num_ops": 2, "magnitude": 9 } 
            },
        },
...
```
Refer to: `configs/algorithms/data_augmentation/randaugment.yaml`

### TrivialAugmentation
- Paper: <https://arxiv.org/abs/2103.10158>
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/data/augmentation/trivialaugment.yaml`
```yaml
transform:
...
        {
          "name": "TorchvisionTransform",
          "args":
            {
              "name": "TrivialAugmentWide",
              "args": { "num_magnitude_bins": 31 },
            },
        },
...
```

### Mixup
- Paper: <https://arxiv.org/abs/1710.09412>
- Note: Commonly used data augmentation strategy for image classification. As the labels are continuous values, the loss function should be modified accordingly.
- Refer to: `configs/data/augmentation/mixup/mixup.yaml`
```yaml
training:
  mixup_cutmix:
    mixup_alpha: 1.0
    cutmix_alpha: 0.0
    cutmix_minmax: null
    prob: 1.0
    switch_prob: 0.5
    mode: "batch"
    correct_lam: True
    label_smoothing: 0.1
    num_classes: 1000

model:
  modules:
    loss_fn:
      name: "SoftTargetCrossEntropy"
```

### CutMix
- Paper: <https://arxiv.org/abs/2103.10158>
- Note: Commonly used data augmentation strategy for image classification. As the labels are continuous values, the loss function should be modified accordingly.
- Refer to: `configs/data/augmentation/mixup/cutmix.yaml`
```yaml
training:
  mixup_cutmix:
    mixup_alpha: 0.0
    cutmix_alpha: 1.0
    cutmix_minmax: null
    prob: 1.0
    switch_prob: 0.5
    mode: "batch"
    correct_lam: True
    label_smoothing: 0.1
    num_classes: "{const.num_classes}"

model:
  modules:
    loss_fn:
      name: "SoftTargetCrossEntropy"
```

### CutOut
- Paper: <https://arxiv.org/abs/1708.04552>
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/data/augmentation/cutout.yaml` and `configs/data/augmentation/cutout_multiple.yaml`
```yaml
transform:
...
    [
      "trn",
      [
        ...
        {
          # CutOut!!
          "name": "CutOut",
          "args": { "mask_size": 0.5, "num_masks": 1 },
        },
      ],
    ],
...
```

### `timm` mixup-cutmix

- Paper: <https://arxiv.org/abs/2110.00476>
- Note: By default, models trained in `timm` randomly switches between `Mixup` and `CutMix` data augmentation. This is found to be effective in their paper.
- Refer to: `configs/data/augmentation/mixup/mixup_cutmix.yaml`
```yaml
training:
  mixup_cutmix:
    mixup_alpha: .8
    cutmix_alpha: 1.0
    cutmix_minmax: null
    prob: 1.0
    switch_prob: 0.5
    mode: "batch"
    correct_lam: True
    label_smoothing: 0.1
    num_classes: 1000
model:
  modules:
    loss_fn:
      name: "SoftTargetCrossEntropy"
```

## Regularization

### Label smoothing
- Note: Commonly used regularization strategy.
- Refer to: <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
```
model:
  modules:
    loss_fn:
      name: "CrossEntropyLoss"
      args:
        label_smoothing: 0.1

```

### Weight decay
- Note: Commonly used regularization strategy.
- Refer to: [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) and `configs/vision/classification/resnet-cifar10.yaml`
```yaml
training:
  optimizer_cfg:
    weight_decay: 0.0005
```

### DropOut(classification)
- Paper: <https://jmlr.org/papers/v15/srivastava14a.html>
- Note: Commonly used regularization strategy.
- Refer to: `configs/vision/classification/resnet-cifar10.yaml`
```yaml
model:
  modules:
    classifier:
      args:
        dropout: 0.2
```

### R-Drop(classification)
- Paper: <https://arxiv.org/abs/2106.14448>
- Note: Regularization strategy that minimizes the KL-divergence between the output distributions of two sub-models sampled by dropout.
- Refer to: `configs/algorithms/rdrop.yaml`
```yaml
training:
  rdrop:
    alpha: 0.6
```

## Loss functions

### Sharpness-aware minimization(SAM)
- Paper: <https://arxiv.org/abs/2010.01412>
- Note: Sharpness aware minimization aims at finding flat minimas. It is demonstrated to improve training speed, generalization, robustness to label noise. However, two backpropagation is needed at every opimization step which doubles the training time.
- Refer to: `configs/algorithms/sharpness-aware-minimization.yaml`
```yaml
training:
  sharpness-aware:
    rho: 0.05
```

### PolyLoss
- Paper: <https://arxiv.org/abs/2204.12511>
- Note: The authors derive the taylor expansion of cross entropy and demonstrate that modifying the coefficient of the first-order term can improve performance.
- Refer to: `configs/algorithms/loss/poly1.yaml`
```yaml
model:
  modules:
    loss_fn:
      name: "PolyLoss"
      file: "loss"
      args:
        eps: 2.0
```

### One-to-all BCE loss for classification

- Paper: <https://arxiv.org/abs/2110.00476>
- Note: The authors show that BCE loss can be used for classification tasks and shows similar or better performance.
- Refer to: `configs/algorithms/loss/classification_bce.yaml`
```yaml
model:
  modules:
    loss_fn:
      name: "OneToAllBinaryCrossEntropy"
      file: "loss"
```

## Knowledge Distillation (WIP)

### Vanila knowlege distillation

(TODO)
- Paper: <https://arxiv.org/abs/1503.02531>
- Note: Distill knowledge from large network to small network by minimizing the KL divergence of the teacher and student prediction.
- Refer to:
```
TODO
```

## Training

### Gradient Clipping
- Note: Used to preventing gradient explosion and stabilize the training by clipping large gradients. Recently, it is demonstrated to have a number of benefits.
- Refer to: `configs/algorithms/optimizer/gradient_clipping.yaml` and `configs/algorithms/optimizer/gradient_clipping_maulat_optimization.yaml`
```
trainer:
  gradient_clip_val: 1.0
```

### Stocahstic Weight Averaging(SWA)
- Paper: <https://arxiv.org/abs/1803.05407>
- Note: Average multiple checkpoints during training for better performance. An awesome overview of the algorithm is provided by [pytorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/). Luckily, `pytorch-lightning` provides an easy-to-use callback that implements SWA. To train a SWA model from an existing checkpoint, you may set `swa_epoch_start: 0.0`.
- Refer to: `configs/algorithms/swa.yaml`
```
callbacks:
  StochasticWeightAveraging:
    name: "StochasticWeightAveraging"
    file: "lightning"
    args:
      swa_lrs: 0.02 # typicall x0.2 ~ x0.5 of initial lr
      swa_epoch_start: 0.75
      annealing_epochs: 5 # smooth the connection between lr schedule and SWA.
      annealing_strategy: "cos"
      avg_fn: null
```

## Pre-training

### Fine-tuning from a pretrained feature extractor backbone

1. `timm` or `torchvision` models have an argument called `pretrained` for loading a pretrained feature extractor(typically ImageNet trained).

2. To load from custom checkpoints, you can specify a url or path to the state dict in `model.backbone.weights`. For example, `configs/vision/models/imagenet21k/imagenet21k_resnet50.yaml` loads ImageNet21K checkpoints from a url proposed in the paper: [ImageNet-21K Pretraining for the Masses](https://github.com/Alibaba-MIIL/ImageNet21K). Path to state dict can be provided in `model.backbone.weights.state_dict_path` instead of the url. This is implemented in `lightning/base.py:L23`
```yaml
model:
  backbone:
    name: "TimmNetwork"
    args:
      name: "resnet50"
      args:
        pretrained: False
    out_features: 2048

    weights:
      is_ckpt: True
      url: "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth"
      # alternatively, `state_dict_path: {PATH TO STATE DICT}`
```

### Loading and resuming from checkpoint

To resume from a checkpoint, provide the path to state dict in `model.state_dict_path`. Checkpoints generated using `ModelCheckpoint` callback contain state dict inside the `state_dict` key while saving using `torch.save(model.state_dict())` directly saves the state dict. The `is_ckpt` argument should be true if the state dict is generated through the `ModelCheckpoint` callback.

```yaml
model:
  is_ckpt: True # True / False according to the type of the state dict.
  state_dict_path: {PATH TO STATE DICT}
```

## Learning rate schedule

### 1-cycle learning rate schedule

- Paper: <https://arxiv.org/abs/1708.07120>
- Note: Linearly increases learning rate from 0 to maximum for the first half of training then linearly decreases to 0. Commonly used learning rate schedule.
- Refer to: `configs/algorithms/lr-schedule/1cycle.yaml` and [torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
```
training:
  lr_scheduler:
    name: "1cycle"
    args:
      # Original version updates per-batch but we modify to update pre-epoch.
      pct_start: 0.3
      max_lr: "{training.lr}"
      anneal_strategy: "linear"
      total_steps: "{training.epochs}"
    cfg:
      interval: "epoch"
```

### Cosine learning rate decay
- Paper: <https://arxiv.org/abs/1608.03983>
- Note: Decays the learning from the initial value to 0 via a cosine function. Commonly used learning rate schedule.
- Refer to: `configs/vision/classification/resnet-cifar10.yaml` and [torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
```
training:
  lr_scheduler:
    name: "cosine"
    args:
      T_max: "{training.epochs}"
    cfg:
      interval: "epoch"
```

### WarmUp
- Paper: <https://arxiv.org/abs/2103.10158>
- Note: Commonly used strategy to stabilize training at early stages.
- Refer to: `configs/algorithms/lr-schedule/warmup.yaml`
```yaml
training:
  lr_warmup:
    multiplier: 1
    total_epoch: 5
```

## Metrics

### torchmetrics

## Test-time augmentation(TTA)

- Refer to: `configs/algorithms/tta/hvflip.yaml`
```yaml
model:
  tta:
    name: "ClassificationTTAWrapper"
    args:
      output_label_key: "logits"
      merge_mode: "mean"
      transforms:
        - name: "HorizontalFlip"
        - name: "VerticalFlip"
```
- and `configs/algorithms/tta/rotation.yaml`
```yaml
model:
  tta:
    name: "ClassificationTTAWrapper"
    args:
      merge_mode: "mean"
      transforms:
        - name: "HorizontalFlip"
        - name: "Rotation"
          args:
            angles:
              - 0
              - 30
              - 60
              - 90
              - 120
              - 150
              - 180
              - 210
              - 240
              - 270
              - 300
              - 330
```

# Overview

## LightningModules

### ClassificationTrainer

## Data
Currently `torchvision` datasets are supported by `Experiment`, however you could use `torchvision.datasets.ImageFolder` to load from custom dataset.

### Transformations(data augmentation): Transforms must be listed in one in [`data/transforms/vision/__init__.py`]


## Utils

### Optimizers

## About me

- Contact: sieunpark77@gmail.com / sp380@student.london.ac.uk / +82) 10-6651-6432
