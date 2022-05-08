# LightCollections⚡️: modular-pytorch-lightning-collections
[WIP] modular-pytorch-lightning, **WARNING: The repository is currently under development, and is unstable.**

If you are interested in participating, please contact `sieunpark77@gmail.com`!!!

What is `modular-pytorch-Lightning-Collections⚡`(LightCollections⚡️) for?
- LightCollection aims at extending the features of `pytorch-lightning`. We provide training procedures of various subtasks in Computer Vision with a collection of `LightningModule` and utilities for easily using model architecture, metrics, and dataset.
- One of the designing principles of `modular-pytorch-lightning` is to make it convinient to be able to use every component(function and classes) independently of the repository. To achive this, I avoided un-essential dependancies on most of the code, by avoiding too much abstraction and subclassing in particular.

I felt that the design of `pytorch-lightning` is very effective in this terms.

This repository is designed to utilize many amazing and robust and open-source projects such as `timm`, `pytorch`, and more.

Notes(rules) for development
- The development is mainly focused in implementing various training procedures as `pl.LightningModule`. Other components such as model architecture is usually adopted from other repository, just for the purpose of testing.
- The development of this repository should stick towards the [PEP](https://peps.python.org/)(Python Enhancement Proposals) conventions and `flake8` linting.
- Function or class docstrings must follow Numpy [style guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
- The performance of every implementation has to be validated before being merged into `main`. The results and config used to reproduce the results should be presented in the relevant `README.MD` file.

## Links to track progress

- Project Dashboard: [\[Trello\]](https://trello.com/b/AnOjqk1F/awesome-modular-pytorch-lightning-development)
- Overview: <https://docs.google.com/document/d/1qqisfpLgEUqgGw1-5WjmaV5Px-y5n8UrBJKVhZfStfE/edit>

## Progress

### Training procedure (`LightningModule`)

### Model architectures

### Dataset

### Other features

#### Optimizers

#### Metrics / loss

### Timeline

- 220508 | Working version of the repository on `Image Classification`.
- 220504 | Create repository! Start of `awesome-modular-pytorch-lightning`.

## About me

- Contact: sieunpark77@gmail.com / sp380@student.london.ac.uk
- @opentowork, internships oppertunities:)
