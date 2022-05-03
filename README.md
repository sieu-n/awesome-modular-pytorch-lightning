# awesome-modular-pytorch-lightning⚡️
[WIP] awesome-modular-pytorch-lightning, **WARNING: The repository is currently under development, and is unstable.**

If you are interested in participating, please contact `sieunpark77@gmail.com`!!!

What is `Awesome-Modular-pytorch-Lightning⚡️`(AML) for?
- AML is a repo based on `pytorch-lightning`, which currently supports a number of popular subtasks and training procedures in Computer Vision. I hope people can use this repo to easily experiment their ideas on the most modern training procudures for a fair comparison.
- While designing this repo, I really wanted to make it convinient so that every component(function and classes) of the repo could easily be copied and used in another repo and run standalone. To achive this, I try to avoid making dependancies on ANY of the code, by avoiding the use interface-classes and subclassing.
- Please feel free to suggest changes to the design of the repo, as I am not the best coder or researcher you might think of 🙂

I felt that the design of `pytorch-lightning` is a very effective method to be less abstract in terms of implementation while being flexible to many deep-learning algorithms. 
Although popular repos such as `MM{Detection, ...}` or `detectron` already has implementaion of many popular frameworks often implemented by the authors of the paper and other
great contributors, some repos are too complicated or abstract to be able to actually look at the implementation of algorithms. 

The motivation of this repo is:
- Abstract implementations in comprehensive repos ㅁㄴ above might be challenging especially to beginners and learners(like me)
- Sometimes, you just want to check the implementation detail of the code, but don't want to dig through the entire repository.
- There is no `MM{Detection, ...}` for `pytorch-lightning`, so why no make one!
In the other side, many reproductions don't discuss the performance, and there might be bugs or differences in other hyperparameters. Despite limitations in GPU compute, we wanted to be robust in terms of the integrity of implementations.

This repository is designed to utilize many amazing and robust and open-source projects such as `timm`, `pytorch`, and more. 

Notes(rules) for development
- The development of this repo should stick towards the [PEP](https://peps.python.org/)(Python Enhancement Proposals) conventions and `flake8` linting.
- Function or class docstrings must follow Numpy [style guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
- The performance of every implementation has to be validated before being merged into `main`. The results and config used to reproduce the results should be presented in the relevant `README.MD` file.

## Links to track progress:

- Project Trello \[Dashboard\]: https://trello.com/b/AnOjqk1F/awesome-modular-pytorch-lightning-development
- Overview: https://docs.google.com/document/d/1qqisfpLgEUqgGw1-5WjmaV5Px-y5n8UrBJKVhZfStfE/edit

## Progress


### Timeline

- 220504 | Create repo! Start of `awesome-modular-pytorch-lightning`.

## About me

- Contact: sieunpark77@gmail.com / sp380@student.london.ac.uk
- @opentowork, internships oppertunities:)
