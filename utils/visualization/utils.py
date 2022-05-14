import os

import numpy as np
import matplotlib.pyplot as plt
from utils.experiment import makedir

from data.transforms.vision import ToPIL, UnNormalize
from .vision import plot_image_classification, plot_object_detection


function_for_plotting = {
    "image classification": plot_image_classification,
    "object detection": plot_object_detection,
}


def plot_sample(task, x, y=None, mode="save", savedir="results/example.png"):
    get_image = function_for_plotting[task]
    image = get_image(x, y)

    assert mode in ["save", "return"]
    if mode == "save":
        plt.imshow(image)
        plt.axis("off")
        plt.savefig()
    elif mode == "return":
        return image
    else:
        raise ValueError(f"Invalid value {mode} for `mode`.")


def plot_samples_from_dataset(
    dataset,
    task,
    subplot_dim=(5, 5),
    save_to="results/samples_vis.png",
    root_dir="",
    image_tensor_to_numpy=False,
    unnormalize=False,
    normalization_mean=(0.5, 0.5, 0.5),
    normalization_std=(0.5, 0.5, 0.5),
    imsize=3,
    preprocess_f=None,
):
    """
    Plot multiple samples using matplotlib.
    TODO: write docstring.
    Parameters
    """
    save_to = os.path.join(root_dir, save_to)
    makedir(save_to)

    w, h = subplot_dim
    plt.figure(figsize=(w * imsize, h * imsize))
    for i in range(1, w * h + 1):
        x, y = dataset[i - 1]

        if preprocess_f:
            x, y = preprocess_f(x, y)
        if unnormalize:
            x = UnNormalize(normalization_mean, normalization_std)(x)
        if image_tensor_to_numpy:
            x = ToPIL()(x)
            np.asarray(x)

        plt.subplot(w, h, i)
        plot_image = plot_sample(task, x, y, mode="return")
        plt.imshow(plot_image)
        plt.axis("off")

    plt.savefig(save_to)
    plt.close()
