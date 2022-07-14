import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from catalog.transforms import UnNormalize
from utils.experiment import makedir

from .vision import plot_image_classification, plot_object_detection

function_for_plotting = {
    "image classification": plot_image_classification,
    "object detection": plot_object_detection,
}


def plot_sample(
    task, data, mode="save", savedir="results/example.png", label_map=None, **kwargs
):
    get_image = function_for_plotting[task]
    image = get_image(label_map=label_map, **data, **kwargs)

    assert mode in ["save", "return"]
    if mode == "save":
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(savedir)
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
    resize_to=None,
    plot_size=3,
    preprocess_f=None,
    label_map=None,
    **kwargs,
):
    """
    Plot multiple samples using matplotlib.
    TODO: write docstring.
    Parameters
    """
    save_to = os.path.join(root_dir, save_to)
    makedir(save_to)

    w, h = subplot_dim
    plt.figure(figsize=(w * plot_size, h * plot_size))
    for i in range(1, w * h + 1):
        data = dataset[i - 1]

        if preprocess_f:
            data = preprocess_f(data)
        if resize_to:
            data["images"] = TF.resize(
                data["images"], resize_to, interpolation=TF.InterpolationMode.NEAREST
            )
        if unnormalize:
            data = UnNormalize(normalization_mean, normalization_std)(data)
        if image_tensor_to_numpy:
            data["images"] = np.asarray(TF.to_pil_image(data["images"]))

        plt.subplot(w, h, i)
        plot_image = plot_sample(
            task, data=data, mode="return", label_map=label_map, **kwargs
        )
        plt.imshow(plot_image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()
