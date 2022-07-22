import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from catalog.transforms import MMdetDataset2Torchvision, UnNormalize
from torchvision.transforms import Compose
from utils.experiment import makedir

from .vision import plot_image_classification, plot_object_detection

function_for_plotting = {
    "image classification": plot_image_classification,
    "object detection": plot_object_detection,
}
preprocessor_factory = {
    "convert-mmdetbbox": [MMdetDataset2Torchvision(to_xywh=False)],
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
    image_tensor_to_numpy=True,
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

    # build preprocessor
    if preprocess_f in preprocessor_factory.keys():
        print(f"Found preprocessor `{preprocess_f}`")
        preprocess_f = Compose(preprocessor_factory[preprocess_f])

    w, h = subplot_dim
    plt.figure(figsize=(w * plot_size, h * plot_size))
    for i in range(1, w * h + 1):
        data = dataset[i - 1]

        if preprocess_f is not None:

            data = preprocess_f(data)
        if resize_to:
            data["images"] = TF.resize(
                data["images"], resize_to, interpolation=TF.InterpolationMode.NEAREST
            )
        if unnormalize:
            data = UnNormalize(normalization_mean, normalization_std)(data)
        if image_tensor_to_numpy:
            data["images"] = data["images"].permute(1, 2, 0).numpy().astype(np.uint8)

        plt.subplot(w, h, i)
        plot_image = plot_sample(
            task, data=data, mode="return", label_map=label_map, **kwargs
        )
        plt.imshow(plot_image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_to)
    print("Visualization of training data saved in:", save_to)
    plt.close()
