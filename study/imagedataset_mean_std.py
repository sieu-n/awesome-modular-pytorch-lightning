from tqdm import tqdm
import numpy as np


def count_dataset_stats(base_train_dataset, nc=3, scale_fn=lambda x: x, key=None):
    # https://bit.ly/3xuGOkN
    # placeholders
    psum = np.zeros(nc)
    psum_sq = np.zeros(nc)
    pixelcount = 0

    for image in tqdm(base_train_dataset):
        if key is not None:
            image = image[key]
        image = np.array(image)
        image = scale_fn(image)

        pixelcount += image.shape[0] * image.shape[1]
        psum += image.sum(axis=0).sum(axis=0)
        psum_sq += (image ** 2).sum(axis=0).sum(axis=0)

    # mean and std
    total_mean = psum / pixelcount
    total_var = (psum_sq / pixelcount) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    return total_mean, total_std
