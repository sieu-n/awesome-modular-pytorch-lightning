import matplotlib.pyplot as plt

from data.transforms.vision import UnNormalize


def PlotSamples(dataset, subplot_dim=(5, 5), save_to="results/samples_vis.png", normalization_mean=(0.5, 0.5, 0.5),
                normalization_std=(0.5, 0.5, 0.5), imsize=3, preprocess_f=None, is_image_only=False):
    """
    Plot multiple samples using matplotlib.
    TODO: write docstring.
    Parameters
    """
    un_norm_f = UnNormalize(normalization_mean, normalization_std)

    w, h = subplot_dim
    plt.figure(figsize=(w * imsize, h * imsize))
    for i in range(1, w * h + 1):
        image = dataset[i - 1]
        if not is_image_only:
            image = image[0]

        if preprocess_f:
            image = preprocess_f(image)
        else:
            image = un_norm_f(image)

        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.savefig(save_to)
    plt.show()
