import math


def batch_images(images, size_divisible=32):
    """
    Batch a list of images by padding all images to the largerst image in the list.
    Reference:
        https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py#L227
    Parameters
    ----------
    images : list[torch.Tensor(C, H, W)]
    size_divisible: int
        Resize images to torch.Tensor(C, size_divisible * h, size_divisible * w).
    Returns
    -------
    torch.Tensor(bs, C, max_h, max_w)
    """
    max_size = _max_by_axis([list(img.shape) for img in images])
    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    batch_shape = [len(images)] + max_size
    # pad 0 until `batch_shape` - https://pytorch.org/docs/stable/generated/torch.Tensor.new_full.html
    batched_imgs = images[0].new_full(batch_shape, fill_value=0)
    for i in range(batched_imgs.shape[0]):
        img = images[i]
        batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
