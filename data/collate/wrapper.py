"""
The collate function could be modified to be able to group more types of objects and support DataLoaders made up of
custom objects. One famous use case is for passing arbitrary-sized inputs(e.g. batch of images with different sizes).
Typically tensors will be stacked in the collate function and sliced along
some dimension in the scatter function. This behavior has some limitations.
1. All tensors have to be the same size.
2. Types are limited (numpy array or Tensor).
"""
from mmcv.parallel import collate
from functools import partial


def mmcv_parallel_collate(samples_per_gpu):
    return partial(collate, samples_per_gpu=samples_per_gpu)
