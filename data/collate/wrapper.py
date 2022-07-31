"""
The collate function could be modified to be able to group more types of objects and support DataLoaders made up of
custom objects. One famous use case is for passing arbitrary-sized inputs(e.g. batch of images with different sizes).
Typically tensors will be stacked in the collate function and sliced along
some dimension in the scatter function. This behavior has some limitations.
1. All tensors have to be the same size.
2. Types are limited (numpy array or Tensor).
"""
# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from utils.data_container import DataContainer


def mmcv_datacontainer_collate(batch):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size. Based on the `mmcv.parallel.collate` function.

    However, don't distribute to multiple GPUs as in `mmcv.parallel.collate`
    because pytorch-lightning will automatically perform such operations. The
    `samples_per_gpu` argument is removed.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], DataContainer):
        stacked = 0
        if batch[0].cpu_only:
            stacked = [sample.data for sample in batch]
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True
            )
        elif batch[0].stack:
            assert isinstance(batch[0].data, torch.Tensor)

            if batch[0].pad_dims is not None:
                ndim = batch[0].dim()
                assert ndim > batch[0].pad_dims
                max_shape = [0 for _ in range(batch[0].pad_dims)]
                for dim in range(1, batch[0].pad_dims + 1):
                    max_shape[dim - 1] = batch[0].size(-dim)
                for sample in batch:
                    for dim in range(0, ndim - batch[0].pad_dims):
                        assert batch[0].size(dim) == sample.size(dim)
                    for dim in range(1, batch[0].pad_dims + 1):
                        max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                padded_samples = []
                for sample in batch:
                    pad = [0 for _ in range(batch[0].pad_dims * 2)]
                    for dim in range(1, batch[0].pad_dims + 1):
                        pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                    padded_samples.append(
                        F.pad(sample.data, pad, value=sample.padding_value)
                    )
                stacked = default_collate(padded_samples)
            elif batch[0].pad_dims is None:
                stacked = default_collate([sample.data for sample in batch])
            else:
                raise ValueError("pad_dims should be either None or integers (1-3)")

        else:
            stacked = [sample.data for sample in batch]
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [mmcv_datacontainer_collate(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_datacontainer_collate([d[key] for d in batch]) for key in batch[0]
        }
    else:
        return default_collate(batch)
