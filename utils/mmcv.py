from collections import OrderedDict
from typing import Union

import torch

from utils.data_container import DataContainer


def unpack_datacontainers(datacontainers):
    """
    Recursively unpack all `mmcv.parallel.DataContainer` objects from a dictionary.
    """
    if isinstance(datacontainers, DataContainer):
        return datacontainers.data

    if isinstance(datacontainers, dict):
        for k, v in datacontainers.items():
            if isinstance(v, DataContainer):
                datacontainers[k] = v.data
            elif isinstance(v, dict):
                datacontainers[k] = unpack_datacontainers(v)
        return datacontainers

    else:
        return datacontainers


def send_datacontainers_to_device(data, device, dont_send=[]):
    """
    Recieves dictionary, send `mmcv.parallel.DataContainer` items that has
    `cpu_only` set to False to the device. Excludes items listed in `dont_send`.
    """
    for k, v in data.items():
        if isinstance(v, DataContainer) and not v.cpu_only:
            if k not in dont_send:
                datacontainer_to_cuda(v, device)


def datacontainer_to_cuda(container, device: Union[str, torch.device]):
    """
    Send :type:`~mmcv.parallel.DataContainer` to device. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes

    Parameters
    ----------
    container: DataContainer
        data container object. `container.data` should be either a single torch.Tensor or a
        list / dictionary of tensors.
    device: Union[str, torch.device]
        device to send the data to.
    """

    assert not container.cpu_only, f"{container} is not meant to be moved to {device}"
    if container.stack:
        assert isinstance(
            container.data, torch.Tensor
        ), f"Expected `torch.Tensor` but got {type(container.data)}"
        container._data = container.data.to(device)
    else:
        if isinstance(container.data, torch.Tensor):
            container._data = container.data.to(device)
        else:
            if isinstance(container.data, list):
                it = range(len(container.data))
            elif isinstance(container.data, dict) or isinstance(
                container.data, OrderedDict
            ):
                it = container.data.keys()
            else:
                raise TypeError(f"Unidentified iterator type: {type(container.data)}")

            for idx in it:
                assert isinstance(
                    container.data[idx], torch.Tensor
                ), f"Expected `torch.Tensor` but {container.data[idx]} has \
                    type: {type(container.data[idx])}"
                container._data[idx] = container.data[idx].to(device)
