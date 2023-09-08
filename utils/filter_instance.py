import dataclasses
from typing import TypeVar

import numpy as np

from VrpInstance import VrpInstance

T = TypeVar("T", bound="VrpInstance")


def filter_instance(instance: T, mask: np.ndarray) -> T:
    """
    Filters all items of an instance using the passed-in mask.
    """
    data = {}

    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        data[field.name] = _filter_instance_value(value, mask)

    return type(instance)(**data)


def _filter_instance_value(value: np.ndarray, mask: np.ndarray):
    """
    Filters an n-dimensional value based on a provided mask.
    """
    ndim = np.ndim(value)

    if ndim == 0:
        return value
    elif ndim == 1:
        return value[mask]
    elif ndim == 2:
        shape = np.shape(value)
        if shape[0] != shape[1]:
            return value[mask]
        else:
            return value[mask][:, mask]

    raise NotImplementedError()
