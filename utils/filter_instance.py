import numpy as np

from VrpInstance import VrpInstance


def filter_instance(instance: VrpInstance, mask: np.ndarray) -> VrpInstance:
    """
    Filters all items of an instance using the passed-in mask.
    """
    new = {}

    for name, value in vars(instance).items():
        new[name.lstrip("_")] = _filter_instance_value(value, mask)

    return VrpInstance(**new)  # type: ignore


def _filter_instance_value(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
