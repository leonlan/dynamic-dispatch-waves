import numpy as np


def filter_instance(instance: dict, mask: np.ndarray) -> dict:
    """
    Filters all items of an instance using the passed-in mask.
    """
    return {
        key: _filter_instance_value(value, mask)
        for key, value in instance.items()
    }


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