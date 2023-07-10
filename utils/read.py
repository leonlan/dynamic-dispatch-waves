import os

import numpy as np
import vrplib


def read(path: os.PathLike | str, instance_format="vrplib"):
    """
    Reads a VRPLIB instance from file and returns an ``instance`` dict,
    containing
    - 'is_depot': boolean np.array. True for depot; False otherwise.
    - 'coords': np.array of locations
    - 'demands': np.array of location demands
    - 'capacity': int of vehicle capacity
    - 'time_windows': np.array of [l, u] time windows
    - 'service_times': np.array of service times
    - 'duration_matrix': distance matrix between locations.

    Parameters
    ----------
    path: str
        Path to the instance file.
    instance_format: str
        Format of the instance file. The default is 'vrplib'.
    """
    instance = vrplib.read_instance(path, instance_format=instance_format)
    dimension: int = instance.get("dimension", instance["demand"].size)

    # Default release time is zero
    release_times = instance.get(
        "release_time", np.zeros(dimension, dtype=int)
    )

    # Default dispatch time is planning horizon
    horizon = instance["time_window"][0][1]  # depot latest tw
    dispatch_times = instance.get(
        "dispatch_time", np.ones(dimension, dtype=int) * horizon
    )

    return {
        "is_depot": np.array([1] + [0] * (dimension - 1), dtype=bool),
        "coords": instance["node_coord"],
        "demands": instance["demand"],
        "capacity": instance["capacity"],
        "time_windows": instance["time_window"],
        "service_times": instance["service_time"],
        "duration_matrix": instance["edge_weight"].astype(int),
        "release_times": release_times,
        "dispatch_times": dispatch_times,
    }
