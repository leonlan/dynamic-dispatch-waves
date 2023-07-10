import numpy as np
import vrplib


def read(filename, instance_format="vrplib"):
    """
    Read a VRPLIB instance from file and return an `instance` dict, containing
    - 'is_depot': boolean np.array. True for depot; False otherwise.
    - 'coords': np.array of locations
    - 'demands': np.array of location demands
    - 'capacity': int of vehicle capacity
    - 'time_windows': np.array of [l, u] time windows
    - 'service_times': np.array of service times
    - 'duration_matrix': distance matrix between locations.
    """
    instance: dict = vrplib.read_instance(
        filename, instance_format=instance_format
    )
    n_locations: int = instance.get("dimension", instance["demand"].size)

    release_times = instance.get(
        "release_time", np.zeros(n_locations, dtype=int)
    )

    horizon = instance["time_window"][0][1]  # depot latest tw
    dispatch_times = instance.get(
        "dispatch_time", np.ones(n_locations, dtype=int) * horizon
    )

    return {
        "is_depot": np.array([1] + [0] * (n_locations - 1), dtype=bool),
        "coords": instance["node_coord"],
        "demands": instance["demand"],
        "capacity": instance["capacity"],
        "time_windows": instance["time_window"],
        "service_times": instance["service_time"],
        "duration_matrix": instance["edge_weight"].astype(int),
        "release_times": release_times,
        "dispatch_times": dispatch_times,
    }
