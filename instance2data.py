from numbers import Number

import numpy as np
from pyvrp import Client, ProblemData, VehicleType

_INT_MAX = np.iinfo(np.int32).max


def instance2data(instance) -> ProblemData:
    """
    Converts an instance to a pyvrp model.
    """
    # A priori checks
    if "dimension" in instance:
        dimension: int = instance["dimension"]
    else:
        if "demands" not in instance:
            raise ValueError("File should either contain dimension or demands")
        dimension = len(instance["demands"])

    depots: np.ndarray = instance.get("depot", np.array([0]))
    num_vehicles: int = instance.get("vehicles", dimension - 1)
    capacity: int = instance.get("capacity", _INT_MAX)

    distances: np.ndarray = instance["duration_matrix"]

    demands: np.ndarray = instance["demands"]
    coords: np.ndarray = instance["coords"]
    durations = distances
    time_windows: np.ndarray = instance["time_windows"]

    if "service_times" in instance:
        if isinstance(instance["service_times"], Number):
            # Some instances describe a uniform service time as a single value
            # that applies to all clients.
            service_times = np.full(dimension, instance["service_times"], int)
            service_times[0] = 0
        else:
            service_times = instance["service_times"]
    else:
        service_times = np.zeros(dimension, dtype=int)

    if "release_times" in instance:
        release_times: np.ndarray = instance["release_times"]
    else:
        release_times = np.zeros(dimension, dtype=int)

    if "dispatch_times" in instance:
        dispatch_times: np.ndarray = instance["dispatch_times"]
    else:
        horizon = time_windows.max()
        dispatch_times = horizon * np.ones(dimension, dtype=int)

    prizes = instance.get("prizes", np.zeros(dimension, dtype=int))

    # Checks
    if len(depots) != 1 or depots[0] != 0:
        raise ValueError(
            "Source file should contain single depot with index 1 "
            + "(depot index should be 0 after subtracting offset 1)"
        )

    if demands[0] != 0:
        raise ValueError("Demand of depot must be 0")

    if time_windows[0, 0] != 0:
        raise ValueError("Depot start of time window must be 0")

    if service_times[0] != 0:
        raise ValueError("Depot service duration must be 0")

    if release_times[0] != 0:
        raise ValueError("Depot release time must be 0")

    if dispatch_times[0] != time_windows[0, 1]:
        raise ValueError("Depot end of time window must be dispatch time")

    if (time_windows[:, 0] > time_windows[:, 1]).any():
        raise ValueError("Time window cannot start after end")

    clients = [
        Client(
            coords[idx][0],  # x
            coords[idx][1],  # y
            demands[idx],
            service_times[idx],
            time_windows[idx][0],  # TW early
            time_windows[idx][1],  # TW late
            release_times[idx],
            dispatch_times[idx],
            prizes[idx],
            np.isclose(prizes[idx], 0),  # required only when prize is zero
        )
        for idx in range(dimension)
    ]
    vehicle_types = [VehicleType(capacity, num_vehicles)]

    return ProblemData(
        clients,
        vehicle_types,
        distances,
        durations,
    )
