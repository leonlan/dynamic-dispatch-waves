import numpy as np
from pyvrp import Client, ProblemData, VehicleType

_INT_MAX = np.iinfo(np.int32).max


def instance2data(instance: dict) -> ProblemData:
    """
    Converts an instance to a ``pyvrp.ProblemData`` instance.
    """
    dimension = instance["demands"].size
    depots: np.ndarray = instance.get("depot", np.array([0]))
    num_vehicles: int = instance.get("vehicles", dimension - 1)
    capacity: int = instance.get("capacity", _INT_MAX)
    distances: np.ndarray = instance["duration_matrix"]
    demands: np.ndarray = instance["demands"]
    coords: np.ndarray = instance["coords"]
    service_times: np.ndarray = instance["service_times"]
    durations = distances
    time_windows: np.ndarray = instance["time_windows"]

    # Default release time is zero
    release_times = instance.get(
        "release_times", np.zeros(dimension, dtype=int)
    )

    # Default dispatch time is planning horizon
    horizon = instance["time_windows"][0][1]  # depot latest tw
    dispatch_times = instance.get(
        "dispatch_times", np.ones(dimension, dtype=int) * horizon
    )

    # Default prize is zero
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
