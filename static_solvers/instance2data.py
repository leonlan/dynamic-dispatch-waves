import numpy as np
from pyvrp import Client, ProblemData, VehicleType

from VrpInstance import VrpInstance


def instance2data(instance: VrpInstance) -> ProblemData:
    """
    Converts an instance to a ``pyvrp.ProblemData`` instance.
    """
    # Check if the data is valid.
    if not instance.is_depot[0] and instance.is_depot.sum() == 1:
        raise ValueError(
            "First request in the instance should be the depot,",
            "only a single depot is allowed.",
        )

    if instance.demands[0] != 0:
        raise ValueError("Demand of depot must be 0")

    if instance.service_times[0] != 0:
        raise ValueError("Depot service duration must be 0")

    if instance.release_times[0] != 0:
        raise ValueError("Depot release time must be 0")

    if instance.dispatch_times[0] != instance.time_windows[0, 1]:
        raise ValueError("Depot end of time window must be dispatch time")

    if (instance.time_windows[:, 0] > instance.time_windows[:, 1]).any():
        raise ValueError("Time window cannot start after end")

    clients = [
        Client(
            instance.coords[idx][0],  # x
            instance.coords[idx][1],  # y
            instance.demands[idx],
            instance.service_times[idx],
            instance.time_windows[idx][0],  # TW early
            instance.time_windows[idx][1],  # TW late
            instance.release_times[idx],
            instance.dispatch_times[idx],
            instance.prizes[idx],
            np.isclose(instance.prizes[idx], 0),  # required when prize is zero
        )
        for idx in range(instance.dimension)
    ]

    # TODO make heterogeneous
    vehicle_types = [VehicleType(instance.capacity, instance.dimension - 1)]

    return ProblemData(
        clients,
        vehicle_types,
        instance.duration_matrix,
        instance.duration_matrix,
    )
