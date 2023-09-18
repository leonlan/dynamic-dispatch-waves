from collections import Counter

import numpy as np
from pyvrp import Client, ProblemData, VehicleType

from VrpInstance import VrpInstance


def instance2data(instance: VrpInstance) -> ProblemData:
    """
    Converts an instance to a ``pyvrp.ProblemData`` instance.
    """
    # TODO Add this to PyVRP's ``Client`` constructor.
    if instance.dispatch_times[0] != instance.time_windows[0, 1]:
        raise ValueError("Depot end of time window must be dispatch time")

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

    vehicle_types = [
        VehicleType(
            instance.capacity,
            num_available,
            tw_early=tw_early,
            tw_late=instance.horizon,
        )
        for tw_early, num_available in Counter(instance.shift_tw_early).items()
    ]

    return ProblemData(
        clients,
        vehicle_types,
        instance.duration_matrix,
        instance.duration_matrix,
    )
