import numpy as np
from numpy.random import Generator


def euro_neurips(
    rng: Generator,
    instance: dict,
    current_time: int,
    departure_time: int,
    epoch_duration: int,
    num_requests: int = 100,
) -> dict:
    """
    Samples requests from a VRP instance following the EURO-NeurIPS 2022
    vehicle routing competition procedure [1].

    Parameters
    ----------
    rng
        Random number generator.
    instance
        Dictionary containing the instance data.
    current_time
        Current time of the epoch.
    departure_time
        The next departure time of the vehicles.
    epoch_duration
        Duration of the epoch.
    num_requests
        Number of requests to sample.

    Returns
    -------
    dict
        Instance containing the sampled, feasible requests.

    References
    ----------
    [1] EURO meets NeurIPS 2022 vehicle routing competition.
        https://euro-neurips-vrp-2022.challenges.ortec.com/
    """
    dist = instance["duration_matrix"]
    num_customers = instance["is_depot"].size - 1

    # Sample requests attributes uniformly from customer data.
    cust_idx = rng.integers(num_customers, size=num_requests) + 1
    tw_idx = rng.integers(num_customers, size=num_requests) + 1
    demand_idx = rng.integers(num_customers, size=num_requests) + 1
    service_idx = rng.integers(num_customers, size=num_requests) + 1

    tw = instance["time_windows"][tw_idx]
    demand = instance["demands"][demand_idx]
    service = instance["service_times"][service_idx]
    release = np.full(num_requests, departure_time)

    # Exclude requests that cannot be served on time in a round trip.
    early_arrive = np.maximum(departure_time + dist[0, cust_idx], tw[:, 0])
    early_return = early_arrive + service + dist[cust_idx, 0]
    depot_closed = instance["time_windows"][0, 1]
    feas = (early_arrive <= tw[:, 1]) & (early_return <= depot_closed)

    return {
        "customer_idx": cust_idx[feas],
        "time_windows": tw[feas],
        "demands": demand[feas],
        "service_times": service[feas],
        "release_times": release[feas],
    }
