import numpy as np
from numpy.random import Generator


def sample_epoch_requests(
    rng: Generator,
    instance: dict,
    current_time: float,
    departure_time: float,
    num_requests: int = 100,
):
    """
    Samples requests from a VRP instance.

    Only requests that can be served in a round trip that starts at
    `departure_time` are returned.

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
    num_requests
        Number of requests to sample. Defaults to 100.
    """
    dist = instance["duration_matrix"]
    n_customers = instance["is_depot"].size - 1  # Exclude depot
    num_samples = num_requests

    # Sample requests attributes uniformly from customer data
    cust_idx = rng.integers(n_customers, size=num_samples) + 1
    tw_idx = rng.integers(n_customers, size=num_samples) + 1
    demand_idx = rng.integers(n_customers, size=num_samples) + 1
    service_idx = rng.integers(n_customers, size=num_samples) + 1

    new_tw = instance["time_windows"][tw_idx]
    new_demand = instance["demands"][demand_idx]
    new_service = instance["service_times"][service_idx]

    # Filter all sampled requests that cannot be served in a round trip that
    # starts at `departure_time`.
    earliest_arrival = np.maximum(
        departure_time + dist[0, cust_idx], new_tw[:, 0]
    )
    earliest_return = earliest_arrival + new_service + dist[cust_idx, 0]
    depot_closed = instance["time_windows"][0, 1]

    feas = (earliest_arrival <= new_tw[:, 1]) & (
        earliest_return <= depot_closed
    )
    num_new_requests = feas.sum()
    new_release = np.full(num_new_requests, departure_time)

    return {
        "customer_idx": cust_idx[feas],
        "time_windows": new_tw[feas],
        "demands": new_demand[feas],
        "service_times": new_service[feas],
        "release_times": new_release,
    }
