import numpy as np


def sample_epoch_requests(
    rng, instance, current_time, dispatch_time, max_requests_per_epoch
):
    """
    Samples requests from an epoch.
    """
    dist = instance["duration_matrix"]
    n_customers = instance["is_depot"].size - 1  # Exclude depot
    n_samples = max_requests_per_epoch

    # Sample data uniformly from customers (1 to num_customers)
    cust_idx = rng.integers(n_customers, size=n_samples) + 1
    tw_idx = rng.integers(n_customers, size=n_samples) + 1
    demand_idx = rng.integers(n_customers, size=n_samples) + 1
    service_idx = rng.integers(n_customers, size=n_samples) + 1

    new_tw = instance["time_windows"][tw_idx]
    new_demand = instance["demands"][demand_idx]
    new_service = instance["service_times"][service_idx]

    # Filter sampled requests that cannot be served in a round trip
    earliest_arrival = np.maximum(
        dispatch_time + dist[0, cust_idx], new_tw[:, 0]
    )
    earliest_return = earliest_arrival + new_service + dist[cust_idx, 0]
    depot_closed = instance["time_windows"][0, 1]

    feas = (earliest_arrival <= new_tw[:, 1]) & (
        earliest_return <= depot_closed
    )
    n_new_requests = feas.sum()
    new_release = np.full(n_new_requests, dispatch_time)

    return {
        "customer_idx": cust_idx[feas],
        "time_windows": new_tw[feas],
        "demands": new_demand[feas],
        "service_times": new_service[feas],
        "release_times": new_release,
    }
