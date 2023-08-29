import numpy as np
from numpy.random import Generator


def custom_time_windows(
    rng: Generator,
    instance: dict,
    current_time: int,
    departure_time: int,
    epoch_duration: int,
    num_requests: int = 100,
    tw_type: str = "deadlines",
    tw_width: int = 1,
    noise_lb: float = 0.9,
    noise_ub: float = 1.1,
) -> dict:
    """
    Samples requests from a VRP instance with custom time windows, following
    the procedure in [1]. Keeps sampling until the number of feasible requests
    is equal to the passed number of requests.

    Parameters
    ----------
    rng
        Random number generator.
    instance
        Dictionary containing the instance data.
    current_time
        Current time of the epoch.
    departure_time
        Departure time of the vehicles.
    epoch_duration
        Duration of the epoch.
    num_requests
        Number of requests to sample.
    tw_type
        Type of the time windows: one of ["deadlines", "time_windows"].
    tw_width
        Width of the time windows in number of epoch durations.
    noise_lb
        Lower bound of the noise factor.
    noise_ub
        Upper bound of the noise factor.

    Returns
    -------
    dict
        The sampled requests.

    References
    ----------
    [1] Lan, L., van Doorn, J., Wouda, N. A., Rijal, A., & Bhulai, S. (2023).
        An iterative conditional dispatch algorithm for the dynamic dispatch
        waves problem.
    """
    dist = instance["duration_matrix"]
    num_customers = instance["is_depot"].size - 1

    noise = rng.uniform(noise_lb, noise_ub)
    num_samples = int(noise * num_requests)

    feas = np.zeros(num_samples, dtype=bool)
    cust_idx = np.empty(num_samples, dtype=int)
    demand_idx = np.empty(num_samples, dtype=int)
    service_idx = np.empty(num_samples, dtype=int)
    old_tw = np.empty(shape=(0, 2), dtype=int)

    while not feas.all():
        num_to_sample = np.sum(~feas)

        new_cust_idx = rng.integers(num_customers, size=num_to_sample) + 1
        cust_idx = np.append(cust_idx[feas], new_cust_idx)

        new_demand_idx = rng.integers(num_customers, size=num_to_sample) + 1
        demand_idx = np.append(demand_idx[feas], new_demand_idx)
        demand = instance["demands"][demand_idx]

        new_service_idx = rng.integers(num_customers, size=num_to_sample) + 1
        service_idx = np.append(service_idx[feas], new_service_idx)
        service = instance["service_times"][service_idx]

        new_tw = _sample_time_windows(
            rng,
            instance,
            num_to_sample,
            tw_type,
            tw_width,
            current_time,
            epoch_duration,
        )
        tw = np.concatenate((old_tw, new_tw))

        # Exclude requests that cannot be served on time in a round trip.
        early_arrive = np.maximum(departure_time + dist[0, cust_idx], tw[:, 0])
        early_return = early_arrive + service + dist[cust_idx, 0]
        depot_closed = instance["time_windows"][0, 1]

        feas = (early_arrive <= tw[:, 1]) & (early_return <= depot_closed)
        old_tw = tw[feas]

    return {
        "customer_idx": cust_idx,
        "time_windows": tw,
        "demands": demand,
        "service_times": service,
        "release_times": np.full(num_samples, departure_time),
    }


def _sample_time_windows(
    rng: Generator,
    instance: dict,
    num_samples: int,
    tw_type: str,
    tw_width: int,
    current_time: int,
    epoch_duration: int,
):
    """
    Samples time windows for the requests.

    Parameters
    ----------
    rng
        Random number generator.
    instance
        Dictionary containing the instance data.
    num_samples
        Number of samples to generate.
    tw_type
        Type of the time windows: one of ["deadlines", "time_windows"].
    tw_width
        Width of the time windows in number of epoch durations.
    current_time
        Current time of the epoch.
    epoch_duration
        Duration of the epoch.

    Returns
    -------
    np.ndarray
        Time windows of the sampled requests.
    """
    horizon = instance["time_windows"][0][1]
    widths = epoch_duration * (rng.integers(tw_width, size=num_samples) + 1)

    if tw_type == "deadlines":
        early = current_time * np.ones(num_samples, dtype=int)
        late = np.minimum(horizon, early + widths)
    elif tw_type == "time_windows":
        early = rng.integers(current_time, horizon, num_samples)
        late = np.minimum(horizon, early + widths)
    else:
        raise ValueError("Time window type unknown.")

    return np.vstack((early, late)).T
