import numpy as np

# Fixed value given in the competition rules.
# TODO Read this from the environment or static info.
_EPOCH_DURATION = 3600


def simulate_instance(
    info,
    obs,
    rng,
    n_lookahead: int,
    n_requests: int,
    to_dispatch=None,
    to_postpone=None,
):
    """
    Simulate a VRPTW instance with n_lookahead epochs.
    - Sample ``EPOCH_N_REQUESTS`` requests per future epoch
    - Filter the customers that cannot be served in a round trip
    - Concatenate the epoch instance and the simulated requests

    Params
    ------
    - `to_dispatch` is a boolean array where True means that the corresponding
    request must be dispatched.
    - `to_postpone` is a boolean array where True mean that the corresponding
    request must be postponed.
    """
    # Parameters
    static_inst = info["dynamic_context"]
    ep_inst = obs["epoch_instance"]
    dispatch_time = obs["dispatch_time"]
    dist = static_inst["duration_matrix"]
    tws = static_inst["time_windows"]

    epochs_left = info["end_epoch"] - obs["current_epoch"]
    max_lookahead = min(n_lookahead, epochs_left)
    n_samples = max_lookahead * n_requests

    n_customers = static_inst["is_depot"].size - 1  # Exclude depot

    feas = np.zeros(n_samples, dtype=bool)

    cust_idx = np.empty(n_samples, dtype=int)
    tw_idx = np.empty(n_samples, dtype=int)
    service_idx = np.empty(n_samples, dtype=int)

    while not feas.all():
        to_sample = np.sum(~feas)

        cust_idx = np.append(cust_idx[feas], rng.integers(n_customers, size=to_sample) + 1)
        tw_idx = np.append(tw_idx[feas], rng.integers(n_customers, size=to_sample) + 1)
        service_idx = np.append(service_idx[feas], rng.integers(n_customers, size=to_sample) + 1)

        # These are static time windows and release times, which are used to
        # determine request feasibility. Will be clipped later to fit the epoch.
        sim_tw = tws[tw_idx]
        sim_epochs = np.repeat(np.arange(1, max_lookahead + 1), n_requests)
        sim_release = dispatch_time + sim_epochs * _EPOCH_DURATION
        sim_service = static_inst["service_times"][service_idx]

        # Earliest arrival is release time + drive time or earliest time window.
        earliest_arrival = np.maximum(
            sim_release + dist[0, cust_idx], sim_tw[:, 0]
        )
        earliest_return = earliest_arrival + sim_service + dist[cust_idx, 0]
        feas = (earliest_arrival <= sim_tw[:, 1]) & (earliest_return <= tws[0, 1])

    # Concatenate the new feasible requests to the epoch instance
    req_customer_idx = np.concatenate((ep_inst["customer_idx"], cust_idx))

    # Simulated request indices are always negative (so we can identify them)
    sim_req_idx = -(np.arange(n_samples) + 1)
    req_idx = np.concatenate((ep_inst["request_idx"], sim_req_idx))

    # Normalize TW and release to start_time, and clip the past
    sim_tw = np.maximum(sim_tw - dispatch_time, 0)
    req_tw = np.concatenate((ep_inst["time_windows"], sim_tw))

    ep_release = (
        to_postpone * _EPOCH_DURATION
        if to_postpone is not None
        else np.zeros_like(ep_inst["is_depot"])
    )
    sim_release = np.maximum(sim_release - dispatch_time, 0)
    req_release = np.concatenate((ep_release, sim_release))

    # Default latest dispatch is the time horizon. For requests that are
    # marked as dispatched, the latest dispatch time becomes zero.
    horizon = req_tw[0][1]
    req_latest_dispatch = np.ones(req_customer_idx.size, dtype=int) * horizon

    if to_dispatch is not None:
        req_latest_dispatch[to_dispatch.nonzero()] = 0

    demand_idx = rng.integers(n_customers, size=n_samples) + 1
    sim_demand = static_inst["demands"][demand_idx]

    req_demand = np.concatenate((ep_inst["demands"], sim_demand))
    req_service = np.concatenate((ep_inst["service_times"], sim_service))

    return {
        "is_depot": static_inst["is_depot"][req_customer_idx],
        "customer_idx": req_customer_idx,
        "request_idx": req_idx,
        "coords": static_inst["coords"][req_customer_idx],
        "demands": req_demand,
        "capacity": static_inst["capacity"],
        "time_windows": req_tw,
        "service_times": req_service,
        "duration_matrix": dist[req_customer_idx][:, req_customer_idx],
        "release_times": req_release,
        "latest_dispatch": req_latest_dispatch,
    }
