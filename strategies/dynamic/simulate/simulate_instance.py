import numpy as np


def simulate_instance(
    env,
    info,
    obs,
    rng,
    n_lookahead: int,
    to_dispatch: np.ndarray,
    to_postpone: np.ndarray,
):
    """
    Simulates a VRPTW scenario instance with ``n_lookahead`` epochs. It uses the
    ``Environment.sample_epoch_requests`` method to sample new requests from
    future epochs. The scenario instance is created by appending the sampled
    requests to the current epoch instance.

    Params
    ------
    to_dispatch
        A boolean array where True means that the corresponding request must be
        dispatched.
    to_postpone
        A boolean array where True mean that the corresponding request must be
        postponed.
    """
    current_epoch = obs["current_epoch"]
    next_epoch = current_epoch + 1
    epochs_left = info["end_epoch"] - current_epoch
    max_lookahead = min(n_lookahead, epochs_left)

    # Parameters
    static_inst = info["dynamic_context"]
    ep_inst = obs["epoch_instance"]
    dispatch_time = obs["dispatch_time"]
    dist = static_inst["duration_matrix"]

    # Simulation instance
    req_customer_idx = ep_inst["customer_idx"]
    req_idx = ep_inst["request_idx"]
    req_demand = ep_inst["demands"]
    req_service = ep_inst["service_times"]
    req_tw = ep_inst["time_windows"]

    # Conditional dispatching
    horizon = req_tw[0][1]
    req_release = to_postpone * env.epoch_duration
    req_dispatch_times = np.where(to_dispatch, 0, horizon)

    for epoch_idx in range(next_epoch, next_epoch + max_lookahead):
        new = env.sample_epoch_requests(epoch_idx, rng)
        n_new_reqs = new["customer_idx"].size

        # Concatenate the new feasible requests to the epoch instance
        req_customer_idx = np.concatenate(
            (req_customer_idx, new["customer_idx"])
        )

        # Simulated request indices are always negative (so we can identify them)
        sim_req_idx = -(np.arange(n_new_reqs) + 1) - len(req_idx)
        req_idx = np.concatenate((ep_inst["request_idx"], sim_req_idx))

        req_demand = np.concatenate((req_demand, new["demands"]))
        req_service = np.concatenate((req_service, new["service_times"]))

        # Normalize TW and release to start_time, and clip the past
        new["time_windows"] = np.maximum(
            new["time_windows"] - dispatch_time, 0
        )
        req_tw = np.concatenate((req_tw, new["time_windows"]))

        new["release_times"] = np.maximum(
            new["release_times"] - dispatch_time, 0
        )
        req_release = np.concatenate((req_release, new["release_times"]))

        # Default latest dispatch is the time horizon.
        req_dispatch_times = np.concatenate(
            (req_dispatch_times, np.full(n_new_reqs, horizon))
        )

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
        "dispatch_times": req_dispatch_times,
    }
