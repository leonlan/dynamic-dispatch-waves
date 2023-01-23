import numpy as np


def threshold(
    solutions,
    to_dispatch,
    to_postpone,
    cycle_idx,
    dispatch_thresholds,
    postpone_thresholds,
):
    # Get the threshold belonging to the current cycle_idx, or the last one
    # available if there are more epochs than thresholds.
    threshold_idx = min(cycle_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[threshold_idx]
    dispatch_threshold = dispatch_thresholds[threshold_idx]

    n_simulations = len(solutions)
    ep_size = to_dispatch.size
    dispatch_count = np.zeros(ep_size, dtype=int)

    for sol in solutions:
        for route in sol:
            # Count a request as dispatched if routed with `to_dispatch`
            if any(to_dispatch[idx] for idx in route if idx < ep_size):
                dispatch_count[route] += 1

    # Mark requests as dispatched or postponed
    to_dispatch = dispatch_count >= dispatch_threshold * n_simulations
    to_dispatch[0] = False  # Do not dispatch the depot

    postpone_count = n_simulations - dispatch_count
    to_postpone = postpone_count > postpone_threshold * n_simulations
    to_postpone[0] = False  # Do not postpone the depot

    return to_dispatch, to_postpone
