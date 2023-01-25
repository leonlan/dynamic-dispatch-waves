import numpy as np


def fixed_threshold(
    cycle_idx,
    solution_pool,
    to_dispatch,
    to_postpone,
    dispatch_thresholds,
    postpone_thresholds,
    **kwargs
):
    """
    Uses an user-specified dispatch and postpone thresholds to mark requests
    as dispatched or postponed.
    """
    # Get the threshold belonging to the current cycle, or the last one
    # available if there are more cycles than thresholds.
    threshold_idx = min(cycle_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[threshold_idx]

    threshold_idx = min(cycle_idx, len(dispatch_thresholds) - 1)
    dispatch_threshold = dispatch_thresholds[threshold_idx]

    # This asserts that we cannot have thresholds that allow a request to be
    # marked both dispatched and postponed.
    assert dispatch_threshold + (1 - postpone_threshold) < 1

    n_simulations = len(solution_pool)
    ep_size = to_dispatch.size
    dispatch_count = np.zeros(ep_size, dtype=int)

    for sol in solution_pool:
        for route in sol:
            # Count a request as dispatched if routed with `to_dispatch` reqs
            if any(to_dispatch[idx] for idx in route if idx < ep_size):
                dispatch_count[route] += 1

    to_dispatch = dispatch_count >= dispatch_threshold * n_simulations

    postpone_count = n_simulations - dispatch_count
    to_postpone = postpone_count > postpone_threshold * n_simulations

    # Never dispatch or postpone the depot
    to_dispatch[0] = False
    to_postpone[0] = False

    return to_dispatch, to_postpone
