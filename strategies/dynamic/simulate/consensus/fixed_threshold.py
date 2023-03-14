import numpy as np

from .utils import is_dispatched


def fixed_threshold(
    cycle_idx,
    scenarios,
    to_dispatch,
    to_postpone,
    dispatch_thresholds,
    postpone_thresholds,
    **kwargs
):
    """
    Uses user-specified dispatch and postpone thresholds to mark requests
    as dispatched or postponed.
    """
    # Get the threshold belonging to the current cycle, or the last one
    # available if there are more cycles than thresholds.
    disp_thresh_idx = min(cycle_idx, len(dispatch_thresholds) - 1)
    dispatch_threshold = dispatch_thresholds[disp_thresh_idx]

    post_thresh_idx = min(cycle_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[post_thresh_idx]

    n_simulations = len(scenarios)
    ep_size = to_dispatch.size

    dispatch_count = np.zeros(ep_size, dtype=int)
    postpone_count = np.zeros(ep_size, dtype=int)

    for (inst, sol) in scenarios:
        for route in sol:
            if is_dispatched(inst, route, to_dispatch, to_postpone):
                dispatch_count[route] += 1
            else:
                # Only count for current epoch requests
                reqs = [idx for idx in route if idx < ep_size]
                postpone_count[reqs] += 1

    assert np.all(dispatch_count + postpone_count <= n_simulations)

    new_dispatch = dispatch_count >= dispatch_threshold * n_simulations
    new_postpone = postpone_count > postpone_threshold * n_simulations

    return new_dispatch, new_postpone
