import numpy as np

from .utils import (
    get_dispatch_matrix,
    select_postpone_on_threshold,
    verify_action,
)


def adaptive_threshold(
    cycle_idx,
    scenarios,
    old_dispatch,
    old_postpone,
    postpone_thresholds,
    **kwargs
):
    """
    Determines how many requests to dispatch based on the average number of
    dispatched requests in the solutions pool. Let k be this number. Then the
    k most frequently dispatched requests are marked dispatched.

    Also uses a fixed postpone threshold.
    """
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )
    dispatch_count = dispatch_matrix.sum(axis=0)
    num_dispatch_per_scenario = dispatch_matrix.sum(axis=1)

    avg_num_dispatch = np.mean(num_dispatch_per_scenario).astype(int)
    top_k_dispatch = (-dispatch_count).argsort()[:avg_num_dispatch]

    new_dispatch = old_dispatch.copy()
    new_dispatch[top_k_dispatch] = True

    # Postpone requests using a fixed threshold, as long as the requests
    # are not yet dispatched.
    post_thresh_idx = min(cycle_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[post_thresh_idx]
    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )
    new_postpone = new_postpone & ~new_dispatch

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
