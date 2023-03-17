import numpy as np

from .utils import (
    get_dispatch_matrix,
    always_postponed,
    sanity_check,
)


def adaptive_threshold(
    cycle_idx, scenarios, old_dispatch, old_postpone, **kwargs
):
    """
    Determines how many requests to dispatch based on the minimum number of
    dispatched requests in the solutions pool. Let k be this number. Then the
    k most frequently dispatched requests are marked dispatched. Also, all
    requests that are always postponed are marked as postponed.
    """
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )
    dispatch_count = dispatch_matrix.sum(axis=0)
    num_dispatch_per_scenario = dispatch_matrix.sum(axis=1)

    # TODO This can also be average, but IDK which works better
    min_num_dispatch = np.min(num_dispatch_per_scenario)
    top_k_dispatch = (-dispatch_count).argsort()[:min_num_dispatch]

    new_dispatch = old_dispatch.copy()
    new_dispatch[top_k_dispatch] = True

    new_postpone = always_postponed(scenarios, old_dispatch, old_postpone)

    sanity_check(old_dispatch, new_dispatch)
    sanity_check(old_postpone, new_postpone)

    return new_dispatch, new_postpone
