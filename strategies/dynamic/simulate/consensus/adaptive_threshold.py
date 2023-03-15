import numpy as np

from .utils import get_counts, get_actions


def adaptive_threshold(
    cycle_idx,
    scenarios,
    old_dispatch,
    old_postpone,
    **kwargs,
):
    """
    Determines how many requests to dispatch based on the average number of
    dispatched requests in the solutions pool. Let k be this number. Then the
    k most frequently dispatched requests are marked dispatched.
    """
    dispatch_count, _ = get_counts(scenarios, old_dispatch, old_postpone)
    dispatch_matrix = get_actions(scenarios, old_dispatch, old_postpone)
    num_dispatch_per_scenario = dispatch_matrix.sum(axis=1)

    avg_num_dispatch = np.mean(num_dispatch_per_scenario, dtype=int)
    top_k_dispatch = (-dispatch_count).argsort()[:avg_num_dispatch]

    new_dispatch = old_dispatch.copy()
    new_dispatch[top_k_dispatch] = True

    # Verify that the previously fixed dispatch actions have not changed
    assert np.all(old_dispatch <= new_dispatch)

    return new_dispatch, old_postpone.copy()
