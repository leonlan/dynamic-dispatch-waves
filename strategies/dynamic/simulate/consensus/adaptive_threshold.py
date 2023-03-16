import numpy as np

from .utils import get_dispatch_matrix


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
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )
    dispatch_count = dispatch_matrix.sum(axis=0)
    num_dispatch_per_scenario = dispatch_matrix.sum(axis=1)

    avg_num_dispatch = np.mean(num_dispatch_per_scenario, dtype=int)
    top_k_dispatch = (-dispatch_count).argsort()[:avg_num_dispatch]

    new_dispatch = old_dispatch.copy()
    new_dispatch[top_k_dispatch] = True

    assert np.all(old_dispatch <= new_dispatch)  # old action shouldn't change
    assert not new_dispatch[0]  # depot should not be dispatched

    return new_dispatch, old_postpone.copy()
