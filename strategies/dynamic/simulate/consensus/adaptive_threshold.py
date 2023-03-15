import numpy as np

from .utils import is_dispatched


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
    ep_size = old_dispatch.size
    new_dispatch = old_dispatch.copy()

    dispatch_count = np.zeros(ep_size, dtype=int)
    num_dispatch_scenario = []

    for (inst, sol) in scenarios:
        num_disp = 0

        for route in sol:
            if is_dispatched(inst, route, old_dispatch, old_postpone):
                dispatch_count[route] += 1
                num_disp += len(route)

        num_dispatch_scenario.append(num_disp)

    avg_num_dispatch = int(np.mean(num_dispatch_scenario))
    top_k_dispatch = (-dispatch_count).argsort()[:avg_num_dispatch]
    new_dispatch[top_k_dispatch] = True

    # Verify that the previously fixed dispatch actions have not changed
    assert np.all(old_dispatch <= new_dispatch)

    return new_dispatch, old_postpone.copy()
