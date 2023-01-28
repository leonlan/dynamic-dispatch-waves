import numpy as np

from .utils import is_dispatched


def adaptive_threshold(
    cycle_idx,
    solution_pool,
    to_dispatch,
    to_postpone,
    pct_dispatch,
    **kwargs,
):
    """
    Determines how many requests to dispatch based on the average number of
    dispatched requests in the solutions pool. Let k be this number. Then the
    top-(k * pct_dispatch) requests that were most frequently dispatched in the
    simulations are marked dispatched.
    """
    ep_size = to_dispatch.size
    dispatch_count = np.zeros(ep_size, dtype=int)

    for sol in solution_pool:
        for route in sol:
            if is_dispatched(route, to_dispatch):
                dispatch_count[route] += 1

    num_disp = [num_dispatched(sol, to_dispatch) for sol in solution_pool]
    avg_num_disp = int(np.mean(num_disp) * pct_dispatch)

    top_k_dispatch = (-dispatch_count).argsort()[:avg_num_disp]
    to_dispatch[top_k_dispatch] = True

    # Never dispatch or postpone the depot
    to_dispatch[0] = False
    to_postpone[0] = False

    return to_dispatch, to_postpone


def num_dispatched(solution, to_dispatch):
    return sum([len(rt) for rt in solution if is_dispatched(rt, to_dispatch)])
