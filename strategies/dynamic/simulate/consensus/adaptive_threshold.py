import numpy as np

from .utils import is_dispatched


def adaptive_threshold(
    cycle_idx,
    scenarios,
    to_dispatch,
    to_postpone,
    pct_dispatch=1,
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

    for (inst, sol) in scenarios:
        for route in sol:
            if is_dispatched(inst, route, to_dispatch, to_postpone):
                dispatch_count[route] += 1

    postpone_count = 1 - dispatch_count

    num_disp = [
        num_dispatched(inst, sol, to_dispatch, to_postpone)
        for (inst, sol) in scenarios
    ]
    min_num_disp = int(np.mean(num_disp) * pct_dispatch)
    min_num_post = np.min([ep_size - n_disp for n_disp in num_disp])
    print("Min dispatch: ", np.min(num_disp))
    print("Min postpone: ", min_num_post)

    top_k_dispatch = (-dispatch_count).argsort()[:min_num_disp]
    to_dispatch[top_k_dispatch] = True

    # if cycle_idx > 0:
    #     top_k_postpone = (-postpone_count).argsort()[:min_num_post]
    #     to_postpone[top_k_postpone] = True

    # Never dispatch or postpone the depot
    to_dispatch[0] = False
    to_postpone[0] = False

    return to_dispatch, to_postpone


def num_dispatched(inst, sol, to_dispatch, to_postpone):
    return sum(
        [
            len(rt)
            for rt in sol
            if is_dispatched(inst, rt, to_dispatch, to_postpone)
        ]
    )
