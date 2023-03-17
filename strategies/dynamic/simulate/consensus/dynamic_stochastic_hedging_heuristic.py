import numpy as np

from .utils import get_dispatch_count


def dynamic_stochastic_hedging_heuristic(
    cycle_idx,
    scenarios,
    old_dispatch,
    old_postpone,
    min_dispatch_threshold,
    dispatch_threshold,
    **kwargs
):
    """
    The Dynamic Stochastic Hedging Heuristic [1] dispatches all requests
    with dispatch probability above ``dispatch_threshold``. Otherwise, it
    selects the most frequently dispatched request that is at least dispatched
    with probability ``min_dispatch_threshold``.

    [1] Hvattum, L. M., Løkketangen, A., & Laporte, G. (2006). Solving a
        Dynamic and Stochastic Vehicle Routing Problem with a Sample
        Scenario Hedging Heuristic. Transportation Science, 40(4), 421–438.
        https://doi.org/10.1287/trsc.1060.0166
    """
    n_simulations = len(scenarios)
    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)

    new_dispatch = dispatch_count >= dispatch_threshold * n_simulations

    if new_dispatch.sum() == old_dispatch.sum():
        # None of the requests have a dispatch probability of at least
        # ``dispatch_threshold``, so we select the most frequently dispatched
        # request if it has probability higher than ``min_dispatch_threshold``.
        candidate = np.where(old_dispatch == 0, dispatch_count, 0).argmax()

        if dispatch_count[candidate] >= min_dispatch_threshold * n_simulations:
            new_dispatch[candidate] = True

    assert np.all(old_dispatch <= new_dispatch)  # old action shouldn't change
    assert not new_dispatch[0]  # depot should not be dispatched

    return new_dispatch, old_postpone.copy()
