import numpy as np

from .utils import get_dispatch_count


def fixed_threshold(
    cycle_idx,
    scenarios,
    old_dispatch,
    old_postpone,
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
    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)
    postpone_count = n_simulations - dispatch_count
    postpone_count[0] = 0  # do not postpone depot

    new_dispatch = dispatch_count >= dispatch_threshold * n_simulations
    new_postpone = postpone_count > postpone_threshold * n_simulations

    assert np.all(old_dispatch <= new_dispatch)  # old action shouldn't change
    assert np.all(old_postpone <= new_postpone)
    assert not new_dispatch[0]  # depot should not be dispatched
    assert not new_postpone[0]  # depot should not be postponed

    return new_dispatch, new_postpone
