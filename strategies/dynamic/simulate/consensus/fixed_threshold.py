import numpy as np

from .utils import (
    select_dispatch_on_threshold,
    select_postpone_on_threshold,
    sanity_check,
)


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

    new_dispatch = select_dispatch_on_threshold(
        scenarios, old_dispatch, old_postpone, dispatch_threshold
    )
    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )

    sanity_check(old_dispatch, new_dispatch)
    sanity_check(old_postpone, new_postpone)

    return new_dispatch, new_postpone
