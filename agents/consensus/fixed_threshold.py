import numpy as np

from .utils import (
    select_dispatch_on_threshold,
    select_postpone_on_threshold,
    verify_action,
)


def fixed_threshold(
    iteration_idx: int,
    scenarios: list[tuple[dict, list[list[int]]]],
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    dispatch_thresholds: list[float],
    postpone_thresholds: list[float],
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses two thresholds to mark requests as either dispatched or postponed
    based on the dispatching frequency in scenario runs.

    Parameters
    ----------
    iteration_idx
        The current iteration index.
    scenarios
        The list of instances and solutions, one for each scenario.
    old_dispatch
        List of dispatch actions from the previous iteration.
    old_postpone
        List of postpone actions from the previous iteration.
    dispatch_thresholds
        List of dispatch thresholds for each iteration.
    postpone_thresholds
        List of postpone thresholds for each iteration.
    """
    # Get the threshold belonging to the current iteration, or the last one
    # available if there are more iterations than thresholds.
    disp_thresh_idx = min(iteration_idx, len(dispatch_thresholds) - 1)
    dispatch_threshold = dispatch_thresholds[disp_thresh_idx]

    post_thresh_idx = min(iteration_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[post_thresh_idx]

    new_dispatch = select_dispatch_on_threshold(
        scenarios, old_dispatch, old_postpone, dispatch_threshold
    )
    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
