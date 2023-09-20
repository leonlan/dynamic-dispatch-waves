import numpy as np

from ddwp.Environment import StaticInfo

from .utils import (
    select_dispatch_on_threshold,
    select_postpone_on_threshold,
    verify_action,
)


def fixed_threshold(
    info: StaticInfo,
    scenarios: list[tuple[dict, list[list[int]]]],
    instance: dict,
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    dispatch_threshold: float,
    postpone_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses two thresholds to mark requests as either dispatched or postponed
    based on the dispatching frequency in scenario runs.

    Parameters
    ----------
    info
        Static information about the problem.
    scenarios
        The list of instances and solutions, one for each scenario.
    old_dispatch
        List of dispatch actions from the previous iteration.
    old_postpone
        List of postpone actions from the previous iteration.
    dispatch_threshold
        Threshold for dispatching requests.
    postpone_threshold
        Threshold for postponing requests.
    """
    if dispatch_threshold > 1:
        new_dispatch = old_dispatch.copy()  # preserve must-dispatch action
    else:
        new_dispatch = select_dispatch_on_threshold(
            scenarios, old_dispatch, old_postpone, dispatch_threshold
        )

    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
