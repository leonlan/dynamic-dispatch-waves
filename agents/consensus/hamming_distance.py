import numpy as np

from .utils import (
    get_dispatch_matrix,
    select_postpone_on_threshold,
    verify_action,
)


def hamming_distance(
    iteration_idx: int,
    scenarios: list[tuple[dict, list[list[int]]]],
    instance,
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    postpone_thresholds: list[float],
    **kwargs
):
    """
    Selects the solution with the smallest average Hamming distance w.r.t.
    the other solutions. The requests of this solution are marked dispatched.
    Optionally, requests are postponed using a threshold.
    """
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )
    num_scenarios = len(scenarios)
    hamming_distances = np.zeros(num_scenarios)

    for i in range(num_scenarios):
        row = dispatch_matrix[i, :]
        other_rows = np.delete(dispatch_matrix, i, axis=0)
        row_distances = np.sum(np.abs(other_rows - row), axis=1)
        hamming_distances[i] = np.sum(row_distances)

    best_idx = hamming_distances.argmin()
    new_dispatch = dispatch_matrix[best_idx].astype(bool)

    # Postpone requests using a fixed threshold, as long as the requests
    # are not yet dispatched.
    post_thresh_idx = min(iteration_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[post_thresh_idx]
    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )

    # Do not postpone requests that were dispatched (because they are part
    # of the solution with the smallest Hamming distance).
    new_postpone = new_postpone & ~new_dispatch

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
