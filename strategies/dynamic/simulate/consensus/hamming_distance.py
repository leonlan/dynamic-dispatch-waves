import numpy as np

from .utils import (
    get_dispatch_matrix,
    select_postpone_on_threshold,
    verify_action,
)


def hamming_distance(
    cycle_idx,
    scenarios,
    old_dispatch,
    old_postpone,
    postpone_thresholds,
    **kwargs
):
    """
    Selects the solution with the smallest average Hamming distance w.r.t.
    the other solutions. The requests of this solution are marked dispatch.
    Also, all requests that are always postponed are marked as postponed.
    """
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )

    n, _ = dispatch_matrix.shape
    hamming_distances = np.zeros(n)

    for i in range(n):
        row = dispatch_matrix[i, :]
        other_rows = np.delete(dispatch_matrix, i, axis=0)
        row_distances = np.sum(np.abs(other_rows - row), axis=1)
        hamming_distances[i] = np.sum(row_distances)

    sol_idx = hamming_distances.argmin()
    new_dispatch = dispatch_matrix[sol_idx].astype(bool)

    # Postpone requests using a fixed threshold, as long as the requests
    # are not yet dispatched.
    post_thresh_idx = min(cycle_idx, len(postpone_thresholds) - 1)
    postpone_threshold = postpone_thresholds[post_thresh_idx]
    new_postpone = select_postpone_on_threshold(
        scenarios, old_dispatch, old_postpone, postpone_threshold
    )
    new_postpone = new_postpone & ~new_dispatch

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
