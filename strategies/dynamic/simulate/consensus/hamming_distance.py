import numpy as np
from .utils import get_dispatch_matrix, always_postponed, verify_action


def hamming_distance(
    cycle_idx, scenarios, old_dispatch, old_postpone, **kwargs
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

    new_postpone = always_postponed(scenarios, old_dispatch, old_postpone)

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)

    return new_dispatch, new_postpone
