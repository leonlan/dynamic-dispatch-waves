import numpy as np

from .utils import get_dispatch_matrix


def hamming_distance(
    cycle_idx, scenarios, old_dispatch, old_postpone, **kwargs
):
    """
    Selects the solution with the smallest average Hamming distance w.r.t.
    the other solutions. The requests of this solution are marked dispatch.
    Also, all requests that are always postponed are marked as postponed.
    # TODO refactor always postpone to some utility function
    """
    dispatch_matrix = get_dispatch_matrix(
        scenarios, old_dispatch, old_postpone
    )

    # Mean absolute error a.k.a. average Hamming distance
    mae = (abs(dispatch_matrix - dispatch_matrix.mean(axis=0))).mean(axis=1)
    new_dispatch = dispatch_matrix[mae.argsort()[0]].astype(bool)

    # Postpone requests that are always postponed
    dispatch_count = dispatch_matrix.sum(axis=0)
    postpone_count = len(scenarios) - dispatch_count
    postpone_count[0] = False  # do not postpone depot
    new_postpone = postpone_count == len(scenarios)

    # TODO these assertions should become utility
    assert np.all(old_dispatch <= new_dispatch)  # old action shouldn't change
    assert np.all(old_postpone <= new_postpone)
    assert not new_dispatch[0]  # depot should not be dispatched
    assert not new_postpone[0]  # depot should not be postponed

    return new_dispatch, new_postpone
