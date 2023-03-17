from .utils import get_dispatch_matrix, always_postponed, sanity_check


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

    # Mean absolute error a.k.a. average Hamming distance
    mae = (abs(dispatch_matrix - dispatch_matrix.mean(axis=0))).mean(axis=1)
    new_dispatch = dispatch_matrix[mae.argsort()[0]].astype(bool)
    new_postpone = always_postponed(scenarios, old_dispatch, old_postpone)

    sanity_check(old_dispatch, new_dispatch)
    sanity_check(old_postpone, new_postpone)

    return new_dispatch, new_postpone
