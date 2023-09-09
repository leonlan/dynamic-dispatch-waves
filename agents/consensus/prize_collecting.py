import numpy as np

from static_solvers import default_solver
from utils import filter_instance

from .utils import get_dispatch_count, verify_action


def prize_collecting(
    scenarios: list[tuple[dict, list[list[int]]]],
    instance,
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    fix_threshold: float,
    seed: int,
    time_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A prize-collecting consensus function. It uses thresholds to fix some
    decisions we are quite sure about, and solves a prize-collecting VRP to
    determine the dispatch/postpone decisions for all other clients.
    """
    assert 0 <= fix_threshold < 0.5

    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)
    normalized_dispatch = dispatch_count / len(scenarios)

    new_dispatch = normalized_dispatch > (1 - fix_threshold)
    new_postpone = normalized_dispatch < fix_threshold
    new_postpone[0] = False  # do not postpone depot

    not_postponed = ~new_postpone
    pc_inst = filter_instance(instance, not_postponed)

    # Prize vector. We compute this as the average arc duration scaled by the
    # dispatch percentage (more scenario dispatch == higher prize).
    # TODO perhaps only look at durations from/to granular neighbourhood?
    prizes = pc_inst.duration_matrix.mean() * normalized_dispatch
    prizes[new_dispatch] = 0  # marks these as required
    pc_inst = pc_inst.replace(prizes=prizes[not_postponed].astype(int))

    sol2ep = np.flatnonzero(not_postponed)
    res = default_solver(pc_inst, seed, time_limit)
    for route in res.best.get_routes():
        new_dispatch[sol2ep[route]] = True

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)
    return new_dispatch, new_postpone
