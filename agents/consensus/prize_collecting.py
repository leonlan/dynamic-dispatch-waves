import numpy as np

from static_solvers import default_solver

from .utils import (
    select_dispatch_on_threshold,
    select_postpone_on_threshold,
    get_dispatch_count,
    verify_action,
)


def prize_collecting(
    iteration_idx: int,
    scenarios: list[tuple[dict, list[list[int]]]],
    instance,
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    fix_threshold: float,
    lamda: float,
    seed: int,
    time_limit: float,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    A prize-collecting consensus function. It uses thresholds to fix some
    decisions we are quite sure about, and solves a prize-collecting VRP to
    determine the dispatch/postpone decisions for all other clients.
    """
    assert 0 <= fix_threshold <= 1
    assert lamda >= 0

    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)
    new_dispatch = dispatch_count >= fix_threshold * len(scenarios)

    postpone_count = len(scenarios) - dispatch_count
    postpone_count[0] = 0  # do not postpone depot
    new_postpone = postpone_count >= (1 - fix_threshold) * len(scenarios)

    # TODO

    pc_inst = {
        "is_depot": instance["is_depot"][req_customer_idx],
        "customer_idx": req_customer_idx,
        "request_idx": req_idx,
        "coords": static_inst["coords"][req_customer_idx],
        "demands": req_demand,
        "capacity": static_inst["capacity"],
        "time_windows": req_tw,
        "service_times": req_service,
        "duration_matrix": dist[req_customer_idx][:, req_customer_idx],
        "release_times": req_release,
        "dispatch_times": req_dispatch,
        "prizes": [],
    }

    if instance["request_idx"].size <= 2:
        # BUG Empty or single client dispatch instance, PyVRP cannot handle
        # this (see https://github.com/PyVRP/PyVRP/issues/272).
        # We don't know for sure if we want to dispatch these, but it's just
        # one client so it cannot be all that bad either way.
        new_dispatch[instance["request_idx"]] = True
    else:
        res = default_solver(instance, seed, time_limit)
        for route in res.best.get_routes():
            new_dispatch[instance["request_idx"][route]] = True

    verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone)
    return new_dispatch, new_postpone
