import numpy as np

from .utils import get_dispatch_count


def branch_and_regret(
    cycle_idx, scenarios, old_dispatch, old_postpone, cost_eval_tlim, **kwargs
):
    """
    The Branch-and-Regret Heuristic [2] selects in each iteration the most
    frequently dispatched request. It then evaluates the scenario solutions
    again, while restricting the selected request to be dispatched and to be
    postponed (separately). The action that leads to the lowest average cost
    is then selected.
    """
    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)

    undecided = (old_dispatch == 0) & (old_postpone == 0)
    candidate = np.where(undecided, dispatch_count, 0).argmax()

    if candidate == 0:  # stop when the best candidate is the depot
        return old_dispatch, old_postpone

    dispatch_cost = evaluate_cost(
        scenarios,
        sim_solver=kwargs["sim_solver"],
        time_limit=cost_eval_tlim,
        dispatch=(candidate,),
    )
    postpone_cost = evaluate_cost(
        scenarios,
        sim_solver=kwargs["sim_solver"],
        time_limit=cost_eval_tlim,
        postpone=(candidate,),
    )
    print(dispatch_cost, postpone_cost)

    new_dispatch = old_dispatch.copy()
    new_postpone = old_postpone.copy()

    if dispatch_cost <= postpone_cost:
        new_dispatch[candidate] = True
        return new_dispatch, new_postpone
    else:
        new_postpone[candidate] = True
        return new_dispatch, new_postpone


def evaluate_cost(scenarios, sim_solver, time_limit, dispatch=(), postpone=()):
    """
    Resolves the scenario solutions, while forcing the candidate to be fixed
    as dispatch or postponed. # TODO
    """
    total = 0
    time_limit_per_instance = time_limit / len(scenarios)

    for inst, sol in scenarios:
        # Change instance to force dispatch candidate
        for req in dispatch:
            inst["latest_dispatch"][req] = 0

        for req in postpone:
            # TODO make this environment dependent?
            inst["release_times"][req] = 3600

        candidates = list(dispatch) + list(postpone)
        init = [
            [idx for idx in route if idx not in candidates] for route in sol
        ] + [[cand] for cand in candidates]

        res = sim_solver(
            inst, time_limit_per_instance, initial_solutions=[init]
        )
        total += res.get_best_found().cost()

        # Undo changes to instance
        for req in dispatch:
            inst["latest_dispatch"][req] = inst["time_windows"][0][1]

        for req in postpone:
            inst["release_times"][req] = 0

    return int(total / len(scenarios))
