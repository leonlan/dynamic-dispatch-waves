import numpy as np

import hgspy
from strategies.static import nearest_neighbour
from strategies.config import Config
from .utils import get_dispatch_count


def branch_and_regret(
    cycle_idx, scenarios, old_dispatch, old_postpone, **kwargs
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

    dispatch_cost = evaluate_cost(scenarios, dispatch=(candidate,))
    postpone_cost = evaluate_cost(scenarios, postpone=(candidate,))
    print(dispatch_cost, postpone_cost)

    new_dispatch = old_dispatch.copy()
    new_postpone = old_postpone.copy()

    if dispatch_cost <= postpone_cost:
        new_dispatch[candidate] = True
        return new_dispatch, new_postpone
    else:
        new_postpone[candidate] = True
        return new_dispatch, new_postpone


def evaluate_cost(scenarios, dispatch=(), postpone=()):
    """
    Resolves the scenario solutions, while forcing the candidate to be fixed
    as dispatch or postponed. # TODO
    """
    total = 0

    for inst, sol in scenarios:
        # Change instance to force dispatch candidate
        for req in dispatch:
            inst["latest_dispatch"][req] = 0

        # HACK The next epoch time is inferred from the smallest non-zero
        # release times.
        release_times = inst["release_times"]
        next_epoch_time = np.min(release_times[np.nonzero(release_times)])

        for req in postpone:
            inst["release_times"][req] = next_epoch_time

        candidates = list(dispatch) + list(postpone)
        old = [[idx for idx in rte if idx not in candidates] for rte in sol]
        init = old + [[cand] for cand in candidates]

        nn_config_loc = "configs/nearest_neighbour.toml"
        nn_config = Config.from_file(nn_config_loc).static()
        res = nearest_neighbour(
            inst,
            config=hgspy.Config(**nn_config.solver_params()),
            node_ops=nn_config.node_ops(),
            initial_solution=init,
        )

        total += res.cost()

        # Undo changes to instance
        for req in dispatch:
            inst["latest_dispatch"][req] = inst["time_windows"][0][1]

        for req in postpone:
            inst["release_times"][req] = 0

    return int(total / len(scenarios))
