import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime

from tools import instance2data
from tools.filter_instance import filter_instance

from .consensus import CONSENSUS
from .simulate_instance import simulate_instance


def simulate(
    env,
    info,
    obs,
    rng,
    strategy_tlim_factor: float,
    final_dispatch: str,
    n_cycles: int,
    n_simulations: int,
    n_lookahead: int,
    consensus: str,
    consensus_params: dict = {},  # noqa: B006
    **kwargs,
):
    """
    Determine the dispatch instance by simulating the next epochs and analyzing
    those simulations.
    """
    # Return the full epoch instance for the last epoch
    current_epoch = obs["current_epoch"]
    if current_epoch == info["end_epoch"]:
        return obs["epoch_instance"]

    # Parameters
    ep_inst = obs["epoch_instance"]
    ep_size = ep_inst["is_depot"].size  # includes depot
    n_cycles = n_cycles if n_cycles > 0 else ep_size  # for DSHH and BRH

    total_sim_tlim = strategy_tlim_factor * info["epoch_tlim"]
    single_sim_tlim = total_sim_tlim / (n_cycles * n_simulations)

    to_dispatch = ep_inst["must_dispatch"].copy()
    to_postpone = np.zeros(ep_size, dtype=bool)

    for cycle_idx in range(n_cycles):
        scenarios = []

        for _ in range(n_simulations):
            sim_inst = simulate_instance(
                env,
                info,
                obs,
                rng,
                n_lookahead,
                to_dispatch,
                to_postpone,
            )

            # TODO make this as a custom solver
            model = Model.from_data(instance2data(sim_inst))
            res = model.solve(MaxRuntime(single_sim_tlim), seed=42)
            sim_sol = [rte.visits() for rte in res.best.get_routes() if rte]

            scenarios.append((sim_inst, sim_sol))

        # Use the consensus function to mark requests dispatched or postponed
        consensus_func = CONSENSUS[consensus]
        to_dispatch, to_postpone = consensus_func(  # type: ignore
            cycle_idx,
            scenarios,
            to_dispatch,
            to_postpone,
            **consensus_params,
        )

        # Stop the simulation run early when all requests have been marked
        if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
            break

    # Select who to dispatch
    if final_dispatch == "to_dispatch":
        selected = to_dispatch
    elif final_dispatch == "not_to_postpone":
        selected = ~to_postpone
    else:
        raise ValueError("Final dispatch action unknown.")

    return filter_instance(ep_inst, ep_inst["is_depot"] | selected)
