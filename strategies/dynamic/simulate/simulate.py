import numpy as np

import tools
from .consensus import CONSENSUS
from strategies.utils import filter_instance
from .simulate_instance import simulate_instance


def simulate(
    env,
    info,
    obs,
    rng,
    sim_solver,
    simulate_tlim_factor: float,
    n_cycles: int,
    n_simulations: int,
    n_lookahead: int,
    consensus: str,
    consensus_params: dict = {},
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

    total_sim_tlim = simulate_tlim_factor * info["epoch_tlim"]
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

            res = sim_solver(sim_inst, single_sim_tlim)
            sim_sol = [r for r in res.get_best_found().get_routes() if r]
            tools.validation.validate_static_solution(sim_inst, sim_sol)

            scenarios.append((sim_inst, sim_sol))

        # Use the consensus function to mark requests dispatched or postponed
        to_dispatch, to_postpone = CONSENSUS[consensus](
            cycle_idx,
            scenarios,
            to_dispatch,
            to_postpone,
            **consensus_params,
        )

        # Stop the simulation run early when all requests have been marked
        if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
            break

    return filter_instance(ep_inst, ep_inst["is_depot"] | to_dispatch)
