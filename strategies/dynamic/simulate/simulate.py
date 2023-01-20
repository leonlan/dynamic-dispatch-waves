import numpy as np

import hgspy
from strategies.static import hgs
from strategies.utils import filter_instance
from .simulate_instance import simulate_instance
import tools


def simulate(
    info,
    obs,
    rng,
    simulate_tlim_factor: float,
    n_cycles: int,
    n_simulations: int,
    n_lookahead: int,
    n_requests: int,
    postpone_thresholds: list,
    dispatch_thresholds: list,
    sim_config: dict,
    node_ops: list,
    route_ops: list,
    crossover_ops: list,
    **kwargs,
):
    """
    Determine the dispatch instance by simulating the next epochs and analyzing
    those simulations.
    """
    # Return the full epoch instance for the last epoch
    if obs["current_epoch"] == info["end_epoch"]:
        return obs["epoch_instance"]

    # Parameters
    ep_inst = obs["epoch_instance"]
    ep_size = ep_inst["is_depot"].size  # includes depot
    total_sim_tlim = simulate_tlim_factor * info["epoch_tlim"]
    single_sim_tlim = total_sim_tlim / (n_cycles * n_simulations)

    dispatch_count = np.zeros(ep_size, dtype=int)
    to_dispatch = ep_inst["must_dispatch"]
    to_postpone = np.zeros(ep_size, dtype=bool)

    # Get the threshold belonging to the current epoch, or the last one
    # available if there are more epochs than thresholds.
    epoch = obs["current_epoch"] - info["start_epoch"]
    num_thresholds = len(postpone_thresholds)
    postpone_threshold = postpone_thresholds[min(epoch, num_thresholds - 1)]
    dispatch_threshold = dispatch_thresholds[min(epoch, num_thresholds - 1)]

    for _ in range(n_cycles):
        for _ in range(n_simulations):
            sim_inst = simulate_instance(
                info,
                obs,
                rng,
                n_lookahead,
                n_requests,
                to_dispatch=to_dispatch,
                to_postpone=to_postpone,
            )

            res = hgs(
                sim_inst,
                hgspy.Config(**sim_config),
                [getattr(hgspy.operators, op) for op in node_ops],
                [getattr(hgspy.operators, op) for op in route_ops],
                [getattr(hgspy.crossover, op) for op in crossover_ops],
                hgspy.stop.MaxRuntime(single_sim_tlim),
            )

            best = res.get_best_found()

            # TODO This can be removed at some point
            tools.validate_static_solution(
                sim_inst, [x for x in best.get_routes() if x]
            )

            for sim_route in best.get_routes():
                # Count a request as dispatched if routed with `to_dispatch`
                if any(to_dispatch[idx] for idx in sim_route if idx < ep_size):
                    dispatch_count[sim_route] += 1

        # Mark requests as dispatched or postponed
        to_dispatch = dispatch_count >= dispatch_threshold * n_simulations
        to_dispatch[0] = False  # Do not dispatch the depot

        postpone_count = n_simulations - dispatch_count
        to_postpone = postpone_count >= postpone_threshold * n_simulations
        to_postpone[0] = False  # Do not postpone the depot

        # Stop the simulation run early when all requests have been marked
        if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
            break

        dispatch_count *= 0  # reset dispatch count

    # TODO This should become a parameter: we can also dispatch only those
    # requests that have been marked `to_dispatch`.
    # Dispatch all requests that are not marked `to_postpone`
    to_dispatch = ep_inst["is_depot"] | ep_inst["must_dispatch"] | ~to_postpone

    return filter_instance(ep_inst, to_dispatch)
