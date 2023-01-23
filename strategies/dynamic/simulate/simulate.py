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

    for cycle_idx in range(n_cycles):
        # Get the threshold belonging to the current epoch, or the last one
        # available if there are more epochs than thresholds.
        threshold_idx = min(cycle_idx, len(postpone_thresholds) - 1)
        postpone_threshold = postpone_thresholds[threshold_idx]
        dispatch_threshold = dispatch_thresholds[threshold_idx]

        # Compute the initial solution for to dispatch requests. These form the
        # "backbone" of the simulation solutions.
        to_dispatch_instance = filter_instance(
            ep_inst, to_dispatch | ep_inst["is_depot"]
        )
        to_dispatch_res = hgs(
            to_dispatch_instance,
            hgspy.Config(),
            [getattr(hgspy.operators, op) for op in node_ops],
            [getattr(hgspy.operators, op) for op in route_ops],
            [getattr(hgspy.crossover, op) for op in crossover_ops],
            hgspy.stop.MaxRuntime(5),
        )

        # We need to map the solution back to the epoch instance indices.
        idx2cust = {
            idx: cust for idx, cust in enumerate(to_dispatch.nonzero()[0], 1)
        }
        to_dispatch_init = [
            [idx2cust[idx] for idx in route]
            for route in to_dispatch_res.get_best_found().get_routes()
            if route
        ]

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
            # We need to add the "non-initialized" requests to make a complete
            # initial solution.
            sim_init = [
                [req]
                for req in range(1, sim_inst["is_depot"].size)
                if req not in to_dispatch.nonzero()[0]
            ]

            res = hgs(
                sim_inst,
                hgspy.Config(**sim_config),
                [getattr(hgspy.operators, op) for op in node_ops],
                [getattr(hgspy.operators, op) for op in route_ops],
                [getattr(hgspy.crossover, op) for op in crossover_ops],
                hgspy.stop.MaxRuntime(single_sim_tlim),
                initial_solutions=(to_dispatch_init + sim_init,),
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
        to_postpone = postpone_count > postpone_threshold * n_simulations
        to_postpone[0] = False  # Do not postpone the depot

        print(
            ep_size,
            " | ",
            to_dispatch.sum() + to_postpone.sum(),
            " = ",
            to_dispatch.sum(),
            " + ",
            to_postpone.sum(),
        )

        # Stop the simulation run early when all requests have been marked
        if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
            break

        dispatch_count *= 0  # reset dispatch count

    # TODO This should become a parameter: we can also dispatch only those
    # requests that have been marked `to_dispatch`.
    # Dispatch all requests that are not marked `to_postpone`
    to_dispatch = ep_inst["is_depot"] | ep_inst["must_dispatch"] | to_dispatch

    return filter_instance(ep_inst, to_dispatch)
