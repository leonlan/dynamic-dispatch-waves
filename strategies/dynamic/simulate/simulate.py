import numpy as np

import hgspy
from strategies.static import hgs
from strategies.utils import filter_instance
from .simulate_instance import simulate_instance
from .threshold import threshold
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

    # Actions
    to_dispatch = ep_inst["must_dispatch"]
    to_postpone = np.zeros(ep_size, dtype=bool)

    for cycle_idx in range(n_cycles):
        disp_init = solve_dispatch_inst(
            ep_inst, to_dispatch, node_ops, route_ops, crossover_ops
        )

        solutions_pool = []

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
                initial_solutions=(make_sim_init(sim_inst, disp_init),),
            )

            sim_sol = res.get_best_found().get_routes()
            tools.validate_static_solution(sim_inst, [x for x in sim_sol if x])

            solutions_pool.append(sim_sol)

        to_dispatch, to_postpone = threshold(
            solutions_pool,
            to_dispatch,
            to_postpone,
            cycle_idx,
            dispatch_thresholds,
            postpone_thresholds,
        )

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

    return filter_instance(ep_inst, ep_inst["is_depot"] | to_dispatch)


def solve_dispatch_inst(
    ep_inst, to_dispatch, node_ops, route_ops, crossover_ops
):
    """
    Solves the instance formed by dispatched requests. The solution indices
    are mapped back to the indices of the epoch instance.
    """
    inst = filter_instance(ep_inst, to_dispatch | ep_inst["is_depot"])
    res = hgs(
        inst,
        hgspy.Config(),
        [getattr(hgspy.operators, op) for op in node_ops],
        [getattr(hgspy.operators, op) for op in route_ops],
        [getattr(hgspy.crossover, op) for op in crossover_ops],
        hgspy.stop.MaxRuntime(3),  # TODO Make param
    )

    # Map the new indices back to the epoch instance indices.
    ep_idcs = to_dispatch.nonzero()[0]
    idx2ep = {idx: cust for idx, cust in enumerate(ep_idcs, 1)}

    sol = res.get_best_found().get_routes()
    disp_init = [[idx2ep[idx] for idx in route] for route in sol if route]
    return disp_init


def make_sim_init(sim_inst, disp_init):
    """
    Makes the initial solution for the simulation instance. The HGS solver only
    accepts initial solutions that contain all requests.
    """
    n_reqs = sim_inst["is_depot"].size
    disp = set(r for route in disp_init for r in route)

    # Make roundtrip routes for requests not marked dispatch
    roundtrips = [[req] for req in range(1, n_reqs) if req not in disp]
    return disp_init + roundtrips
