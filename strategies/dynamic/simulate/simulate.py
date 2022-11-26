import numpy as np

import hgspy
from plot.plot_dynamic_instance import save_fig
from strategies.static import hgs
from strategies.utils import filter_instance
from .simulate_instance import simulate_instance


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
    n_ep_reqs = ep_inst["is_depot"].size
    must_dispatch = set(np.flatnonzero(ep_inst["must_dispatch"]))
    total_sim_tlim = simulate_tlim_factor * info["epoch_tlim"]
    single_sim_tlim = total_sim_tlim / (n_cycles * n_simulations)

    dispatch_count = np.zeros(n_ep_reqs, dtype=int)
    to_postpone = np.zeros(n_ep_reqs, dtype=bool)

    # Get the threshold belonging to the current epoch, or the last one
    # available if there are more epochs than thresholds.
    epoch = obs["current_epoch"] - info["start_epoch"]
    num_thresholds = len(postpone_thresholds)
    postpone_threshold = postpone_thresholds[min(epoch, num_thresholds - 1)]

    for cycle_idx in range(n_cycles):
        for sim_idx in range(n_simulations):
            sim_inst = simulate_instance(
                info,
                obs,
                rng,
                n_lookahead,
                n_requests,
                ep_release=to_postpone * 3600,
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

            for sim_route in best.get_routes():
                # Only dispatch routes that contain must dispatch requests
                if any(idx in must_dispatch for idx in sim_route):
                    dispatch_count[sim_route] += 1

            dispatch_count[0] += 1  # depot

            # We need an array of sim_inst size to also draw the postponed
            # requests in the simulation cycles >= 2
            sim_to_postpone = np.zeros_like(sim_inst["is_depot"])
            sim_to_postpone[to_postpone.nonzero()] = True

            if epoch != 2:  # Only plot epoch 2
                continue

            # Plot simulation instance
            save_fig(
                f"figs/simulation_instance_{epoch}_{cycle_idx}_{sim_idx}.jpg",
                "Simulation instance",
                sim_inst,
                postponed=sim_to_postpone,
                description="""
We sample future requests to obtain a simulated instance.
Each simulated request has a nonzero release date.""",
            )

            if cycle_idx > 0:
                description = """
We repeat the full simulation procedure.
The postponed requests now also have a nonzero release date."""
            elif sim_idx == 0:
                description = """
We solve the simulation instance as VRPTW problem with release dates.
The instance is solved very briefly, using less than 0.5 seconds."""
            else:
                description = """
We repeat this for 40 simulation instances."""

            # Plot simulation instance with solution
            save_fig(
                f"figs/simulation_instance_with_solution_{epoch}_{cycle_idx}_{sim_idx}.jpg",
                "Simulation instance",
                sim_inst,
                best.get_routes(),
                postponed=sim_to_postpone,
                description=description,
            )

        # Select requests to postpone based on thresholds
        postpone_count = n_simulations - dispatch_count
        to_postpone = postpone_count >= postpone_threshold * n_simulations

        dispatch_count *= 0  # reset dispatch count

        if epoch != 2:  # Only plot epoch 2
            continue

        if cycle_idx > 0:
            desc = """
We again mark the requests that were frequently postponed in the simulations."""
        else:
            desc = """
We count for each request how often it was postponed.
We mark all requests with postponement frequency higher than a threshold value."""

        # HACK A proxy to find the requests that were postponed during
        # previous simulation cycles
        already_postponed = postpone_count == n_simulations
        already_postponed_idcs = np.flatnonzero(already_postponed)

        # Plot dispatch instance with thresholds as labels
        labels = dict(enumerate((postpone_count / n_simulations).round(2)))
        labels = {
            k: v for k, v in labels.items() if k not in already_postponed_idcs
        }  # Ignore already postponed requests

        save_fig(
            f"figs/epoch_instance_with_labels_{epoch}_{cycle_idx}.jpg",
            "Epoch instance",
            ep_inst,
            labels=labels,
            description=desc,
            postponed=already_postponed,
        )

        # Plot dispatch instance after first simulation cycle
        save_fig(
            f"figs/epoch_instance_with_labels_and_colors_{epoch}_{cycle_idx}.jpg",
            "Epoch instance",
            ep_inst,
            labels=labels,
            description=desc,
            postponed=to_postpone,
        )

    to_dispatch = ep_inst["is_depot"] | ep_inst["must_dispatch"] | ~to_postpone

    return filter_instance(ep_inst, to_dispatch)
