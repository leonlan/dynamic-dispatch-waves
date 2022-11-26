import time

import numpy as np
import matplotlib.pyplot as plt

import hgspy
from plot.plot_dynamic_instance import save_fig
from strategies.dynamic import STRATEGIES
from strategies.static import hgs
from .utils import sol2ep


def solve_dynamic(env, config, solver_seed):
    """
    Solve the dynamic VRPTW problem using the passed-in dispatching strategy.
    The given seed is used to initialise both the random number stream on the
    Python side, and for the static solver on the C++ side.

    Parameters
    ----------
    env : Environment
    config : Config
        The configuration object, storing the strategy and solver parameters.
    solver_seed : int
        RNG seed. Seed is shared between static and dynamic solver.
    """
    rng = np.random.default_rng(solver_seed)

    observation, static_info = env.reset()
    ep_tlim = static_info["epoch_tlim"]

    solutions = {}
    costs = {}
    done = False

    config = config.dynamic()

    while not done:
        start = time.perf_counter()

        strategy = STRATEGIES[config.strategy()]
        ep_inst = observation["epoch_instance"]

        epoch = observation["current_epoch"] - static_info["start_epoch"]

        dispatch_inst = strategy(
            static_info, observation, rng, **config.strategy_params()
        )

        solve_tlim = ep_tlim - (time.perf_counter() - start) + 1
        solve_tlim = 3  # Temporary

        # TODO use a seed different from the dynamic rng for the static solver
        res = hgs(
            dispatch_inst,
            hgspy.Config(seed=solver_seed, **config.solver_params()),
            config.node_ops(),
            config.route_ops(),
            config.crossover_ops(),
            hgspy.stop.MaxRuntime(solve_tlim),
        )

        best = res.get_best_found()
        routes = [route for route in best.get_routes() if route]

        ep_sol = sol2ep(routes, dispatch_inst, postpone_routes=False)

        if epoch == 2:  # This is a nice epoch instance!
            # Plot epoch instance
            save_fig(
                f"figs/epoch_instance_{epoch}.jpg",
                "Epoch instance",
                ep_inst,
                description="""
The epoch instance consists of must-dispatch and optional requests.
Optional requests may be postponed to the next epoch.""",
            )

            desc = """
After finishing the simulation cycles, we remove the postponed requests,
and solve the resulting instance for roughly 20 seconds."""

            # Plot dispatch instance
            save_fig(
                f"figs/dispatch_instance_{epoch}.jpg",
                "Dispatch instance",
                dispatch_inst,
                description=desc,
            )

            # "Final" dispatch instance with solution
            save_fig(
                f"figs/dispatch_instance_with_solution_{epoch}.jpg",
                "Dispatch instance",
                dispatch_inst,
                routes,
                description=desc,
            )

            # Dispatch instance with must-dispatch routes only (solution+)
            save_fig(
                f"figs/dispatch_instance_with_solution_plus_{epoch}.jpg",
                "Dispatch instance",
                dispatch_inst,
                [
                    route
                    for route in best.get_routes()
                    if dispatch_inst["must_dispatch"][route].any()
                ],
                description="""
Finally, we remove all postpone-able routes and submit this new solution to the environment.""",
            )

        # Submit solution
        ep_sol = sol2ep(routes, dispatch_inst, postpone_routes=True)
        current_epoch = observation["current_epoch"]
        solutions[current_epoch] = ep_sol

        observation, reward, done, info = env.step(ep_sol)
        costs[current_epoch] = abs(reward)

        assert info["error"] is None, info["error"]

    return costs, solutions
