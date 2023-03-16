import numpy as np

import hgspy
from strategies.dynamic import STRATEGIES
from strategies.static import hgs
from .utils import sol2ep


def solve_dynamic(env, config, sim_solver, disp_solver, solver_seed):
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
        strategy = STRATEGIES[config.strategy()]
        dispatch_inst = strategy(
            env,
            static_info,
            observation,
            rng,
            sim_solver,
            **config.strategy_params()
        )

        solve_tlim = ep_tlim

        # Reduce the solving time limit by the simulation time
        strategy_params = config.get("strategy_params", {})
        sim_tlim_factor = strategy_params.get("simulate_tlim_factor", 0)
        solve_tlim *= 1 - sim_tlim_factor

        res = disp_solver(dispatch_inst, solve_tlim)
        best = res.get_best_found()
        routes = [route for route in best.get_routes() if route]

        ep_sol = sol2ep(routes, dispatch_inst, config["postpone_routes"])

        current_epoch = observation["current_epoch"]
        solutions[current_epoch] = ep_sol

        observation, reward, done, info = env.step(ep_sol)
        costs[current_epoch] = abs(reward)

        assert info["error"] is None, info["error"]

    return costs, solutions
