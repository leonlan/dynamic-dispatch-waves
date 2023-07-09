import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime

from instance2data import instance2data
from strategies.dynamic import STRATEGIES

from .utils import sol2ep


def solve_dynamic(env, dyn_config, solver_seed):
    """
    Solve the dynamic VRPTW problem using the passed-in dispatching strategy.
    The given seed is used to initialise both the random number stream on the
    Python side, and for the static solver on the C++ side.

    Parameters
    ----------
    env : Environment
    dyn_config : Config
        Configuration object storing parameters for the dynamic solver.
    solver_seed : int
        RNG seed for the dynamic solver.
    """
    rng = np.random.default_rng(solver_seed)

    observation, static_info = env.reset()
    ep_tlim = static_info["epoch_tlim"]

    solutions = {}
    costs = {}
    done = False

    while not done:
        strategy = STRATEGIES[dyn_config.strategy()]
        dispatch_inst = strategy(
            env, static_info, observation, rng, **dyn_config.strategy_params()
        )

        solve_tlim = ep_tlim

        # Reduce the solving time limit by the simulation time
        strategy_params = dyn_config.get("strategy_params", {})
        strategy_tlim_factor = strategy_params.get("strategy_tlim_factor", 0)
        solve_tlim *= 1 - strategy_tlim_factor

        if dispatch_inst["request_idx"].size > 1:
            model = Model.from_data(instance2data(dispatch_inst))
            res = model.solve(MaxRuntime(solve_tlim), seed=solver_seed)
            routes = [
                route.visits() for route in res.best.get_routes() if route
            ]

            ep_sol = sol2ep(
                routes, dispatch_inst, dyn_config["postpone_routes"]
            )
        else:  # No requests to dispatch
            ep_sol = []

        current_epoch = observation["current_epoch"]
        solutions[current_epoch] = ep_sol

        observation, reward, done, info = env.step(ep_sol)
        costs[current_epoch] = abs(reward)

        assert info["error"] is None, info["error"]

    return costs, solutions
