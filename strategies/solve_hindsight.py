from pyvrp import CostEvaluator, Model
from pyvrp.stop import MaxRuntime

from tools import instance2data


def solve_hindsight(env, solver_seed: int):
    """
    Solve the dynamic VRPTW problem using the oracle strategy, i.e., the
    problem is solved as static VRPTW with release dates using the information
    that is known in hindsight. The found solution is then submitted to the
    environment. The given seed is passed to the static solver.
    """
    observation, info = env.reset()
    hindsight_inst = env.get_hindsight_problem()

    model = Model.from_data(instance2data(hindsight_inst))
    res = model.solve(MaxRuntime(info["epoch_tlim"]), seed=solver_seed)

    best = res.best
    routes = [route.visits() for route in best.get_routes() if route]
    observation, _ = env.reset()

    # Submit the solution from the hindsight problem
    while not env.is_done:
        ep_inst = observation["epoch_instance"]
        requests = set(ep_inst["request_idx"])

        # This is a proxy to extract the routes from the hindsight
        # solution that are dispatched in the current epoch.
        ep_sol = [
            route
            for route in routes
            if len(requests.intersection(route)) == len(route)
        ]

        observation, _, _, info = env.step(ep_sol)
        assert info["error"] is None, f"{info['error']}"

    # Check that the cost of the dynamic problem is equal to the cost of the
    # hindsight solution.
    cost_eval = CostEvaluator(0, 0)
    assert sum(env.final_costs.values()) == cost_eval.cost(best)

    return env.final_costs, env.final_solutions
