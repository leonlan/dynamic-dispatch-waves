import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from pyvrp import CostEvaluator, Model
from pyvrp.stop import MaxRuntime
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import tools
from environments import EnvironmentCompetition
from strategies.config import Config
from strategies.dynamic import STRATEGIES
from strategies.utils import client2req
from tools import instance2data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--instance_format", default="vrplib")
    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument(
        "--dyn_config_loc", default="configs/fixed_threshold.toml"
    )
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument("--environment", type=str, default="competition")
    parser.add_argument("--epoch_tlim", type=float, default=60)

    return parser.parse_args()


def solve(
    loc: str,
    instance_format: str,
    instance_seed: int,
    solver_seed: int,
    dyn_config_loc: str,
    hindsight: bool,
    environment: str,
    epoch_tlim: int,
    **kwargs,
):
    path = Path(loc)

    if environment == "competition":
        env = EnvironmentCompetition(
            seed=instance_seed,
            instance=tools.read(path, instance_format),
            epoch_tlim=epoch_tlim,
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")

    start = perf_counter()

    if hindsight:
        costs, routes = solve_hindsight(env, solver_seed)
    else:
        dyn_config = Config.from_file(dyn_config_loc).dynamic()
        costs, routes = solve_dynamic(env, dyn_config, solver_seed)

    run_time = round(perf_counter() - start, 2)

    return (
        path.stem,
        instance_seed,
        sum(costs.values()),
        tuple(costs.values()),
        run_time,
    )


def tabulate(headers, rows) -> str:
    # These lengths are used to space each column properly.
    lengths = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            lengths[idx] = max(lengths[idx], len(str(cell)))

    header = [
        "  ".join(f"{h:<{l}s}" for l, h in zip(lengths, headers)),
        "  ".join("-" * l for l in lengths),
    ]

    content = [
        "  ".join(f"{str(c):>{l}s}" for l, c in zip(lengths, row))
        for row in rows
    ]

    return "\n".join(header + content)


def solve_dynamic(env, dyn_config, solver_seed):
    """
    Solve the dynamic VRPTW problem using the passed-in dispatching strategy.
    The given seed is used to initialise both the random number stream on the
    Python side, and for the static solver on the C++ side.

    Parameters
    ----------
    env: Environment
    dyn_config: Config
        Configuration object storing parameters for the dynamic solver.
    solver_seed: int
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

            ep_sol = client2req(routes, dispatch_inst)
        else:  # No requests to dispatch
            ep_sol = []

        current_epoch = observation["current_epoch"]
        solutions[current_epoch] = ep_sol

        observation, reward, done, info = env.step(ep_sol)
        costs[current_epoch] = abs(reward)

        assert info["error"] is None, info["error"]

    return costs, solutions


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


def benchmark_solve(instance: str, **kwargs):
    return solve(instance, **kwargs)


def benchmark(instances: list[str], num_procs: int = 1, **kwargs):
    func = partial(benchmark_solve, **kwargs)
    args = sorted(instances)

    if len(instances) == 1 or num_procs == 1:
        res = [func(arg) for arg in tqdm(args, unit="instance")]
    else:
        res = process_map(func, args, max_workers=num_procs, unit="instance")

    dtypes = [
        ("inst", "U37"),
        ("seed", int),
        ("total", int),
        ("costs", tuple),
        ("time", float),
    ]
    data = np.asarray(res, dtype=dtypes)

    headers = [
        "Instance",
        "Seed",
        "Total",
        "Costs",
        "Time (s)",
    ]

    table = tabulate(headers, data)

    print("\n", table, "\n", sep="")
    print(f"      Avg. objective: {data['total'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


def main():
    benchmark(**vars(parse_args()))


if __name__ == "__main__":
    main()
