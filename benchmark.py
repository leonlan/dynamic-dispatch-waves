import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from pyvrp import CostEvaluator, Model
from pyvrp.stop import MaxRuntime
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import utils
from environments import EnvironmentCompetition
from strategies.config import Config
from strategies.dynamic import STRATEGIES
from utils import instance2data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--env_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument(
        "--dyn_config_loc", default="configs/fixed_threshold.toml"
    )
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument("--epoch_tlim", type=float, default=60)

    return parser.parse_args()


def solve(
    loc: str,
    env_seed: int,
    solver_seed: int,
    dyn_config_loc: str,
    hindsight: bool,
    epoch_tlim: int,
    **kwargs,
):
    path = Path(loc)
    static_instance = utils.read(path)
    env = EnvironmentCompetition(env_seed, static_instance, epoch_tlim)

    start = perf_counter()

    if hindsight:
        costs, _ = solve_hindsight(env, solver_seed)
    else:
        dyn_config = Config.from_file(dyn_config_loc).dynamic()
        costs, _ = solve_dynamic(env, dyn_config, solver_seed)

    return (
        path.stem,
        env_seed,
        sum(costs),
        round(perf_counter() - start, 2),
    )


def solve_dynamic(env, dyn_config, solver_seed):
    """
    Solves the dynamic problem using the passed-in dynamic strategy
    configuration.

    Parameters
    ----------
    env: Environment
        Environment of the dynamic problem.
    dyn_config: Config
        Configuration object storing parameters for the dynamic solver.
    solver_seed: int
        RNG seed for the dynamic solver.
    """
    rng = np.random.default_rng(solver_seed)

    done = False
    solutions = []
    costs = []
    observation, static_info = env.reset()
    ep_tlim = static_info["epoch_tlim"]

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

        if dispatch_inst["request_idx"].size == 1:
            # Empty dispatch instance, so no requests to dispatch.
            ep_sol = []
        else:
            model = Model.from_data(instance2data(dispatch_inst))
            res = model.solve(MaxRuntime(solve_tlim), seed=solver_seed)
            routes = [route.visits() for route in res.best.get_routes()]

            # Map solution client indices to request indices.
            ep_sol = [dispatch_inst["request_idx"][route] for route in routes]

        observation, reward, done, info = env.step(ep_sol)
        assert info["error"] is None, info["error"]

        solutions.append(ep_sol)
        costs.append(abs(reward))

    return costs, solutions


def solve_hindsight(env, solver_seed: int):
    """
    Solves the dynamic problem in hindsight.
    """
    observation, info = env.reset()
    hindsight_inst = env.get_hindsight_problem()

    # Solve the hindsight instance using PyVRP.
    model = Model.from_data(instance2data(hindsight_inst))
    res = model.solve(MaxRuntime(info["epoch_tlim"]), seed=solver_seed)
    hindsight_sol = [route.visits() for route in res.best.get_routes()]

    done = False
    solutions = []
    costs = []
    observation, _ = env.reset()

    # Submit the solution from the hindsight instance to the environment to
    # verify feasibility of the solution.
    while not done:
        # Routes with a maximum release time of the dispatch time, i.e., the
        # moment that the routes will be dispatched if they are submitted as
        # the current epoch solution.
        dispatch_time = observation["dispatch_time"]
        ep_sol = [
            route
            for route in hindsight_sol
            if hindsight_inst["release_times"][route].max() == dispatch_time
        ]

        observation, reward, done, info = env.step(ep_sol)
        assert info["error"] is None, f"{info['error']}"

        solutions.append(ep_sol)
        costs.append(abs(reward))

    # Check that the cost of the dynamic problem is equal to the cost of the
    # hindsight solution.
    cost_eval = CostEvaluator(0, 0)
    assert sum(costs) == cost_eval.cost(res.best)

    return costs, solutions


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


def benchmark(instances: list[str], num_procs: int = 1, **kwargs):
    func = partial(solve, **kwargs)
    args = sorted(instances)

    if len(instances) == 1 or num_procs == 1:
        res = [func(arg) for arg in tqdm(args, unit="instance")]
    else:
        res = process_map(func, args, max_workers=num_procs, unit="instance")

    dtypes = [
        ("inst", "U37"),
        ("seed", int),
        ("cost", int),
        ("time", float),
    ]
    data = np.asarray(res, dtype=dtypes)

    headers = [
        "Instance",
        "Seed",
        "Cost",
        "Time (s)",
    ]

    table = tabulate(headers, data)

    print("\n", table, "\n", sep="")
    print(f"      Avg. objective: {data['cost'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


def main():
    benchmark(**vars(parse_args()))


if __name__ == "__main__":
    main()
