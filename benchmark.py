import argparse
from functools import partial
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Union

import numpy as np
from pyvrp import CostEvaluator, Model
from pyvrp.stop import MaxRuntime
from tqdm.contrib.concurrent import process_map

import tools
from environments import Environment, EnvironmentCompetition
from strategies.config import Config
from strategies.dynamic import STRATEGIES
from strategies.utils import client2req
from tools import instance2data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instance_pattern", default="instances/ortec/ORTEC-VRPTW-ASYM-*.txt"
    )
    parser.add_argument("--instance_format", default="vrplib")
    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument(
        "--dyn_config_loc", default="configs/fixed_threshold.toml"
    )
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument(
        "--environment",
        type=str,
        default="paper",
        choices=["paper", "competition"],
    )
    parser.add_argument("--epoch_tlim", type=float, default=60)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument(
        "--requests_per_epoch", type=int, nargs="+", default=50
    )
    parser.add_argument(
        "--time_window_style", type=str, default="fixed_time_windows"
    )
    parser.add_argument("--time_window_width", type=int, default=2)
    return parser.parse_args()


def manhattan_distance_matrix(coords):
    coords_expanded = coords[:, np.newaxis, :]
    diffs = np.abs(coords_expanded - coords)
    distance_matrix = np.sum(diffs, axis=-1)
    return distance_matrix


def solve(
    loc: str,
    instance_format: str,
    instance_seed: int,
    solver_seed: int,
    dyn_config_loc: str,
    hindsight: bool,
    environment: str,
    epoch_tlim: int,
    num_epochs: int,
    requests_per_epoch: Union[int, list],
    time_window_style: str,
    time_window_width: int,
    **kwargs,
):
    path = Path(loc)

    env: Union[Environment, EnvironmentCompetition]

    if environment == "competition":
        env = EnvironmentCompetition(
            seed=instance_seed,
            instance=tools.read(path, instance_format),
            epoch_tlim=epoch_tlim,
        )
    else:
        instance = tools.read(path, instance_format)
        TIME = 600  # TODO rename this

        # Normalize the distances so that the further customer can be served
        # in one hour. Service times are also scaled accordingly.
        factor = instance["duration_matrix"].max() / TIME
        instance["duration_matrix"] = np.ceil(
            instance["duration_matrix"] / factor
        ).astype(int)
        instance["service_times"] = np.ceil(
            instance["service_times"] / factor
        ).astype(int)

        # Normalize the depot time windows to be TIME * ``num_epochs``; customer
        # time windows are not used for the sampling.
        instance["time_windows"][0, :] = [0, num_epochs * TIME]

        env = Environment(
            seed=instance_seed,
            instance=instance,
            epoch_tlim=epoch_tlim,
            num_epochs=num_epochs,
            requests_per_epoch=requests_per_epoch,
            time_window_style=time_window_style,
            time_window_width=time_window_width,
        )

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
        tuple([len(rts) for rts in routes.values()]),
        tuple([sum(len(route) for route in sol) for sol in routes.values()]),
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


def main():
    args = parse_args()

    func = partial(solve, **vars(args))
    func_args = glob(args.instance_pattern)

    if args.num_procs > 1:
        tqdm_kwargs = {"max_workers": args.num_procs, "unit": "instance"}
        data = process_map(func, func_args, **tqdm_kwargs)
    else:  # process_map cannot be used with interactive debugging
        data = [func(args) for args in func_args]

    headers = [
        "Instance",
        "Seed",
        "Total",
        "Costs",
        "Routes",
        "Requests",
        "Time (s)",
    ]

    dtypes = [
        ("inst", "U37"),
        ("seed", int),
        ("total", int),
        ("costs", tuple),
        ("routes", tuple),
        ("requests", tuple),
        ("time", float),
    ]
    data = np.asarray(data, dtype=dtypes)

    table = tabulate(headers, data)

    print(
        Path(__file__).name,
        " ".join(f"--{key} {value}" for key, value in vars(args).items()),
    )

    print("\n", table, "\n", sep="")
    print(f"      Avg. objective: {data['total'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


if __name__ == "__main__":
    main()
