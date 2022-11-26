import argparse
from functools import partial
from glob import glob
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.contrib.concurrent import process_map

import tools
from environment import VRPEnvironment
from strategies import solve_dynamic, solve_hindsight
from strategies.config import Config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument(
        "--config_loc", default="configs/benchmark_dynamic.toml"
    )
    parser.add_argument(
        "--instance_pattern", default="instances/ORTEC-VRPTW-ASYM-*.txt"
    )
    parser.add_argument("--epoch_tlim", type=int, default=60)

    return parser.parse_args()


def solve(
    loc: str,
    instance_seed: int,
    solver_seed: int,
    config_loc: str,
    hindsight: bool,
    epoch_tlim: int,
    **kwargs,
):
    path = Path(loc)

    instance = tools.read_vrplib(path)
    instance["capacity"] = 80  # Smaller capacity to make shorter routes

    env = VRPEnvironment(
        seed=instance_seed,
        instance=instance,
        epoch_tlim=epoch_tlim,
        max_requests_per_epoch=10,
    )

    start = perf_counter()

    config = Config.from_file(config_loc)

    if hindsight:
        costs, routes = solve_hindsight(env, config.static(), solver_seed)
    else:
        costs, routes = solve_dynamic(env, config, solver_seed)

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


def main():
    args = parse_args()

    func = partial(solve, **vars(args))
    func_args = glob(args.instance_pattern)

    if args.num_procs > 1:
        tqdm_kwargs = dict(max_workers=args.num_procs, unit="instance")
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

    table = tools.tabulate(headers, data)

    print(
        Path(__file__).name,
        " ".join(f"--{key} {value}" for key, value in vars(args).items()),
    )

    config = Config.from_file(args.config_loc)
    print("dynamic config:")
    print(config.dynamic())

    print("\n", table, "\n", sep="")
    print(f"      Avg. objective: {data['total'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


if __name__ == "__main__":
    main()
