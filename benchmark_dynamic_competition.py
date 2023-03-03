import argparse
from functools import partial
from glob import glob
from pathlib import Path
import pickle
from time import perf_counter
from typing import Optional

import numpy as np
from tqdm.contrib.concurrent import process_map

import tools
from environment_competition import VRPEnvironment
from strategies import solve_dynamic, solve_hindsight
from strategies.config import Config
from strategies.statistics import Statistics


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
        "--instance_pattern", default="instances/ortec/ORTEC-VRPTW-ASYM-*.txt"
    )
    parser.add_argument("--epoch_tlim", type=float, default=60)
    parser.add_argument("--stats_dir")

    return parser.parse_args()


def solve(
    loc: str,
    instance_seed: int,
    solver_seed: int,
    config_loc: str,
    hindsight: bool,
    epoch_tlim: int,
    stats_dir: Optional[str],
    **kwargs,
):
    path = Path(loc)

    env = VRPEnvironment(
        seed=instance_seed,
        instance=tools.io.read_vrplib(path),
        epoch_tlim=epoch_tlim,
    )

    start = perf_counter()

    config = Config.from_file(config_loc)
    stats = Statistics(config)

    if hindsight:
        costs, routes = solve_hindsight(env, config.static(), solver_seed)
    else:
        costs, routes = solve_dynamic(env, config, stats, solver_seed)

    run_time = round(perf_counter() - start, 2)

    if stats_dir:
        where = Path(stats_dir)
        where.mkdir(parents=True, exist_ok=True)
        with open(where / (path.stem + ".pickle"), "wb") as fh:
            pickle.dump(stats, fh)

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

    table = tools.io.tabulate(headers, data)

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
