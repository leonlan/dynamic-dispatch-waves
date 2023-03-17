import argparse
from functools import partial
from glob import glob
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.contrib.concurrent import process_map

import tools
import hgspy
from environment_competition import VRPEnvironment
from strategies import solve_dynamic, solve_hindsight
from strategies.config import Config
from strategies.static import hgs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument("--dyn_config_loc", default="configs/dynamic.toml")
    parser.add_argument("--sim_config_loc", default="configs/simulation.toml")
    parser.add_argument("--disp_config_loc", default="configs/dispatch.toml")
    parser.add_argument(
        "--instance_pattern", default="instances/ortec/ORTEC-VRPTW-ASYM-*.txt"
    )
    parser.add_argument("--instance_format", default="vrplib")
    parser.add_argument("--epoch_tlim", type=float, default=60)

    return parser.parse_args()


def solve(
    loc: str,
    instance_seed: int,
    instance_format: str,
    solver_seed: int,
    config_loc: str,
    sim_config_loc: str,
    disp_config_loc: str,
    hindsight: bool,
    epoch_tlim: int,
    **kwargs,
):
    path = Path(loc)

    env = VRPEnvironment(
        seed=instance_seed,
        instance=tools.io.read_vrplib(path, instance_format),
        epoch_tlim=epoch_tlim,
    )

    start = perf_counter()

    dyn_config = Config.from_file(dyn_config_loc).dynamic()
    sim_config = Config.from_file(sim_config_loc).static()
    disp_config = Config.from_file(disp_config_loc).static()

    if hindsight:
        costs, routes = solve_hindsight(env, disp_config, solver_seed)
    else:
        costs, routes = solve_dynamic(
            env, dyn_config, disp_config, sim_config, solver_seed
        )

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
