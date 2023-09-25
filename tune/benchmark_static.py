import argparse
from functools import partial
from pathlib import Path

import numpy as np

try:
    import tomli
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map
except ModuleNotFoundError as exc:
    msg = "Install 'tqdm' and 'tomli' to use the command line program."
    raise ModuleNotFoundError(msg) from exc

import pyvrp.search
from pyvrp import (
    GeneticAlgorithm,
    GeneticAlgorithmParams,
    PenaltyManager,
    PenaltyParams,
    Population,
    PopulationParams,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.read import read
from pyvrp.search import (
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
)
from pyvrp.stop import MaxRuntime


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--instance_format", default="solomon")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--config_loc", default="configs/benchmark.toml")
    parser.add_argument("--max_runtime", type=float)

    return parser.parse_args()


def tabulate(headers: list[str], rows: np.ndarray) -> str:
    """
    Creates a simple table from the given header and row data.
    """
    # These lengths are used to space each column properly.
    lens = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            lens[idx] = max(lens[idx], len(str(cell)))

    header = [
        "  ".join(f"{hdr:<{ln}s}" for ln, hdr in zip(lens, headers)),
        "  ".join("-" * ln for ln in lens),
    ]

    content = [
        "  ".join(f"{c!s:>{ln}s}" for ln, c in zip(lens, r)) for r in rows
    ]

    return "\n".join(header + content)


def solve(
    data_loc: Path,
    instance_format: str,
    seed: int,
    max_runtime: float,
    **kwargs,
) -> tuple[str, str, float, int, float]:
    """
    Solves a single VRPLIB instance.

    Parameters
    ----------
    data_loc
        Filesystem location of the VRPLIB instance.
    instance_format
        Data format of the filesystem instance. Argument is passed to
        ``read()``.
    seed
        Seed to use for the RNG.
    max_runtime
        Maximum runtime (in seconds) for solving. Either ``max_runtime`` or
        ``max_iterations`` must be specified.

    Returns
    -------
    tuple[str, str, float, int, float]
        A tuple containing the instance name, whether the solution is feasible,
        the solution cost, the number of iterations, and the runtime.
    """
    if kwargs.get("config_loc"):
        with open(kwargs["config_loc"], "rb") as fh:
            config = tomli.load(fh)
    else:
        config = {}

    gen_params = GeneticAlgorithmParams(**config.get("genetic", {}))
    pen_params = PenaltyParams(**config.get("penalty", {}))
    pop_params = PopulationParams(**config.get("population", {}))
    nb_params = NeighbourhoodParams(**config.get("neighbourhood", {}))

    data = read(data_loc, instance_format, "dimacs")
    rng = RandomNumberGenerator(seed=seed)
    pen_manager = PenaltyManager(pen_params)
    pop = Population(bpd, params=pop_params)

    neighbours = compute_neighbours(data, nb_params)
    ls = LocalSearch(data, rng, neighbours)

    node_ops = [
        getattr(pyvrp.search, op)
        for op, include in config["node_ops"].items()
        if include
    ]
    for node_op in node_ops:
        ls.add_node_operator(node_op(data))

    route_ops = [
        getattr(pyvrp.search, op)
        for op, include in config["route_ops"].items()
        if include
    ]
    for route_op in route_ops:
        ls.add_route_operator(route_op(data))

    init = [
        Solution.make_random(data, rng) for _ in range(pop_params.min_pop_size)
    ]
    algo = GeneticAlgorithm(
        data, pen_manager, rng, pop, ls, srex, init, gen_params
    )
    stop = MaxRuntime(max_runtime)

    result = algo.run(stop)
    instance_name = data_loc.stem

    return (
        instance_name,
        "Y" if result.is_feasible() else "N",
        round(result.cost(), 2),
        result.num_iterations,
        round(result.runtime, 3),
    )


def benchmark(instances: list[Path], num_procs: int = 1, **kwargs):
    """
    Solves a list of instances, and prints a table with the results. Any
    additional keyword arguments are passed to ``solve()``.

    Parameters
    ----------
    instances
        Paths to the VRPLIB instances to solve.
    num_procs
        Number of processors to use. Default 1.
    kwargs
        Any additional keyword arguments to pass to the solving function.
    """
    func = partial(solve, **kwargs)
    args = sorted(instances)

    if len(instances) == 1 or num_procs == 1:
        res = [func(arg) for arg in tqdm(args, unit="instance")]
    else:
        res = process_map(func, args, max_workers=num_procs, unit="instance")

    dtypes = [
        ("inst", "U37"),
        ("ok", "U1"),
        ("obj", float),
        ("iters", int),
        ("time", float),
    ]

    data = np.asarray(res, dtype=dtypes)
    headers = ["Instance", "OK", "Obj.", "Iters. (#)", "Time (s)"]

    print("\n", tabulate(headers, data), "\n", sep="")
    print(f"      Avg. objective: {data['obj'].mean():.0f}")
    print(f"     Avg. iterations: {data['iters'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")
    print(f"        Total not OK: {np.count_nonzero(data['ok'] == 'N')}")


if __name__ == "__main__":
    benchmark(**vars(parse_args()))
