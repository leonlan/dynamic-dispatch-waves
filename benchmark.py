import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import tomli
from pyvrp import CostEvaluator
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ddwp.agents import AGENTS, Agent
from ddwp.Environment import Environment
from ddwp.read import read
from ddwp.sampling import SAMPLING_METHODS
from ddwp.static_solvers import default_solver


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "instances", nargs="+", type=Path, help="Instance paths."
    )
    parser.add_argument(
        "--instance_format",
        type=str,
        choices=["vrplib", "solomon"],
        default="vrplib",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["euro_neurips", "paper"],
        default="euro_neurips",
    )
    parser.add_argument("--env_seed", type=int, default=1)
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=SAMPLING_METHODS.keys(),
        default="euro_neurips",
    )
    parser.add_argument(
        "--agent_config_loc",
        type=str,
        default="configs/icd-double-threshold.toml",
    )
    parser.add_argument("--agent_seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--num_procs_scenarios", type=int, default=1)
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument("--limited_vehicles", action="store_true")
    parser.add_argument("--epoch_tlim", type=float, default=60)
    parser.add_argument("--strategy_tlim", type=float, default=30)
    parser.add_argument("--sol_dir", type=str)

    return parser.parse_args()


def solve(
    loc: str,
    instance_format: str,
    environment: str,
    env_seed: int,
    sampling_method: str,
    agent_config_loc: str,
    agent_seed: int,
    num_procs_scenarios: int,
    hindsight: bool,
    limited_vehicles: bool,
    epoch_tlim: float,
    strategy_tlim: float,
    sol_dir: str,
    **kwargs,
):
    if strategy_tlim > epoch_tlim:
        raise ValueError("Strategy time limit >= epoch time limit.")

    path = Path(loc)
    static_instance = read(path, instance_format)

    if environment == "euro_neurips":
        env_constructor = Environment.euro_neurips  # type: ignore
    elif environment == "paper":
        env_constructor = Environment.paper  # type: ignore
    else:
        raise ValueError(f"Unknown environment: {environment}")

    env = env_constructor(
        env_seed,
        static_instance,
        epoch_tlim,
        SAMPLING_METHODS[sampling_method],
    )

    with open(agent_config_loc, "rb") as fh:
        config = tomli.load(fh)
        params = config.get("agent_params", {})

        if config["agent"] == "icd":
            # Set the scenario solving time limit based on the time budget for
            # scenarios divided by the total number of scenarios to solve.
            total = params["num_iterations"] * params["num_scenarios"]
            params["scenario_time_limit"] = strategy_tlim / total

            params["num_parallel_solve"] = num_procs_scenarios
            params["dispatch_time_limit"] = epoch_tlim - strategy_tlim
            params["sampling_method"] = SAMPLING_METHODS[sampling_method]

        if config["agent"] == "rolling_horizon":
            params["time_limit"] = epoch_tlim
            params["sampling_method"] = SAMPLING_METHODS[sampling_method]

        agent = AGENTS[config["agent"]](agent_seed, **params)

    start = perf_counter()

    if hindsight:
        costs, routes = solve_hindsight(env, agent_seed, epoch_tlim)
    elif limited_vehicles:
        # First use greedy to determine the number of available vehicles.
        greedy = AGENTS["greedy"](agent_seed)
        _, greedy_sol = solve_dynamic(env, greedy)
        num_vehicles_per_epoch = [len(route) for route in greedy_sol.values()]

        # Re-run environment with greedy number of available vehicles.
        env = env_constructor(
            env_seed,
            static_instance,
            epoch_tlim,
            SAMPLING_METHODS[sampling_method],
            num_vehicles_per_epoch=num_vehicles_per_epoch,
        )
        costs, routes = solve_dynamic(env, agent)
    else:
        costs, routes = solve_dynamic(env, agent)

    if sol_dir:
        instance_name = Path(loc).stem
        where = Path(sol_dir) / (instance_name + ".txt")

        with open(where, "w") as fh:
            fh.write(str(routes))

    return (
        path.stem,
        env_seed,
        agent_seed,
        sum(costs.values()),
        round(perf_counter() - start, 2),
    )


def solve_dynamic(env: Environment, agent: Agent):
    """
    Solves the dynamic problem.

    Parameters
    ----------
    env
        Environment of the dynamic problem.
    agent
        Agent that selects the dispatch action.
    """
    done = False
    observation, static_info = env.reset()

    while not done:
        action = agent.act(static_info, observation)
        observation, _, done = env.step(action)  # type: ignore
        assert observation is not None

    return env.final_costs, env.final_solutions


def solve_hindsight(env: Environment, seed: int, time_limit: float):
    """
    Solves the dynamic problem in hindsight.

    Parameters
    ----------
    env: Environment
        Environment of the dynamic problem.
    seed: int
        RNG seed used to solve the hindsight instances.
    time_limit: float
        Time limit for solving the hindsight instance.
    """
    observation, _ = env.reset()
    hindsight_inst = env.get_hindsight_problem()

    res = default_solver(hindsight_inst, seed, time_limit)
    hindsight_sol = [route.visits() for route in res.best.get_routes()]
    assert res.best.is_feasible(), "Infeasible hindsight solution."

    done = False
    observation, _ = env.reset()

    # Submit the solution from the hindsight instance to the environment to
    # verify feasibility of the solution.
    while not done:
        # Routes that have a maximum release time equal to the epoch departure
        # time are to be dispatched in the current epoch.
        departure_time = observation.departure_time
        ep_sol = [
            route
            for route in hindsight_sol
            if hindsight_inst.release_times[route].max() == departure_time
        ]

        observation, _, done = env.step(ep_sol)

    costs = env.final_costs
    solutions = env.final_solutions

    # Check that the cost of the dynamic problem is equal to the cost of the
    # hindsight solution.
    assert sum(costs.values()) == CostEvaluator(0, 0).cost(res.best)

    return costs, solutions


def maybe_mkdir(where: str):
    if where:
        stats_dir = Path(where)
        stats_dir.mkdir(parents=True, exist_ok=True)


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
    maybe_mkdir(kwargs.get("sol_dir", ""))

    func = partial(solve, **kwargs)
    args = sorted(instances)

    if len(instances) == 1 or num_procs == 1:
        res = [func(arg) for arg in tqdm(args, unit="instance")]
    else:
        res = process_map(func, args, max_workers=num_procs, unit="instance")

    dtypes = [
        ("inst", "U37"),
        ("env_seed", int),
        ("agent_seed", int),
        ("cost", int),
        ("time", float),
    ]
    data = np.asarray(res, dtype=dtypes)

    headers = [
        "Instance",
        "Env. seed",
        "Agent seed",
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
