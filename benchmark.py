import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import tomli
from pyvrp import CostEvaluator
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import utils
from agents import AGENTS, Agent
from environments import EnvironmentCompetition
from sampling import sample_epoch_requests
from static_solvers import default_solver
from utils import filter_instance


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--env_seed", type=int, default=1)
    parser.add_argument("--agent_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument(
        "--agent_config_loc",
        type=str,
        default="configs/icd-double-threshold.toml",
    )
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--hindsight", action="store_true")
    parser.add_argument("--epoch_tlim", type=float, default=60)
    parser.add_argument("--solve_tlim", type=float, default=10)

    return parser.parse_args()


def solve(
    loc: str,
    agent_config_loc: str,
    env_seed: int,
    agent_seed: int,
    solver_seed: int,
    hindsight: bool,
    epoch_tlim: float,
    solve_tlim: float,
    **kwargs,
):
    path = Path(loc)
    static_instance = utils.read(path)
    env = EnvironmentCompetition(
        env_seed, static_instance, epoch_tlim, sample_epoch_requests
    )

    with open(agent_config_loc, "rb") as fh:
        config = tomli.load(fh)
        agent_params = config.get("agent_params", {})
        agent = AGENTS[config["agent"]](agent_seed, **agent_params)

    start = perf_counter()

    if hindsight:
        costs, _ = solve_hindsight(env, solver_seed, solve_tlim)
    else:
        costs, _ = solve_dynamic(env, agent, solver_seed, solve_tlim)

    return (
        path.stem,
        env_seed,
        sum(costs),
        round(perf_counter() - start, 2),
    )


def solve_dynamic(env, agent: Agent, solver_seed: int, solve_tlim: float):
    """
    Solves the dynamic problem.

    Parameters
    ----------
    env: Environment
        Environment of the dynamic problem.
    agent: Agent
        Agent that selects the dispatch action.
    solver_seed: int
        RNG seed for the dispatch instance solver.
    solve_tlim: float
        Time limit for the dispatch instance solver.
    """
    done = False
    solutions = []
    costs = []
    observation, static_info = env.reset()

    while not done:
        dispatch_action = agent.act(observation, static_info)

        epoch_instance = observation["epoch_instance"]
        dispatch_instance = filter_instance(epoch_instance, dispatch_action)

        if dispatch_instance["request_idx"].size <= 2:
            # Empty or single client dispatch instance. PyVRP cannot handle
            # this, so we manually build such a solution.
            ep_sol = [[req] for req in dispatch_instance["request_idx"] if req]
        else:
            res = default_solver(dispatch_instance, solver_seed, solve_tlim)
            routes = [route.visits() for route in res.best.get_routes()]

            # Map solution client indices to request indices.
            ep_sol = [dispatch_instance["request_idx"][rte] for rte in routes]

        observation, reward, done, info = env.step(ep_sol)
        assert info["error"] is None, info["error"]

        solutions.append(ep_sol)
        costs.append(abs(reward))

    return costs, solutions


def solve_hindsight(env, solver_seed: int, solve_tlim: float):
    """
    Solves the dynamic problem in hindsight.

    Parameters
    ----------
    env: Environment
        Environment of the dynamic problem.
    solver_seed: int
        RNG seed used to solve the dispatch instances.
    solve_tlim: float
        Time limit for solving the hindsight instance.

    """
    observation, info = env.reset()
    hindsight_inst = env.get_hindsight_problem()

    res = default_solver(hindsight_inst, solver_seed, solve_tlim)
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
