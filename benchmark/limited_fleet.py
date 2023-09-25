from pathlib import Path
from time import perf_counter

from ddwp.agents import GreedyDispatch
from ddwp.read import read
from ddwp.sampling import SAMPLING_METHODS

from .base import (
    base_parser,
    benchmark,
    configure_agent,
    solve_dynamic,
    solve_hindsight,
)
from .paper import configure_environment


def make_parser():
    parser = base_parser()

    parser.add_argument(
        "--instance_format",
        type=str,
        choices=["vrplib", "solomon"],
        default="solomon",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=SAMPLING_METHODS.keys(),
        required=True,
    )
    parser.add_argument(
        "--num_requests_per_epoch", type=int, nargs="+", default=[75] * 8
    )
    parser.add_argument("--pct_vehicles", type=float, default=1)
    parser.add_argument(
        "--secondary_fleet_fixed_cost", type=int, default=14400
    )

    return parser


def solve(
    loc: str,
    instance_format: str,
    env_seed: int,
    sampling_method: str,
    num_requests_per_epoch: list[int],
    pct_vehicles: float,
    secondary_fleet_fixed_cost: int,
    agent_config_loc: str,
    agent_seed: int,
    hindsight: bool,
    epoch_tlim: float,
    strategy_tlim: float,
    sol_dir: str,
    **kwargs,
):
    if strategy_tlim > epoch_tlim:
        raise ValueError("Strategy time limit >= epoch time limit.")

    path = Path(loc)
    static_instance = read(path, instance_format)
    sampler = SAMPLING_METHODS[sampling_method]

    # First use a greedy agent to determine the number of available vehicles
    # assuming that the number of vehicles is unlimited.
    greedy = GreedyDispatch(agent_seed)
    env_unlimited = configure_environment(
        env_seed, static_instance, epoch_tlim, sampler, num_requests_per_epoch
    )

    _, greedy_sol = solve_dynamic(env_unlimited, greedy)
    num_routes_per_epoch = [len(route) for route in greedy_sol.values()]
    num_vehicles_per_epoch = [
        max(1, int(pct_vehicles * num)) for num in num_routes_per_epoch
    ]

    # Run environment again with restricted number of available vehicles.
    agent = configure_agent(
        agent_config_loc, agent_seed, sampler, epoch_tlim, strategy_tlim
    )
    env = configure_environment(
        env_seed,
        static_instance,
        epoch_tlim,
        sampler,
        num_requests_per_epoch,
        num_vehicles_per_epoch=num_vehicles_per_epoch,
        secondary_fleet_fixed_cost=secondary_fleet_fixed_cost,
    )

    start = perf_counter()

    if hindsight:
        costs, routes = solve_hindsight(env, agent_seed, epoch_tlim)
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


def main():
    benchmark(solve, **vars(make_parser().parse_args()))


if __name__ == "__main__":
    main()
