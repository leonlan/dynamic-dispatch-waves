from pathlib import Path
from time import perf_counter
from typing import Optional

from ddwp.Environment import Environment
from ddwp.read import read
from ddwp.sampling import SamplingMethod
from ddwp.sampling import euro_neurips as euro_neurips_sampling_method
from ddwp.VrpInstance import VrpInstance

from .base import (
    base_parser,
    benchmark,
    configure_agent,
    solve_dynamic,
    solve_hindsight,
)


def configure_environment(
    seed: int,
    instance: VrpInstance,
    epoch_tlim: float,
    sampling_method: SamplingMethod,
    num_requests: int = 100,
    epoch_duration: int = 3600,
    dispatch_margin: int = 3600,
    num_vehicles_per_epoch: Optional[list[int]] = None,
):
    """
    Creates a DDWP environment identical to the one used in [1].

    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled.
    epoch_tlim
        The epoch time limit.
    sampling_method
        The sampling method to use.
    num_requests
        The expected number of revealed requests per epoch.
    epoch_duration
        The time between two consecutive epochs.
    dispatch_margin
        The preparation time needed to dispatch a set of routes. That is, when
        a set of routes are to be dispatched at epoch t, then the start time of
        the routes is `t * epoch_duration + dispatch_margin`.
    num_vehicles_per_epoch
        The available number of primary vehicles per epoch. If None, then
        there is no limit on the number of primary vehicles.

    References
    ----------
    [1] EURO meets NeurIPS 2022 vehicle routing competition.
        https://euro-neurips-vrp-2022.challenges.ortec.com/
    """
    tw = instance.time_windows
    earliest = tw[1:, 0].min() - dispatch_margin
    latest = tw[1:, 0].max() - dispatch_margin

    # The start and end epochs are determined by the earliest and latest
    # opening client time windows, corrected by the dispatch margin.
    start_epoch = int(max(earliest // epoch_duration, 0))
    end_epoch = int(max(latest // epoch_duration, 0))

    num_requests_per_epoch = [num_requests] * (end_epoch + 1)

    return Environment(
        seed=seed,
        instance=instance,
        epoch_tlim=epoch_tlim,
        sampling_method=sampling_method,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        num_requests_per_epoch=num_requests_per_epoch,
        num_vehicles_per_epoch=num_vehicles_per_epoch,
        secondary_fleet_fixed_cost=0,
        epoch_duration=epoch_duration,
        dispatch_margin=dispatch_margin,
    )


def solve(
    loc: str,
    env_seed: int,
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
    static_instance = read(path, "vrplib")

    env = configure_environment(
        env_seed, static_instance, epoch_tlim, euro_neurips_sampling_method
    )
    agent = configure_agent(
        agent_config_loc,
        agent_seed,
        euro_neurips_sampling_method,
        epoch_tlim,
        strategy_tlim,
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
    benchmark(solve, **vars(base_parser().parse_args()))


if __name__ == "__main__":
    main()
