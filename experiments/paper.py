from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from ddwp.Environment import Environment
from ddwp.read import read
from ddwp.sampling import SAMPLING_METHODS, SamplingMethod
from ddwp.VrpInstance import VrpInstance

from .base import (
    base_parser,
    benchmark,
    configure_agent,
    solve_dynamic,
    solve_hindsight,
)


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

    return parser


def configure_environment(
    seed: int,
    instance: VrpInstance,
    epoch_tlim: float,
    sampling_method: SamplingMethod,
    num_requests_per_epoch: list[int],
    num_vehicles_per_epoch: Optional[list[int]] = None,
    secondary_fleet_fixed_cost: int = 0,
):
    """
    Creates a DDWP environment identical to the one used in paper [1].

    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled. Note that
        the time windows are ignored in this environment.
    epoch_tlim
        The epoch time limit.
    sampling_method
        The sampling method to use.
    num_requests_per_epoch
        The expected number of revealed requests per epoch.
    num_vehicles_per_epoch
        The available number of primary vehicles per epoch. If None, then
        there is no limit on the number of primary vehicles.
    secondary_fleet_fixed_cost
        The fixed cost of the secondary fleet.

    References
    ----------
    [1] Lan, L., van Doorn, J., Wouda, N. A., Rijal, A., & Bhulai, S. (2023).
        An iterative conditional dispatch algorithm for the dynamic dispatch
        waves problem.
    """
    num_epochs = len(num_requests_per_epoch)

    # Assume an epoch duration of one hour (in seconds) and a horizon of
    # ``num_epochs`` hours.
    epoch_duration = 600
    horizon = num_epochs * epoch_duration
    start_epoch = 0
    end_epoch = num_epochs - 1

    # Custom depot time windows. Instance time windows are not used!
    time_windows = instance.time_windows.copy()
    time_windows[0, :] = [0, horizon]

    # Normalize the distances so that the furthest customer can be reached
    # in half hour from the depot. Service times are also scaled accordingly.
    scale = instance.duration_matrix.max() / epoch_duration
    dur_mat = np.ceil(instance.duration_matrix / scale).astype(int)
    service_times = np.ceil(instance.service_times / scale).astype(int)

    new_instance = instance.replace(
        time_windows=time_windows,
        duration_matrix=dur_mat,
        service_times=service_times,
    )

    return Environment(
        seed=seed,
        instance=new_instance,
        epoch_tlim=epoch_tlim,
        sampling_method=sampling_method,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        num_requests_per_epoch=num_requests_per_epoch,
        num_vehicles_per_epoch=num_vehicles_per_epoch,
        secondary_fleet_fixed_cost=secondary_fleet_fixed_cost,
        epoch_duration=epoch_duration,
        dispatch_margin=0,
    )


def solve(
    loc: str,
    instance_format: str,
    env_seed: int,
    sampling_method: str,
    num_requests_per_epoch: list[int],
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
    env = configure_environment(
        env_seed, static_instance, epoch_tlim, sampler, num_requests_per_epoch
    )
    agent = configure_agent(
        agent_config_loc, agent_seed, sampler, epoch_tlim, strategy_tlim
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
