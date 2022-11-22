import argparse
import cProfile
import pstats
from datetime import datetime

import tools
from environment import VRPEnvironment
from strategies import solve_dynamic, solve_hindsight
from strategies.config import Config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance")
    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--solver_seed", type=int, default=1)
    parser.add_argument("--epoch_tlim", type=int, default=120)
    parser.add_argument("--config_loc", default="configs/solver.toml")
    parser.add_argument("--profile", action="store_true")

    problem_type = parser.add_mutually_exclusive_group()
    problem_type.add_argument("--static", action="store_true")
    problem_type.add_argument("--hindsight", action="store_true")

    return parser.parse_args()


def run(args):
    env = VRPEnvironment(
        seed=args.instance_seed,
        instance=tools.read_vrplib(args.instance),
        epoch_tlim=args.epoch_tlim,
        is_static=args.static,
    )

    config = Config.from_file(args.config_loc)

    if args.hindsight:
        solve_hindsight(env, config.static(), args.solver_seed)
    else:
        solve_dynamic(env, config, args.solver_seed)


def main():
    args = parse_args()

    if args.profile:
        with cProfile.Profile() as profiler:
            run(args)

        stats = pstats.Stats(profiler).strip_dirs().sort_stats("time")
        stats.print_stats()

        now = datetime.now().isoformat()
        stats.dump_stats(f"tmp/profile-{now}.pstat")
    else:
        run(args)


if __name__ == "__main__":
    main()
