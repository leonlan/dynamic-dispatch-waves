import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import tomli_w
from scipy.stats import qmc


@dataclass
class Integer:
    interval: tuple[int, int]
    default: int

    def ppf(self, q: float) -> int:
        lo, hi = self.interval
        return round(lo + q * (hi - lo))


@dataclass
class Float:
    interval: tuple[float, float]
    default: float

    def ppf(self, q: float) -> float:
        lo, hi = self.interval
        return lo + q * (hi - lo)


@dataclass
class Bool:
    default: bool

    def ppf(self, q: float) -> bool:
        return bool(q > 0.5)


# These parameter groups, ranges, and default values have been discussed in
# https://github.com/N-Wouda/Euro-NeurIPS-2022/issues/33.
PARAM_SPACE = {
    "node_ops": {
        "Exchange10": Bool(True),
        "Exchange11": Bool(True),
        "Exchange20": Bool(True),
        "MoveTwoClientsReversed": Bool(True),
        "Exchange21": Bool(True),
        "Exchange22": Bool(True),
        "Exchange30": Bool(True),
        "Exchange31": Bool(True),
        "Exchange32": Bool(True),
        "Exchange33": Bool(True),
        "TwoOpt": Bool(True),
    },
    "route_ops": {
        "RelocateStar": Bool(True),
        "SwapStar": Bool(True),
    },
    "genetic": {
        "repair_probability": Float((0.8, 0.8), 0.80),
        "nb_iter_no_improvement": Integer((20_000, 20_000), 20_000),
    },
    "population": {
        "min_pop_size": Integer((1, 100), 25),
        "generation_size": Integer((1, 100), 40),
        "nb_elite": Integer((0, 2), 4),
        "nb_close": Integer((1, 2), 5),
        "lb_diversity": Float((0, 0.25), 0.1),
        "ub_diversity": Float((0.25, 1), 0.5),
    },
    "neighbourhood": {
        "weight_wait_time": Float((0, 100), 0.2),
        "weight_time_warp": Float((0, 100), 1),
        "nb_granular": Integer((10, 100), 40),
        "symmetric_proximity": Bool(True),
        "symmetric_neighbours": Bool(False),
    },
    "penalty": {
        "init_capacity_penalty": Integer((1, 25), 20),
        "init_time_warp_penalty": Integer((1, 25), 6),
        "repair_booster": Integer((1, 25), 12),
        "num_registrations_between_penalty_updates": Integer((25, 500), 50),
        "penalty_increase": Float((1, 5), 1.34),
        "penalty_decrease": Float((0.25, 1), 0.32),
        "target_feasible": Float((0, 1), 0.43),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("param_space", nargs="+", choices=PARAM_SPACE.keys())
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default="data/tune")

    return parser.parse_args()


def write(where: Path, params, exp: int):
    where.mkdir(parents=True, exist_ok=True)

    with open(where / f"{exp}.toml", "wb") as fh:
        tomli_w.dump(params, fh)


def main():
    args = parse_args()

    # Create the default configuration.
    default = {
        group: {name: val.default for name, val in params.items()}
        for group, params in PARAM_SPACE.items()
    }
    write(args.out_dir, default, 1)

    # Create the test configurations.
    groups = {group: PARAM_SPACE[group] for group in args.param_space}
    num_params = sum(len(group) for group in groups)

    sampler = qmc.LatinHypercube(d=num_params, centered=True, seed=args.seed)
    samples = sampler.random(args.num_samples - 1)

    for exp, sample in enumerate(samples, 2):
        config = deepcopy(default)

        idx = 0
        for group, params in groups.items():
            for name, val in params.items():
                config[group][name] = val.ppf(sample[idx])
                idx += 1

        write(args.out_dir, config, exp)


if __name__ == "__main__":
    main()
