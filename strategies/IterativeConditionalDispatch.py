import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime

from utils import instance2data

from .Agent import Agent
from .simulate_instance import simulate_instance


class IterativeConditionalDispatch(Agent):
    def __init__(
        self,
        seed: int,
        consensus,
        num_iterations: int,
        num_lookahead: int,
        num_scenarios: int,
        strategy_tlim_factor: float = 0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.consensus = consensus
        self.num_iterations = num_iterations
        self.num_lookahead = num_lookahead
        self.num_scenarios = num_scenarios
        self.strategy_tlim_factor = strategy_tlim_factor

    def act(self, observation, info) -> np.ndarray:
        ep_inst = observation["epoch_instance"]
        ep_size = ep_inst["is_depot"].size  # includes depot

        total_sim_tlim = self.strategy_tlim_factor * info["epoch_tlim"]
        single_sim_tlim = total_sim_tlim / (
            self.num_iterations * self.num_scenarios
        )

        to_dispatch = ep_inst["must_dispatch"].copy()
        to_postpone = np.zeros(ep_size, dtype=bool)

        # Dispatch everything in the last iteration
        if observation["current_epoch"] == info["end_epoch"]:
            return np.ones(ep_size, dtype=bool)

        for iter_idx in range(self.num_iterations):
            scenarios = []

            for _ in range(self.num_scenarios):
                sim_inst = simulate_instance(
                    info,
                    observation,
                    self.rng,
                    self.num_lookahead,
                    to_dispatch,
                    to_postpone,
                )

                # TODO make this as a custom solver
                model = Model.from_data(instance2data(sim_inst))
                res = model.solve(MaxRuntime(single_sim_tlim), seed=42)
                sim_sol = [route.visits() for route in res.best.get_routes()]

                scenarios.append((sim_inst, sim_sol))

            to_dispatch, to_postpone = self.consensus(
                iter_idx, scenarios, to_dispatch, to_postpone
            )

            # Stop the run early when all requests have been marked
            if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
                break

        return to_dispatch | ep_inst["is_depot"]  # include depot
