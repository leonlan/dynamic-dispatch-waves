from pyvrp import Model, Result
from pyvrp.stop import MaxRuntime

from .instance2data import instance2data


def default_solver(instance: dict, seed: int, time_limit: float) -> Result:
    # TODO customize the solver
    data = instance2data(instance)
    model = Model.from_data(data)
    return model.solve(MaxRuntime(time_limit), seed=seed)
