from pyvrp import Model, Result
from pyvrp.stop import MaxRuntime

from .instance2data import instance2data


def default_solver(instance: dict, seed: int, time_limit: float) -> Result:
    """
    Solves the given instance with the default PyVRP solver.

    Parameters
    ----------
    instance: dict
        The instance to solve.
    seed: int
        The seed to use for the solver.
    time_limit: float
        The time limit in seconds.

    Returns
    -------
    Result
        Object storing the solver outcome.
    """
    data = instance2data(instance)
    model = Model.from_data(data)
    return model.solve(MaxRuntime(time_limit), seed=seed)
