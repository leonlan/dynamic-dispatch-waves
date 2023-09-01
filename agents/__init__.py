from .Agent import Agent
from .IterativeConditionalDispatch import IterativeConditionalDispatch
from .RandomDispatch import GreedyDispatch, LazyDispatch, UniformDispatch
from .RollingHorizon import RollingHorizon

AGENTS = {
    "greedy": GreedyDispatch,
    "lazy": LazyDispatch,
    "uniform": UniformDispatch,
    "icd": IterativeConditionalDispatch,
    "rolling_horizon": RollingHorizon,
}
