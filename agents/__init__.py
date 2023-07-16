from .Agent import Agent
from .IterativeConditionalDispatch import IterativeConditionalDispatch
from .RandomDispatch import GreedyDispatch, LazyDispatch, UniformDispatch

AGENTS = {
    "greedy": GreedyDispatch,
    "lazy": LazyDispatch,
    "uniform": UniformDispatch,
    "icd": IterativeConditionalDispatch
}
