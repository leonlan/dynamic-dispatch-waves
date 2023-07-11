from .Agent import Agent
from .RandomDispatch import GreedyDispatch, LazyDispatch, UniformDispatch

AGENTS = {
    "greedy": GreedyDispatch,
    "lazy": LazyDispatch,
    "uniform": UniformDispatch,
}
