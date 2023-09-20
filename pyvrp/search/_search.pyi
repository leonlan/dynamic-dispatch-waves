from typing import Iterator, List, Optional

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
    TimeWindowSegment,
)

class NodeOperator:
    def __init__(self, data: ProblemData) -> None: ...
    def evaluate(
        self, U: Node, V: Node, cost_evaluator: CostEvaluator
    ) -> int: ...
    def apply(self, U: Node, V: Node) -> None: ...

class RouteOperator:
    def __init__(self, data: ProblemData) -> None: ...
    def evaluate(
        self, U: Route, V: Route, cost_evaluator: CostEvaluator
    ) -> int: ...
    def apply(self, U: Route, V: Route) -> None: ...

class Exchange10(NodeOperator): ...
class Exchange11(NodeOperator): ...
class Exchange20(NodeOperator): ...
class Exchange21(NodeOperator): ...
class Exchange22(NodeOperator): ...
class Exchange30(NodeOperator): ...
class Exchange31(NodeOperator): ...
class Exchange32(NodeOperator): ...
class Exchange33(NodeOperator): ...
class MoveTwoClientsReversed(NodeOperator): ...
class RelocateStar(RouteOperator): ...
class SwapRoutes(RouteOperator): ...
class SwapStar(RouteOperator): ...
class TwoOpt(NodeOperator): ...

class LocalSearch:
    def __init__(
        self,
        data: ProblemData,
        neighbours: List[List[int]],
    ) -> None: ...
    def add_node_operator(self, op: NodeOperator) -> None: ...
    def add_route_operator(self, op: RouteOperator) -> None: ...
    def set_neighbours(self, neighbours: List[List[int]]) -> None: ...
    def get_neighbours(self) -> List[List[int]]: ...
    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
    ) -> Solution: ...
    def shuffle(self, rng: RandomNumberGenerator) -> None: ...
    def intensify(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        overlap_tolerance: float = 0.05,
    ) -> Solution: ...
    def search(
        self, solution: Solution, cost_evaluator: CostEvaluator
    ) -> Solution: ...

class Route:
    def __init__(
        self, data: ProblemData, idx: int, vehicle_type: int
    ) -> None: ...
    @property
    def idx(self) -> int: ...
    @property
    def vehicle_type(self) -> int: ...
    def __delitem__(self, idx: int) -> None: ...
    def __getitem__(self, idx: int) -> Node: ...
    def __iter__(self) -> Iterator[Node]: ...
    def __len__(self) -> int: ...
    def is_feasible(self) -> bool: ...
    def has_excess_load(self) -> bool: ...
    def has_time_warp(self) -> bool: ...
    def capacity(self) -> int: ...
    def fixed_cost(self) -> int: ...
    def load(self) -> int: ...
    def distance(self) -> int: ...
    def time_warp(self) -> int: ...
    def dist_between(self, start: int, end: int) -> int: ...
    def load_between(self, start: int, end: int) -> int: ...
    def tws(self, idx: int) -> TimeWindowSegment: ...
    def tws_between(self, start: int, end: int) -> TimeWindowSegment: ...
    def tws_before(self, end: int) -> TimeWindowSegment: ...
    def tws_after(self, start: int) -> TimeWindowSegment: ...
    def overlaps_with(self, other: Route, tolerance: float) -> bool: ...
    def append(self, node: Node) -> None: ...
    def clear(self) -> None: ...
    def insert(self, idx: int, node: Node) -> None: ...
    def update(self) -> None: ...

class Node:
    def __init__(self, loc: int) -> None: ...
    @property
    def client(self) -> int: ...
    @property
    def idx(self) -> int: ...
    @property
    def route(self) -> Optional[Route]: ...
    def is_depot(self) -> bool: ...

def insert_cost(
    U: Node, V: Node, data: ProblemData, cost_evaluator: CostEvaluator
) -> int: ...
def remove_cost(
    U: Node, data: ProblemData, cost_evaluator: CostEvaluator
) -> int: ...
