from collections import defaultdict


class Statistics:
    def __init__(self, config, collect=True):
        self.collect = collect
        self.config = config
        self.scenario_solutions = defaultdict(list)
        self.dispatch = {}
        self.epoch = {}

    def collect_static(self, static_info):
        if self.collect:
            self.static_info = static_info

    def collect_iteration_action(self, epoch, solutions):
        if self.collect:
            self.scenario_solutions[epoch].append(solutions)

    def collect_epoch(self, epoch, instance, solution):
        if self.collect:
            self.epoch[epoch] = (instance, solution)
