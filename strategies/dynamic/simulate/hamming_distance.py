import numpy as np


def hamming_distance(solutions, to_dispatch, to_postpone, **kwargs):
    """
    Selects the solution with the smallest hamming distance w.r.t. the other
    solutions. This request on this solution are marked dispatch.
    """
    n_simulations = len(solutions)
    ep_size = to_dispatch.size
    dispatch_actions = np.zeros((n_simulations, ep_size), dtype=int)

    for sim_idx, sol in enumerate(solutions):
        for route in sol:
            if any(to_dispatch[idx] for idx in route if idx < ep_size):
                dispatch_actions[sim_idx, route] += 1

    # Mean absolute error a.k.a. Hamming distance
    mae = (abs(dispatch_actions - dispatch_actions.mean(axis=0))).mean(axis=1)
    to_dispatch = dispatch_actions[mae.argsort()[0]].astype(bool)

    return to_dispatch, to_postpone


# TODO
# How can we postpone things? If it is not dispatched, then it is postponed.
# So postpone_actions = dispatch_actions?


# TODO this is pretty good stuff. The results are as good as the compettion results.
# I'm not quite sure yet how to improve this, i.e., whether we must postpone more or dispatch more.
# But there's definitely an interesting idea here, without thresholds!
#
# benchmark_dynamic.py --instance_seed 0 --solver_seed 1 --num_procs 8 --hindsight False --config_loc configs/benchmark_dynamic.toml --instance_pattern instances/ortec/ORTEC-VRPTW-ASYM-0*.txt --epoch_tlim 60.0
# dynamic config:
# {'node_ops': ['Exchange10', 'Exchange11', 'Exchange20', 'MoveTwoClientsReversed', 'Exchange21', 'Exchange22', 'TwoOpt'], 'route_ops': ['RelocateStar', 'SwapStar'], 'crossover_ops': ['selective_route_exchange'], 'strategy': 'simulate', 'params': {'nbGranular': 30, 'weightWaitTime': 2, 'weightTimeWarp': 5, 'postProcessPathLength': 1, 'initialTimeWarpPenalty': 1, 'nbPenaltyManagement': 100, 'feasBooster': 2.0, 'penaltyIncrease': 1.2, 'penaltyDecrease': 0.85, 'targetFeasible': 0.4, 'repairProbability': 50, 'repairBooster': 10, 'minPopSize': 38, 'generationSize': 32, 'nbElite': 10, 'lbDiversity': 0.14015151515151514, 'ubDiversity': 0.36742424242424243, 'nbClose': 17, 'nbIter': 6318}, 'strategy_params': {'simulate_tlim_factor': 0.7703517587939699, 'n_cycles': 5, 'n_simulations': 20, 'n_lookahead': 2, 'n_requests': 100, 'postpone_thresholds': [0.95, 0.85, 0.75], 'dispatch_thresholds': [0.45, 0.35, 0.25], 'node_ops': ['Exchange10', 'Exchange11', 'TwoOpt'], 'route_ops': [], 'crossover_ops': ['selective_route_exchange'], 'sim_config': {'seed': 1, 'initialTimeWarpPenalty': 14, 'nbPenaltyManagement': 1, 'feasBooster': 9.045454545454545, 'penaltyIncrease': 2.0303030303030303, 'penaltyDecrease': 0.33712121212121215, 'targetFeasible': 0.18686868686868688, 'repairProbability': 0, 'repairBooster': 10, 'minPopSize': 3, 'generationSize': 8, 'shouldIntensify': 0, 'nbGranular': 16, 'weightWaitTime': 5, 'weightTimeWarp': 18}}}

# Instance                               Seed  Total   Costs                                                 Routes                      Requests                           Time (s)
# -------------------------------------  ----  ------  ----------------------------------------------------  --------------------------  ---------------------------------  --------
# ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12     0  262987                (0, 68213, 52161, 55191, 46925, 40497)       (0, 11, 10, 10, 8, 8)         (0, 106, 115, 104, 57, 42)    343.29
# ORTEC-VRPTW-ASYM-0dc59ef2-d1-n213-k25     0  301341   (0, 25657, 73524, 51126, 40228, 45809, 55249, 9748)  (0, 4, 12, 9, 7, 8, 11, 2)    (0, 58, 170, 99, 73, 68, 46, 5)    437.68
# ORTEC-VRPTW-ASYM-01829532-d1-n324-k22     0  601275         (11871, 205163, 33979, 206474, 43300, 100488)       (1, 24, 4, 23, 5, 11)          (3, 169, 26, 147, 22, 43)     365.8
# ORTEC-VRPTW-ASYM-0797afaf-d1-n313-k20     0  277742                (0, 55507, 76368, 64615, 39161, 42091)         (0, 6, 10, 8, 7, 8)          (0, 89, 131, 104, 86, 63)    343.77
# ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35     0  406737           (13593, 76604, 84856, 73102, 48607, 109975)      (1, 10, 12, 10, 7, 15)         (2, 100, 99, 109, 57, 119)    365.95
# ORTEC-VRPTW-ASYM-02182cf8-d1-n327-k20     0  393860               (0, 119415, 78800, 66227, 55685, 73733)         (0, 15, 9, 7, 6, 9)           (0, 159, 87, 84, 62, 56)    343.46
# ORTEC-VRPTW-ASYM-04c694cd-d1-n254-k18     0  324359  (0, 27214, 61492, 50707, 60434, 50476, 44888, 29148)  (0, 4, 10, 8, 11, 9, 9, 8)  (0, 60, 166, 96, 113, 60, 38, 12)    486.77
# ORTEC-VRPTW-ASYM-08d8e660-d1-n460-k42     0  318485                (0, 72050, 46910, 66190, 75186, 58149)         (0, 8, 6, 8, 10, 8)          (0, 119, 91, 106, 98, 55)    344.03

#       Avg. objective: 360848
#    Avg. run-time (s): 378.84
