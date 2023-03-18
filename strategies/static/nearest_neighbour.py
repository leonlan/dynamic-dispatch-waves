import hgspy
import tools


def nearest_neighbour(
    instance,
    config,
    node_ops,
    initial_solution=None,
):
    params = hgspy.Params(config, **tools.io.inst_to_vars(instance))

    rng = hgspy.XorShift128(seed=config.seed)

    if initial_solution is not None:
        indiv = hgspy.Individual(params, initial_solution)
    else:
        indiv = hgspy.Individual(params, rng)

    node_ops = [op(params) for op in node_ops]

    ls = hgspy.LocalSearch(params, rng)

    for op in node_ops:
        ls.add_node_operator(op)

    ls.search(indiv)

    return indiv
