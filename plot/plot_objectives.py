import numpy as np
from .x_axis import x_axis

_N_POINTS = 100


def plot_objectives(ax, stats, step=None, plot_runtimes=False):
    if step is None:
        step = max(1, stats.num_iters() // _N_POINTS)

    x_vals, x_label = x_axis(stats, step, plot_runtimes)

    global_best = np.minimum.accumulate(stats.feas_best_cost())
    ax.plot(x_vals, global_best[::step], label="Global best", c="tab:blue")

    ax.plot(
        x_vals,
        stats.feas_best_cost()[::step],
        label="Feas best",
        c="tab:green",
    )
    ax.plot(
        x_vals,
        stats.feas_avg_cost()[::step],
        label="Feas avg.",
        c="tab:green",
        alpha=0.3,
        linestyle="dashed",
    )
    ax.plot(
        x_vals,
        stats.infeas_best_cost()[::step],
        label="Infeas best",
        c="tab:red",
    )
    ax.plot(
        x_vals,
        stats.infeas_avg_cost()[::step],
        label="Infeas avg.",
        c="tab:red",
        alpha=0.3,
        linestyle="dashed",
    )

    ax.set_title("Population objectives")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective")

    # Use global best objectives to set reasonable y-limits
    best = min(global_best)
    ax.set_ylim(best * 0.995, best * 1.03)
    ax.legend(frameon=False)
