from .x_axis import x_axis

_N_POINTS = 100


def plot_population(ax, stats, step=None, plot_runtimes=False):
    if step is None:
        step = max(1, stats.num_iters() // _N_POINTS)

    x_vals, x_label = x_axis(stats, step, plot_runtimes)

    ax.set_title("Population diversity")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Avg. diversity")

    ax.plot(
        x_vals,
        stats.feas_avg_diversity()[::step],
        label="Feas. diversity",
        c="tab:green",
    )
    ax.plot(
        x_vals,
        stats.infeas_avg_diversity()[::step],
        label="Infeas. diversity",
        c="tab:red",
    )

    ax.legend(frameon=False)
