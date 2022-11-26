import numpy as np
import matplotlib.pyplot as plt


plt.style.use("dark_background")


def plot_dynamic_instance(
    ax,
    title,
    instance,
    routes=(),
    labels={},
    postponed=(),
    description="",
):
    """
    Plot a dynamic instance and optionally a solution. This plot contains the
    depot location (blue star) and request locations.
    - Must-dispatch requests are colored red.
    - Optional requests (i.e., non-must-dispatch) are colored yellow.
    - Simulated requests are colored grey and smaller.
    - Postponed requests are colored green.

    A given list of routes can also be plotted, if provided.
    - Routes with must-dispatch requests are colored red.
    - Routes without must-dispatch requests are colored grey.

    This code is specifically tailored towards the ORTEC-01 instance.
    """
    coordinates = instance["coords"].copy()
    coordinates[:, 0] += 49
    coordinates[:, 1] += 85

    # Depot
    kwargs = dict(marker="*", zorder=5, s=500, edgecolors="black")
    depot_coords = coordinates[0].T
    ax.scatter(*depot_coords, c="tab:blue", label="depot", **kwargs)

    # Must dispatch
    kwargs = dict(marker=".", zorder=3, s=300, edgecolors="black")

    must_dispatch = instance["must_dispatch"]
    coords = coordinates[must_dispatch].T
    ax.scatter(*coords, c="tab:red", label="must-dispatch", **kwargs)

    # Optional
    optional = ~instance["must_dispatch"] & (instance["request_idx"] > 0)
    coords = coordinates[optional].T
    ax.scatter(*coords, c="white", label="optional", **kwargs)

    # Postponed
    postponed = np.array(postponed).astype(bool)
    coords = coordinates[postponed].T
    ax.scatter(*coords, c="tab:green", label="postponed", **kwargs)

    # Simulated
    simulated = instance["request_idx"] < 0
    coords = coordinates[simulated].T

    ax.scatter(
        *coords,
        marker=".",
        c="tab:grey",
        label="simulated",
        s=75,
        edgecolors="black"
    )

    # Labels
    for idx, label in labels.items():
        if idx > 0 and ~instance["must_dispatch"][idx]:
            margin = 30
            x, y = coordinates[idx]
            ax.annotate(label, (x + margin, y + margin), fontsize=10)

    # Plot routes
    # Dummy plot to create route labels
    ax.plot(*[[]], alpha=0.75, c="tab:red", label="must-dispatch route")
    ax.plot(*[[]], alpha=0.5, c="tab:grey", label="postponed route")

    for route in routes:
        coords = coordinates[[0] + route + [0]].T

        if instance["must_dispatch"][route].any():
            ax.plot(
                *coords, alpha=0.75, c="tab:red", label="must-dispatch route"
            )
        else:
            ax.plot(*coords, alpha=0.75, c="tab:grey", label="postponed route")

    ax.grid(color="grey", linestyle="--", linewidth=0.25)

    ax.set_title(description)  # REVIEW Try this out

    ax.set_xlim(-2000, 10000)
    ax.set_ylim(0, 4000)

    # ax.set_xlabel(description)

    # Prevent duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        ncol=1,
        loc="upper right",
    )

    # Turn off axis ticks and labels
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


def save_fig(
    path, title, instance, routes=(), labels={}, postponed=(), description=""
):
    fig, ax = plt.subplots(figsize=[10, 7.5], dpi=150)
    plot_dynamic_instance(
        ax,
        title,
        instance,
        routes=routes,
        labels=labels,
        postponed=postponed,
        description=description,
    )

    fig.savefig(path)
    plt.close()
