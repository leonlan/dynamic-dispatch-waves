from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("dark_background") # nice for dark presentation


def plot_dynamic_instance(
    ax,
    title,
    instance,
    routes=(),
    postponed=(),
    dispatched=(),
    labels: Optional[dict[int, float]] = None,
    description=None,
):
    """
    Plot a dynamic instance and optionally a solution. This plot contains the
    depot location (blue star) and request locations.
    - Must-dispatch requests are colored red.
    - Undecided requests (i.e., non-must-dispatch) are colored yellow.
    - Sampled requests are colored grey and smaller.
    - Postponed requests are colored green.

    A given list of routes can also be plotted, if provided.
    - Routes with must-dispatch requests are colored red.
    - Routes without must-dispatch requests are colored grey.

    This code is specifically tailored towards the ORTEC-01 instance.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axes to plot on.
    title: str
        The title of the plot.
    instance: dict
        The instance to plot.
    routes: list of list of int, optional
        The routes to plot.
    postponed: list of int, optional
        The postponed requests to plot.
    dispatched: list of int, optional
        The dispatched requests to plot.
    labels: dict of int to str, optional
        The labels to plot for each request.
    description: str, optional
        The description of the instance, placed underneath the title.
    """
    # Scale the coordinates to let depot be in the middle
    coordinates = instance["coords"].copy()
    coordinates[:, 0] += 49
    coordinates[:, 1] += 85

    # Depot
    kwargs = {"marker": "*", "zorder": 5, "s": 500, "edgecolors": "black"}
    depot_coords = coordinates[0].T
    ax.scatter(*depot_coords, c="tab:blue", label="depot", **kwargs)

    # Must dispatch
    kwargs = {"marker": ".", "zorder": 3, "s": 300, "edgecolors": "black"}

    must_dispatch = instance["must_dispatch"]
    coords = coordinates[must_dispatch].T
    ax.scatter(*coords, c="tab:red", label="must-dispatch", **kwargs)

    # Undecided
    undecided = ~instance["must_dispatch"] & (instance["request_idx"] > 0)
    coords = coordinates[undecided].T
    ax.scatter(*coords, c="white", label="undecided", **kwargs)

    # Dispatched
    dispatched_idcs = np.flatnonzero(dispatched)
    coords = coordinates[dispatched_idcs].T
    ax.scatter(*coords, c="tab:red", label="dispatched", **kwargs)

    # Postponed
    postponed_idcs = np.flatnonzero(postponed)
    coords = coordinates[postponed_idcs].T
    ax.scatter(*coords, c="tab:green", label="postponed", **kwargs)

    # Sampled
    sampled = instance["request_idx"] < 0
    coords = coordinates[sampled].T

    if coords.any():
        # Make requests with late release times smaller
        release_times = instance["release_times"][sampled]
        delta = release_times.max() - release_times.min()
        scale = (release_times.max() - release_times) / delta  # between [0, 1]
    else:
        scale = 1

    ax.scatter(
        *coords,
        marker=".",
        c="silver",
        label="sampled",
        s=50 + 100 * scale,
        edgecolors="black",
        zorder=3,
    )

    # Labels
    if labels is not None:
        for idx, label in labels.items():
            if idx > 0 and ~instance["must_dispatch"][idx]:
                margin = 50
                x, y = coordinates[idx]
                ax.annotate(label, (x + margin, y + margin), fontsize=10)

    # Plot routes
    for route in routes:
        coords = coordinates[route].T
        ax.plot(*coords, alpha=0.5, c="tab:grey", label="route")

        # Edges from and to the depot, very thinly dashed.
        coords = coordinates[route]  # no transpose
        depot = coordinates[0]
        kwargs = {"ls": (0, (5, 15)), "linewidth": 0.25, "color": "grey"}

        ax.plot([depot[0], coords[0][0]], [depot[1], coords[0][1]], **kwargs)
        ax.plot([coords[-1][0], depot[0]], [coords[-1][1], depot[1]], **kwargs)

    ax.grid(color="grey", linestyle="--", linewidth=0.25)

    ax.set_title(description)  # REVIEW Try this out

    ax.set_xlim(-500, 8500)
    ax.set_ylim(0, 4000)

    # Prevent duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # type: ignore
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
    path,
    title,
    instance,
    routes=(),
    postponed=(),
    dispatched=(),
    labels: Optional[dict[int, float]] = None,
    description="",
):
    fig, ax = plt.subplots(figsize=[8, 6], dpi=150)
    plot_dynamic_instance(
        ax,
        title,
        instance,
        routes=routes,
        labels=labels,
        postponed=postponed,
        dispatched=dispatched,
        description=description,
    )

    fig.savefig(path)
    plt.close()
