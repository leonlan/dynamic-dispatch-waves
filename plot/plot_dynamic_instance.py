def plot_dynamic_instance(ax, instance, routes=()):
    """
    Plot a dynamic instance and optionally a solution. This plot contains the
    depot location (blue star) and request locations.
    - Must-dispatch requests are colored red.
    - Optional requests (i.e., non-must-dispatch) are colored yellow.
    - Simulated requests are colored grey and smaller.
    - Postponed requests are colored green.

    A given list of routes can also be plotted, if provided.
    - Routes with must-dispatch requests are colored red.
    - Routes without must-dispatch requests are colored green/grey.
    """
    # Depot
    kwargs = dict(marker="*", zorder=3, s=500)
    depot_coords = instance["coords"][0].T
    ax.scatter(*depot_coords, c="tab:blue", label="Depot", **kwargs)

    # Must dispatch
    must_dispatch = instance["is_must_dispatch"]
    coords = instance["coords"][must_dispatch].T

    kwargs = dict(s=(0.0008) ** 2, alpha=0.1, zorder=3)
    ax.scatter(*coords, c="tab:red", label="must-dispatch", **kwargs)

    # Optional
    optional = instance["is_optional"]
    coords = instance["coords"][optional].T

    kwargs = dict(s=(0.0008) ** 2, alpha=0.1, zorder=3)
    ax.scatter(*coords, c="tab:yellow", label="optional", **kwargs)

    # Simulated
    simulated = instance["is_simulated"]
    coords = instance["coords"][simulated].T

    kwargs = dict(s=(0.0008) ** 2, alpha=0.1, zorder=3)
    ax.scatter(*coords, c="tab:grey", label="simulated", **kwargs)

    for route in routes:
        if instance["is_must_dispatch"][route].any():
            ax.plot(*instance["coords"][[0] + route + [0]].T, c="tab:red")
        else:
            ax.plot(*instance["coords"][[0] + route + [0]].T, c="tab:grey")

    ax.grid(color="grey", linestyle="--", linewidth=0.25)

    ax.set_title("Solution" if routes else "Instance")
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, ncol=3)
