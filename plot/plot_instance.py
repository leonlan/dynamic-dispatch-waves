def plot_instance(ax, instance, routes=()):
    """
    Plot an instance and optionally a solution. This plot contains the depot
    location (yellow star) and customer locations. A client is represented by a
    blue dot, with a size relative to when its time window opens. Around this
    dot, the relative size of the blue circle represents when a time windows
    closes.

    A given list of routes can also be plotted, if provided.
    """
    is_client = ~instance["is_depot"]
    coords = instance["coords"][is_client].T
    tws_open = instance["time_windows"][is_client, 0]
    tws_close = instance["time_windows"][is_client, 1]
    depot_coords = instance["coords"][~is_client].T

    kwargs = dict(s=(0.0003 * tws_open) ** 2, zorder=3)
    ax.scatter(*coords, c="tab:blue", label="TW open", **kwargs)

    kwargs = dict(s=(0.0008 * tws_close) ** 2, alpha=0.1, zorder=3)
    ax.scatter(*coords, c="tab:blue", label="TW close", **kwargs)

    kwargs = dict(marker="*", zorder=3, s=750)
    ax.scatter(*depot_coords, c="tab:red", label="Depot", **kwargs)

    for route in routes:
        ax.plot(*instance["coords"][[0] + route + [0]].T)

    ax.grid(color="grey", linestyle="--", linewidth=0.25)

    ax.set_title("Solution" if routes else "Instance")
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, ncol=3)
