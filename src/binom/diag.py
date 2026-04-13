import matplotlib.pyplot as plt
import numpy as np

"""
This module contains functions that allow the user to visualize/diagnose BINOM results
"""

def plot_solar_angles(df, title=None):

    """
    Plot solar zenith and azimuth from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from get_solar_position_df()
    """

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Zenith
    ax1.plot(df.index, df["apparent_zenith"], label="Zenith (deg)")
    ax1.set_ylabel("Zenith (deg)")
    ax1.set_xlabel("Time")

    # Azimuth (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["azimuth"], linestyle="--", label="Azimuth (deg)")
    ax2.set_ylabel("Azimuth (deg)")

    # Title
    if title is None:
        title = "Solar Zenith and Azimuth"
    ax1.set_title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()

def plot_canopy_fapar(df, title="Canopy fAPAR over time"):
    """
    Plot canopy fAPAR over time from dataframe containing both
    BINOM inputs and outputs.
    """
    required_cols = ["Rc_PARdir", "Rc_PARdiff", "Srad_dir", "Srad_diff", "fvis"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_df = df.copy()

    absorbed_canopy = plot_df["Rc_PARdir"] + plot_df["Rc_PARdiff"]
    incoming_par = (plot_df["Srad_dir"] + plot_df["Srad_diff"]) * plot_df["fvis"]

    plot_df["canopy_fAPAR"] = np.where(
        incoming_par > 0,
        absorbed_canopy / incoming_par,
        np.nan
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(plot_df.index, plot_df["canopy_fAPAR"], label="canopy_fAPAR")
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Canopy fAPAR")
    ax.set_title(title)

    ymax = np.nanmax(plot_df["canopy_fAPAR"])
    ax.set_ylim(0, ymax * 1.05)

    ax.legend()
    plt.tight_layout()
    plt.show()

    return plot_df

def plot_params_over_time(df, params, title=None):
    """
    Plot multiple parameters over time, each with its own y-axis and color.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe indexed by time
    params : str or list of str
        Column name or list of column names to plot
    title : str or None
        Plot title
    """
    if isinstance(params, str):
        params = [params]

    missing = [p for p in params if p not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Use matplotlib default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    lines = []

    # First parameter (base axis)
    p0 = params[0]
    color0 = colors[0 % len(colors)]
    line, = ax.plot(df.index, df[p0], label=p0, color=color0)
    ax.set_ylabel(p0, color=color0)
    ax.tick_params(axis='y', colors=color0)
    lines.append(line)

    # Additional parameters
    for i, p in enumerate(params[1:], start=1):
        ax_new = ax.twinx()

        # Offset axis
        ax_new.spines["right"].set_position(("outward", 60 * (i - 1)))

        color = colors[i % len(colors)]

        line, = ax_new.plot(df.index, df[p], label=p, color=color)
        ax_new.set_ylabel(p, color=color)
        ax_new.tick_params(axis='y', colors=color)

        lines.append(line)

    ax.set_xlabel("Time")

    if title is None:
        title = "Parameters over time"
    ax.set_title(title)

    # Combined legend
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()