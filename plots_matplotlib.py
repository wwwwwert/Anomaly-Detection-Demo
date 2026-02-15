import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def add_line(
    ax: Axes,
    x_values: Iterable,
    y_values: Iterable,
    name: str,
    color: str,
) -> Axes:
    """Add a line plot to the axes."""
    ax.plot(x_values, y_values, color=color, label=name, linewidth=1.5)
    return ax


def update_layout(ax: Axes, fig: Figure):
    """Update the layout with time series specific settings."""
    ax.set_title("Time Series", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Set figure size (equivalent to height=600 in plotly)
    fig.set_size_inches(20, 8)

    # Enable grid for better readability
    ax.grid(True, alpha=0.3)

    # Format x-axis for dates if the index contains datetime
    if hasattr(ax.get_xaxis(), "set_major_formatter"):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout to prevent label cutoff
    fig.tight_layout()

    return ax


def plot_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    """Create a basic time series plot."""
    fig, ax = plt.subplots(figsize=(20, 8))

    # Add the main time series line
    add_line(ax, time_series.index, time_series["value_0"], title, "blue")

    # Update layout
    update_layout(ax, fig)

    return ax


def add_confidence_interval(ax: Axes, forecast: pd.DataFrame):
    """Add forecast line with confidence interval."""
    ds = forecast.index

    # Add forecast line
    ax.plot(
        ds,
        forecast["expected"],
        color=(31 / 255, 119 / 255, 180 / 255, 0.8),
        label="Forecast",
        linewidth=2,
    )

    # Add confidence interval as filled area
    ax.fill_between(
        ds,
        forecast["lower"],
        forecast["upper"],
        color=(31 / 255, 119 / 255, 180 / 255),
        alpha=0.2,
        label="Confidence Interval",
    )

    return ax


def add_points(
    ax: Axes,
    x_values: Iterable,
    y_values: Iterable,
    name: str = "Anomalies",
    color: str = "red",
):
    """Add scatter points to the plot."""
    ax.scatter(
        x_values,
        y_values,
        color=color,
        label=name,
        s=50,  # equivalent to size=10 in plotly
        zorder=5,  # ensure points are on top
    )
    return ax


def add_anomalies(
    ax: Axes,
    time_series: pd.DataFrame,
    is_anomaly: np.ndarray,
    expected_values: np.array,
    expected_bounds: np.array,
):
    """Add anomaly points with confidence intervals."""
    time_series = time_series.copy()
    time_series["expected"] = expected_values
    time_series["upper"] = expected_bounds[:, 0]
    time_series["lower"] = expected_bounds[:, 1]

    # Filter anomaly points
    anomaly_points = time_series[is_anomaly == 1]

    # Add anomaly points
    ax = add_points(
        ax, anomaly_points.index, anomaly_points["value_0"], "Anomalies", "red"
    )

    # Add confidence interval
    ax = add_confidence_interval(ax, time_series)

    return ax


def create_seaborn_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    """Create a time series plot using seaborn style."""
    # Set seaborn style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use seaborn lineplot
    sns.lineplot(
        data=time_series.reset_index(),
        x=time_series.index.name or "index",
        y="value_0",
        ax=ax,
        color="blue",
        linewidth=2,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()

    return fig, ax


def add_seaborn_confidence_interval(ax: Axes, forecast: pd.DataFrame):
    """Add confidence interval using seaborn style."""
    ds = forecast.index

    # Main forecast line
    sns.lineplot(
        x=ds,
        y=forecast["expected"],
        ax=ax,
        color="steelblue",
        linewidth=2,
        label="Forecast",
    )

    # Confidence interval
    ax.fill_between(
        ds,
        forecast["lower"],
        forecast["upper"],
        alpha=0.3,
        color="steelblue",
        label="Confidence Interval",
    )

    return ax


def add_seaborn_anomalies(
    ax: Axes,
    time_series: pd.DataFrame,
    is_anomaly: np.ndarray,
    expected_values: np.array,
    expected_bounds: np.array,
):
    """Add anomalies using seaborn style."""
    time_series = time_series.copy()
    time_series["expected"] = expected_values
    time_series["upper"] = expected_bounds[:, 0]
    time_series["lower"] = expected_bounds[:, 1]

    # Filter anomaly points
    anomaly_points = time_series[is_anomaly == 1]

    # Add anomaly points using seaborn
    if not anomaly_points.empty:
        sns.scatterplot(
            x=anomaly_points.index,
            y=anomaly_points["value_0"],
            ax=ax,
            color="red",
            s=100,
            label="Anomalies",
            zorder=5,
        )

    # Add confidence interval
    ax = add_seaborn_confidence_interval(ax, time_series)

    return ax
