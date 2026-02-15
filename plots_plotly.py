import plotly.graph_objects as go
from typing import Iterable
import pandas as pd
import numpy as np


def add_line(
    fig: go.Figure,
    x_values: Iterable,
    y_values: Iterable,
    name: str,
    color: str,
) -> go.Figure:
    trace = go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        line={"color": color},
    )
    fig.add_trace(trace)
    return fig


def update_layout(fig: go.Figure):
    fig.update_layout(
        title="Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="day", step="day", stepmode="backward"),
                    dict(count=7, label="week", step="day", stepmode="backward"),
                    dict(count=1, label="month", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider={"visible": True},
            type="date",
        ),
        yaxis={"fixedrange": False},
    )
    return fig


def plot_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    fig = go.Figure()
    add_line(fig, time_series.index, time_series["value_0"], title, "blue")
    update_layout(fig)
    return fig


def add_confidence_interval(fig: go.Figure, forecast: pd.DataFrame):
    ds = forecast.index
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["expected"],
            mode="lines",
            name="Forecast",
            line=dict(color="rgba(31, 119, 180, 0.8)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["upper"],
            mode="lines",
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["lower"],
            mode="lines",
            name="Lower Bound",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )
    return fig


def add_points(
    fig: go.Figure,
    x_values: Iterable,
    y_values: Iterable,
    name: str = "Anomalies",
    color: str = "red",
):
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name=name,
            marker=dict(color=color, size=10),
        )
    )
    return fig


def add_anomalies(
    fig: go.Figure,
    time_series: pd.DataFrame,
    is_anomaly: np.ndarray,
    expected_values: np.array,
    expected_bounds: np.array,
):
    time_series = time_series.copy()
    time_series["expected"] = expected_values
    time_series["upper"] = expected_bounds[:, 0]
    time_series["lower"] = expected_bounds[:, 1]
    anomaly_points = time_series[is_anomaly == 1]
    fig = add_points(
        fig, anomaly_points.index, anomaly_points["value_0"], "Anomalies", "red"
    )
    fig = add_confidence_interval(fig, time_series)
    return fig
