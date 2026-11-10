"""Plotly figures shared by the Streamlit app and the HTML report."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .parsing import TestFile

PALETTE = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167),
           (230, 159, 0), (86, 180, 233), (240, 228, 66), (0, 0, 0)]
DASHES = ["solid", "dash", "dot", "dashdot"]


def color(i, a=1.0):
    r, g, b = PALETTE[i % len(PALETTE)]
    return f"rgba({r},{g},{b},{a})"


def _label(tf: TestFile) -> str:
    return f"{tf.test_type} ln {tf.lot} S#{tf.sample}" if tf.lot else tf.name


def overlay(files: list[TestFile], title="Force vs Position", max_points_per_run=3000) -> go.Figure:
    """Force vs position; later cycles drawn darker. Long runs are decimated for speed."""
    fig = go.Figure()
    for i, tf in enumerate(files):
        runs = tf.runs()
        for j, run in enumerate(runs):
            step = max(1, len(run.x) // max_points_per_run)
            alpha = 1.0 if len(runs) == 1 else 0.15 + 0.85 * j / (len(runs) - 1)
            last = j == len(runs) - 1
            fig.add_trace(go.Scattergl(
                x=run.x[::step], y=run.y[::step], mode="lines",
                line=dict(color=color(i, alpha), dash=DASHES[i % len(DASHES)], width=1.5),
                name=_label(tf), legendgroup=tf.name, showlegend=last,
                hovertemplate=f"{_label(tf)} cyc {run.cycle}<br>%{{x:.2f}} cm, %{{y:.1f}} g<extra></extra>",
            ))
    fig.update_layout(template="plotly_white", title=title, height=550,
                      xaxis_title="Position (cm)", yaxis_title="Force (g)  (+ push / - pull)")
    return fig


def sf_breakaway_bar(sf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sf.empty:
        return fig
    sf = sf.assign(Label="ln " + sf["Lot"].astype(str) + " S#" + sf["Sample"].astype(str))
    for k, (ev, sub) in enumerate(sf.groupby("Event", sort=False)):
        fig.add_trace(go.Bar(name=ev, x=sub["Label"], y=sub["|Breakaway Peak| (g)"],
                             marker_color=color(k),
                             customdata=sub[["Breakaway Peak (g)", "Peak - Steady (g)"]],
                             hovertemplate="%{x}<br>Peak %{customdata[0]:.1f} g"
                                           "<br>Peak-Steady %{customdata[1]:.1f} g<extra>" + ev + "</extra>"))
    fig.update_layout(template="plotly_white", barmode="group", height=420,
                      title="SF Breakaway Peak |Force| by Sample", yaxis_title="|Force| (g)")
    return fig


def dur_trend(cyc: pd.DataFrame, labels: dict[str, str] | None = None) -> go.Figure:
    fig = go.Figure()
    if cyc.empty:
        return fig
    for i, (name, sub) in enumerate(cyc.groupby("File", sort=False)):
        lab = (labels or {}).get(name, name)
        sub = sub.sort_values("Cycle")
        fig.add_trace(go.Scatter(x=sub["Cycle"], y=sub["Max Push (g)"], mode="lines+markers",
                                 name=f"{lab} push", line=dict(color=color(i))))
        fig.add_trace(go.Scatter(x=sub["Cycle"], y=sub["Max Pull (g)"], mode="lines+markers",
                                 name=f"{lab} pull", line=dict(color=color(i), dash="dot"),
                                 marker=dict(symbol="diamond")))
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(template="plotly_white", height=480, title="DUR Max Push / Pull per Cycle",
                      xaxis_title="Cycle", yaxis_title="Force (g)")
    return fig
