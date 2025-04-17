# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:20:17 2025

@author: tschweiz
"""
import streamlit as st

st.set_page_config(
    page_title="CSV Overlay Plotter",
    layout="wide",           # ← makes your app use the full browser width
    initial_sidebar_state="auto"
)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import colorsys
from io import StringIO

def generate_n_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.65
        value = 0.80
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

def read_csv_with_flexible_header(file_like):
    lines = file_like.read().splitlines()
    header_index = None
    for i, line in enumerate(lines):
        low = line.lower()
        if 'prox' in low and ('encoder' in low or 'motor' in low):
            header_index = i
            break
    if header_index is None:
        st.error("No header row found.")
        return None
    csv_data = "\n".join(lines[header_index:])
    df = pd.read_csv(StringIO(csv_data), engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

def extract_runs_from_df(df):
    # find X column
    encoder = [c for c in df.columns if 'encoder' in c.lower()]
    motor = [c for c in df.columns if 'motor' in c.lower()]
    x_col = encoder[0] if encoder else (motor[0] if motor else None)
    prox_col = [c for c in df.columns if 'prox' in c.lower()]
    if x_col is None or not prox_col:
        return []
    prox_col = prox_col[0]
    cycle_cols = [c for c in df.columns if c.lower() == 'cycle']
    if cycle_cols:
        df['cycle_num'] = pd.to_numeric(df[cycle_cols[0]], errors='coerce')
        runs = []
        for cycle in sorted(df['cycle_num'].dropna().unique()):
            sub = df[df['cycle_num'] == cycle]
            x = pd.to_numeric(sub[x_col], errors='coerce')
            y = pd.to_numeric(sub[prox_col], errors='coerce')
            valid = ~(x.isna() | y.isna())
            runs.append((cycle, x[valid], y[valid]))
        return runs
    else:
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[prox_col], errors='coerce')
        valid = ~(x.isna() | y.isna())
        return [(None, x[valid], y[valid])]

def main():
    st.title("CSV Overlay Plotter")
    st.write("Upload one or more CSV files to see overlaid prox vs encoder/motor plots.")
    uploaded = st.file_uploader("Select CSV files", type="csv", accept_multiple_files=True)
    if not uploaded:
        return

    colors = generate_n_colors(len(uploaded))
    fig = go.Figure()

    for idx, file in enumerate(uploaded):
        df = read_csv_with_flexible_header(StringIO(file.getvalue().decode("utf-8-sig")))
        if df is None:
            continue
        runs = extract_runs_from_df(df)
        base = colors[idx]
        if len(runs) > 1:
            for i, (_, x, y) in enumerate(runs):
                alpha = 0.1 + 0.9 * i/(len(runs)-1)
                rgba = f"rgba({base[0]},{base[1]},{base[2]},{alpha})"
                show = (i == len(runs)-1)
                name = file.name if show else None
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba),
                                         name=name, showlegend=show, legendgroup=file.name))
        else:
            _, x, y = runs[0]
            rgba = f"rgba({base[0]},{base[1]},{base[2]},1)"
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba), name=file.name))

    fig.update_layout(template="plotly_white",
                      xaxis_title="Encoder/Motor",
                      yaxis_title="Prox",
                      title="Overlay Plot")
    st.plotly_chart(fig, use_container_width=True, height=900)

if __name__ == "__main__":
    main()
