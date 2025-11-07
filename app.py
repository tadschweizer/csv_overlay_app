
# -*- coding: utf-8 -*-
"""
Enhanced CSV Overlay Plotter (v2)
- Keeps original overlay plotting
- SF: 3 static-friction pauses, peak & % increase vs baseline
- DUR: first/last cycle peak table
- DUR NEW: per-cycle max |force| chart with multi-file selection (x=cycle, y=max |force|)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorsys
from io import StringIO

# -----------------------------
# Plot colors
# -----------------------------
def generate_n_colors(n):
    colors = []
    for i in range(max(n,1)):
        hue = i / max(n,1)
        saturation = 0.65
        value = 0.80
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

# -----------------------------
# CSV parsing
# -----------------------------
def read_csv_with_flexible_header(file_like):
    """Find the header row containing 'prox' and ('encoder' or 'motor') then read CSV from there."""
    text = file_like.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", errors="ignore")
    lines = text.splitlines()
    header_index = None
    for i, line in enumerate(lines):
        low = line.lower()
        if 'prox' in low and ('encoder' in low or 'motor' in low):
            header_index = i
            break
    if header_index is None:
        st.error("No header row found (need a row with 'prox' and 'encoder' or 'motor').")
        return None
    csv_data = "\n".join(lines[header_index:])
    df = pd.read_csv(StringIO(csv_data), engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

def extract_runs_from_df(df):
    """Return list of runs as (cycle_label, x_series, y_series). Uses 'Cycle' column if present."""
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
            sub = df[df['cycle_num'] == cycle].copy()
            x = pd.to_numeric(sub[x_col], errors='coerce')
            y = pd.to_numeric(sub[prox_col], errors='coerce')
            valid = ~(x.isna() | y.isna())
            if valid.any():
                runs.append((int(cycle), x[valid].reset_index(drop=True), y[valid].reset_index(drop=True)))
        return runs
    else:
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[prox_col], errors='coerce')
        valid = ~(x.isna() | y.isna())
        return [(1, x[valid].reset_index(drop=True), y[valid].reset_index(drop=True))]

# -----------------------------
# SF spike detection utilities
# -----------------------------
def _histogram_peaks(x, bin_width=0.05, expected=3):
    if len(x) < 10:
        return []
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return []
    n_bins = max(10, int(np.ceil((x_max - x_min) / bin_width)))
    edges = np.linspace(x_min, x_max, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    thr = max(np.mean(counts) + 2.5*np.std(counts), np.percentile(counts, 95))
    candidate_idx = np.where(counts >= thr)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.argsort(counts)[-expected:]
    positions = sorted(centers[candidate_idx])
    merged = []
    for pos in positions:
        if not merged or abs(pos - merged[-1]) > 0.5:
            merged.append(pos)
    if len(merged) > expected:
        idx_map = [np.argmin(np.abs(centers - m)) for m in merged]
        order = np.argsort(counts[idx_map])[::-1][:expected]
        merged = sorted([merged[i] for i in order])
    return merged

def _local_direction(x, pos, look=0.4):
    before = (x >= pos - look) & (x < pos - 0.05)
    after  = (x > pos + 0.05) & (x <= pos + look)
    if before.sum() == 0 or after.sum() == 0:
        return 0.0
    return (x[after].mean() - x[before].mean())

def _window_mask(x, center, half_width):
    return (x >= center - half_width) & (x <= center + half_width)

def _baseline(y, x, pos, left=1.0, right=1.5, exclude_half=0.25):
    mask = (x >= pos - left) & (x <= pos + right)
    core = _window_mask(x, pos, exclude_half)
    use = mask & (~core)
    vals = y[use]
    if vals.size == 0:
        return np.nan
    q05, q95 = np.nanpercentile(vals, [5, 95])
    trimmed = vals[(vals >= q05) & (vals <= q95)]
    if trimmed.size == 0:
        return np.nanmedian(vals)
    return np.nanmedian(trimmed)

def analyze_sf_run(x, y, file_name):
    rows = []
    pause_positions = _histogram_peaks(np.array(x), bin_width=0.05, expected=3)
    pause_positions = sorted(pause_positions)
    for i, pos in enumerate(pause_positions, start=1):
        dir_delta = _local_direction(np.array(x), pos, look=0.4)
        win_mask = _window_mask(np.array(x), pos, half_width=0.5)
        win_vals = np.array(y)[win_mask]
        if win_vals.size == 0:
            peak = np.nan
        else:
            peak = np.nanmax(win_vals) if dir_delta >= 0 else np.nanmin(win_vals)
        base = _baseline(np.array(y), np.array(x), pos, left=1.0, right=1.5, exclude_half=0.25)
        base_mag = np.nan if not np.isfinite(base) else abs(base)
        peak_mag = np.nan if not np.isfinite(peak) else abs(peak)
        pct_increase = np.nan if (not np.isfinite(base_mag) or base_mag < 1e-9) else (peak_mag - base_mag)/base_mag*100.0
        rows.append({
            "File": file_name,
            "Pause #": i,
            "Estimated Position (cm)": round(float(pos), 3) if np.isfinite(pos) else np.nan,
            "Direction": "Advance" if dir_delta >= 0 else "Retract",
            "Peak Force (g)": round(float(peak), 3) if np.isfinite(peak) else np.nan,
            "Baseline |Force| (g)": round(float(base_mag), 3) if np.isfinite(base_mag) else np.nan,
            "% Increase vs Baseline": round(float(pct_increase), 2) if np.isfinite(pct_increase) else np.nan,
        })
    return rows

# -----------------------------
# DUR analysis utilities
# -----------------------------
def analyze_dur_runs(runs, file_name):
    """First/last cycle peak signed values, with positions."""
    if len(runs) == 0:
        return []
    sorted_runs = sorted(runs, key=lambda t: t[0])
    first = sorted_runs[0]
    last  = sorted_runs[-1]

    _, x1, y1 = first
    _, xL, yL = last

    def peak_abs(x, y):
        if len(y) == 0:
            return np.nan, np.nan
        idx = int(np.nanargmax(np.abs(y)))
        return float(y.iloc[idx]), float(x.iloc[idx])

    p1, x_at1 = peak_abs(x1, y1)
    pL, x_atL = peak_abs(xL, yL)

    return [{
        "File": file_name,
        "First Cycle Peak Force (g)": round(p1, 3) if np.isfinite(p1) else np.nan,
        "First Cycle Position (cm)": round(x_at1, 3) if np.isfinite(x_at1) else np.nan,
        "Last Cycle Peak Force (g)": round(pL, 3) if np.isfinite(pL) else np.nan,
        "Last Cycle Position (cm)": round(x_atL, 3) if np.isfinite(x_atL) else np.nan,
    }]

def dur_cycle_maxima(runs, file_name):
    """Per-cycle max |force| (magnitude) and location."""
    rows = []
    for cycle_label, x, y in sorted(runs, key=lambda t: t[0]):
        if len(y) == 0:
            continue
        idx = int(np.nanargmax(np.abs(y)))
        rows.append({
            "File": file_name,
            "Cycle": int(cycle_label),
            "Max |Force| (g)": float(abs(y.iloc[idx])),
            "Signed Peak (g)": float(y.iloc[idx]),
            "Position (cm)": float(x.iloc[idx]),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="CSV Overlay Plotter", layout="wide", initial_sidebar_state="auto")
st.title("CSV Overlay Plotter")
st.write("Upload one or more CSV files to see overlaid prox vs encoder/motor plots.")
uploaded = st.file_uploader("Select CSV files", type="csv", accept_multiple_files=True)

if uploaded:
    colors = generate_n_colors(len(uploaded))
    fig = go.Figure()

    sf_rows_all = []
    dur_rows_all = []
    dur_cycle_tables = {}  # file -> per-cycle maxima DF

    for idx, file in enumerate(uploaded):
        # Read
        df = read_csv_with_flexible_header(StringIO(file.getvalue().decode("utf-8-sig")))
        if df is None or df.empty:
            continue

        # Extract runs for plotting (unchanged behavior)
        runs = extract_runs_from_df(df)
        base = colors[idx]

        if len(runs) > 1:
            for i, (cycle_label, x, y) in enumerate(runs):
                alpha = 0.1 + 0.9 * i/(len(runs)-1) if len(runs) > 1 else 1.0
                rgba = f"rgba({base[0]},{base[1]},{base[2]},{alpha})"
                show = (i == len(runs)-1)
                name = file.name if show else None
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba),
                                         name=name, showlegend=show, legendgroup=file.name))
        else:
            _, x, y = runs[0]
            rgba = f"rgba({base[0]},{base[1]},{base[2]},1)"
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba), name=file.name))

        # ----- NEW: Analyses -----
        fname_lower = file.name.lower()
        if "sf" in fname_lower:
            run_for_sf = runs[-1] if len(runs) else (None, pd.Series(dtype=float), pd.Series(dtype=float))
            _, x_sf, y_sf = run_for_sf
            sf_rows_all.extend(analyze_sf_run(x_sf, y_sf, file.name))

        if "dur" in fname_lower:
            dur_rows_all.extend(analyze_dur_runs(runs, file.name))
            # per-cycle maxima for separate chart
            dur_cycle_tables[file.name] = dur_cycle_maxima(runs, file.name)

    # Overlay plot (original)
    fig.update_layout(template="plotly_white",
                      xaxis_title="Encoder/Motor",
                      yaxis_title="Prox",
                      title="Overlay Plot")
    st.plotly_chart(fig, use_container_width=True, height=900)

    # ----- SF table -----
    if sf_rows_all:
        st.subheader("SF Static-Friction Peaks")
        df_sf = pd.DataFrame(sf_rows_all)
        st.dataframe(df_sf, use_container_width=True)
        st.download_button("Download SF summary (CSV)",
                           data=df_sf.to_csv(index=False).encode("utf-8"),
                           file_name="sf_static_friction_summary.csv",
                           mime="text/csv")

    # ----- DUR table: first/last -----
    if dur_rows_all:
        st.subheader("DUR First/Last Cycle Peak Magnitudes")
        df_dur = pd.DataFrame(dur_rows_all)
        st.dataframe(df_dur, use_container_width=True)
        st.download_button("Download DUR summary (CSV)",
                           data=df_dur.to_csv(index=False).encode("utf-8"),
                           file_name="dur_cycle_peaks_summary.csv",
                           mime="text/csv")

    # ----- DUR chart: per-cycle maxima -----
    if dur_cycle_tables:
        st.subheader("DUR Per-Cycle Max Magnitude")
        all_files = list(dur_cycle_tables.keys())
        selected = st.multiselect("Select DUR files to compare", all_files, default=all_files)
        fig2 = go.Figure()
        for name in selected:
            dfc = dur_cycle_tables[name].copy()
            dfc['Cycle'] = pd.to_numeric(dfc['Cycle'], errors='coerce')
            dfc = dfc.sort_values('Cycle')
            fig2.add_trace(go.Scatter(x=dfc['Cycle'], y=dfc['Max |Force| (g)'],
                                      mode="lines+markers", name=name))
        fig2.update_layout(template="plotly_white",
                           xaxis_title="Cycle",
                           yaxis_title="Max |Force| (g)",
                           title="Per-Cycle Peak Magnitudes")
        st.plotly_chart(fig2, use_container_width=True, height=500)

        # Combined CSV
        combined = pd.concat([v for v in dur_cycle_tables.values()], ignore_index=True)
        st.download_button("Download DUR per-cycle maxima (CSV)",
                           data=combined.to_csv(index=False).encode("utf-8"),
                           file_name="dur_per_cycle_maxima.csv",
                           mime="text/csv")
