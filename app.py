
# -*- coding: utf-8 -*-
"""
CSV Overlay Plotter (v3)
- Keeps original overlay plotting
- SF: three pauses always; baseline computed on the *same trace* (index-aware)
  * Pause 1 (advance near ~20 cm): baseline = mean of [pos-1.0,pos) + (pos,pos+1.5]
  * Pause 2 (turnaround near ~35 cm): baseline = mean of [pos-1.0,pos] AFTER the spike only
  * Pause 3 (retract near ~20 cm): baseline = mean of [pos-1.0,pos) + (pos,pos+1.5] on retract trace
- DUR: first/last table + per-cycle maxima chart with colors matching overlay
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
# SF pause detection (index-aware to split ~20cm into two pauses)
# -----------------------------
def _find_pause_groups(x, window_cm=0.25):
    x = np.asarray(x, dtype=float)
    dx = np.gradient(x)
    dwell = np.abs(dx) < np.percentile(np.abs(dx), 20)
    groups = []
    i = 0; n = len(x)
    while i < n:
        if dwell[i]:
            j = i
            while j+1 < n and dwell[j+1]:
                j += 1
            if (j - i + 1) >= 10:
                groups.append((i, j))
            i = j + 1
        else:
            i += 1
    merged = []
    for g in groups:
        if not merged: merged.append(g); continue
        pi, pj = merged[-1]; gi, gj = g
        if gi - pj <= 5:
            merged[-1] = (pi, gj)
        else:
            merged.append(g)
    return merged

def _pause_events_three(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    groups = _find_pause_groups(x)
    if not groups:
        return []
    events = []
    for (i0, i1) in groups:
        c = (i0 + i1) // 2
        pos = float(np.nanmedian(x[i0:i1+1]))
        j1 = min(len(x)-2, i1 + 5)
        j0 = max(1, i1 + 1)
        dir_after = np.nanmean(np.gradient(x[j0:j1+1]))
        events.append({"idx": int(c), "pos": pos, "dir_after": dir_after})
    events_sorted = sorted(events, key=lambda e: e["idx"])
    cleaned = []
    for e in events_sorted:
        if not cleaned or (e["idx"] - cleaned[-1]["idx"]) > 20:
            cleaned.append(e)
    if len(cleaned) >= 3:
        final = cleaned[:3]
    elif len(cleaned) == 2:
        xs = [ev["pos"] for ev in cleaned]
        idx_20 = int(np.argmin(np.abs(np.array(xs) - 20.0)))
        e20 = cleaned[idx_20]
        e20b = dict(e20); e20b["idx"] = e20["idx"] + 200; e20b["dir_after"] = -abs(e20b["dir_after"])
        final = sorted([e for e in cleaned] + [e20b], key=lambda z: z["idx"])[:3]
    else:
        final = cleaned
    for k, e in enumerate(final):
        e["pause_num"] = k+1
    return final[:3]

def _peak_in_window(x, y, pos, idx_center, direction, half_cm=0.5):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = (x >= pos - half_cm) & (x <= pos + half_cm)
    if not np.any(mask):
        return np.nan
    vals = y[mask]
    return float(np.nanmax(vals)) if direction >= 0 else float(np.nanmin(vals))

def _baseline_same_trace(x, y, pause_num, pos, idx_center):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    idx = np.arange(len(x))
    if pause_num in (1,3):
        before = (idx < idx_center) & (x >= pos - 1.0) & (x < pos)
        after  = (idx > idx_center) & (x > pos) & (x <= pos + 1.5)
        sample = np.concatenate([y[before], y[after]])
    else:
        after_only = (idx > idx_center) & (x <= pos) & (x >= pos - 1.0)
        sample = y[after_only]
    if sample.size == 0:
        return np.nan
    return float(np.nanmean(sample))

def analyze_sf_run_corrected(x, y, file_name):
    events = _pause_events_three(x, y)
    events = sorted(events, key=lambda e: e["idx"])[:3]
    for e in events:
        e["direction"] = 1 if e["pause_num"] == 1 else -1
    out = {
        "File": file_name,
        "Pause1 Peak (g)": np.nan, "Pause1 Baseline (g)": np.nan, "Pause1 %Inc": np.nan,
        "Pause2 Peak (g)": np.nan, "Pause2 Baseline (g)": np.nan, "Pause2 %Inc": np.nan,
        "Pause3 Peak (g)": np.nan, "Pause3 Baseline (g)": np.nan, "Pause3 %Inc": np.nan,
    }
    for e in events:
        peak = _peak_in_window(x, y, e["pos"], e["idx"], e["direction"], half_cm=0.5)
        base = _baseline_same_trace(x, y, e["pause_num"], e["pos"], e["idx"])
        if not np.isfinite(base) or abs(base) < 1e-9 or not np.isfinite(peak):
            pct = np.nan
        else:
            pct = (abs(peak) - abs(base)) / abs(base) * 100.0
        if e["pause_num"] == 1:
            out["Pause1 Peak (g)"] = round(float(peak), 3) if np.isfinite(peak) else np.nan
            out["Pause1 Baseline (g)"] = round(float(base), 3) if np.isfinite(base) else np.nan
            out["Pause1 %Inc"] = round(float(pct), 2) if np.isfinite(pct) else np.nan
        elif e["pause_num"] == 2:
            out["Pause2 Peak (g)"] = round(float(peak), 3) if np.isfinite(peak) else np.nan
            out["Pause2 Baseline (g)"] = round(float(base), 3) if np.isfinite(base) else np.nan
            out["Pause2 %Inc"] = round(float(pct), 2) if np.isfinite(pct) else np.nan
        else:
            out["Pause3 Peak (g)"] = round(float(peak), 3) if np.isfinite(peak) else np.nan
            out["Pause3 Baseline (g)"] = round(float(base), 3) if np.isfinite(base) else np.nan
            out["Pause3 %Inc"] = round(float(pct), 2) if np.isfinite(pct) else np.nan
    return out

# -----------------------------
# DUR utilities
# -----------------------------
def analyze_dur_runs(runs, file_name):
    if len(runs) == 0:
        return []
    sorted_runs = sorted(runs, key=lambda t: t[0])
    first = sorted_runs[0]
    last  = sorted_runs[-1]
    _, x1, y1 = first; _, xL, yL = last
    def peak_abs(x, y):
        if len(y) == 0: return np.nan, np.nan
        idx = int(np.nanargmax(np.abs(y)))
        return float(y.iloc[idx]), float(x.iloc[idx])
    p1, x_at1 = peak_abs(x1, y1); pL, x_atL = peak_abs(xL, yL)
    return [{
        "File": file_name,
        "First Cycle Peak Force (g)": round(p1, 3) if np.isfinite(p1) else np.nan,
        "First Cycle Position (cm)": round(x_at1, 3) if np.isfinite(x_at1) else np.nan,
        "Last Cycle Peak Force (g)": round(pL, 3) if np.isfinite(pL) else np.nan,
        "Last Cycle Position (cm)": round(x_atL, 3) if np.isfinite(x_atL) else np.nan,
    }]

def dur_cycle_maxima(runs, file_name):
    rows = []
    for cycle_label, x, y in sorted(runs, key=lambda t: t[0]):
        if len(y) == 0: continue
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
    dur_cycle_tables = {}
    color_map = {}

    for idx, file in enumerate(uploaded):
        df = read_csv_with_flexible_header(StringIO(file.getvalue().decode("utf-8-sig")))
        if df is None or df.empty:
            continue
        runs = extract_runs_from_df(df)
        base = colors[idx]; color_map[file.name] = base

        if len(runs) > 1:
            for i, (cycle_label, x, y) in enumerate(runs):
                alpha = 0.1 + 0.9 * i/(len(runs)-1) if len(runs) > 1 else 1.0
                rgba = f"rgba({base[0]},{base[1]},{base[2]},{alpha})"
                show = (i == len(runs)-1); name = file.name if show else None
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba),
                                         name=name, showlegend=show, legendgroup=file.name))
        else:
            _, x, y = runs[0]
            rgba = f"rgba({base[0]},{base[1]},{base[2]},1)"
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba), name=file.name))

        fname_lower = file.name.lower()
        if "sf" in fname_lower:
            _, x_sf, y_sf = runs[-1]
            row = analyze_sf_run_corrected(x_sf, y_sf, file.name)
            sf_rows_all.append(row)
        if "dur" in fname_lower:
            dur_rows_all.extend(analyze_dur_runs(runs, file.name))
            dur_cycle_tables[file.name] = dur_cycle_maxima(runs, file.name)

    # Overlay plot
    fig.update_layout(template="plotly_white",
                      xaxis_title="Encoder/Motor",
                      yaxis_title="Prox",
                      title="Overlay Plot")
    st.plotly_chart(fig, use_container_width=True, height=900)

    # SF table
    if sf_rows_all:
        st.subheader("SF Static-Friction Summary")
        df_sf = pd.DataFrame(sf_rows_all)
        st.dataframe(df_sf, use_container_width=True)
        st.download_button("Download SF summary (CSV)",
                           data=df_sf.to_csv(index=False).encode("utf-8"),
                           file_name="sf_static_friction_summary.csv",
                           mime="text/csv")

    # DUR first/last
    if dur_rows_all:
        st.subheader("DUR First/Last Cycle Peak Magnitudes")
        df_dur = pd.DataFrame(dur_rows_all)
        st.dataframe(df_dur, use_container_width=True)
        st.download_button("Download DUR summary (CSV)",
                           data=df_dur.to_csv(index=False).encode("utf-8"),
                           file_name="dur_cycle_peaks_summary.csv",
                           mime="text/csv")

    # DUR per-cycle chart colored to overlay
    if dur_cycle_tables:
        st.subheader("DUR Per-Cycle Max Magnitude")
        all_files = list(dur_cycle_tables.keys())
        selected = st.multiselect("Select DUR files to compare", all_files, default=all_files)
        fig2 = go.Figure()
        for name in selected:
            dfc = dur_cycle_tables[name].copy()
            dfc['Cycle'] = pd.to_numeric(dfc['Cycle'], errors='coerce')
            dfc = dfc.sort_values('Cycle')
            r, g, b = color_map.get(name, (50,50,50))
            fig2.add_trace(go.Scatter(x=dfc['Cycle'], y=dfc['Max |Force| (g)'],
                                      mode="lines+markers", name=name,
                                      line=dict(color=f"rgba({r},{g},{b},1)"),
                                      marker=dict(color=f"rgba({r},{g},{b},1)")))
        fig2.update_layout(template="plotly_white",
                           xaxis_title="Cycle",
                           yaxis_title="Max |Force| (g)",
                           title="Per-Cycle Peak Magnitudes")
        st.plotly_chart(fig2, use_container_width=True, height=500)

        combined = pd.concat([v for v in dur_cycle_tables.values()], ignore_index=True)
        st.download_button("Download DUR per-cycle maxima (CSV)",
                           data=combined.to_csv(index=False).encode("utf-8"),
                           file_name="dur_per_cycle_maxima.csv",
                           mime="text/csv")
