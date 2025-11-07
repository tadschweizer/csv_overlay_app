
# -*- coding: utf-8 -*-
"""
CSV Overlay Plotter (v3.2)
- SF: robust pause detection via x-histogram + index grouping; strict same-trace baselines
- DUR: per-cycle *peak force* = max(y) (positive-only). NaN if cycle has no positive values.
- Per-file try/except so one bad file won't crash
- Colors for DUR per-cycle chart match overlay
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
        return None
    csv_data = "\n".join(lines[header_index:])
    df = pd.read_csv(StringIO(csv_data), engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

def extract_runs_from_df(df):
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
            xv = x[valid].reset_index(drop=True)
            yv = y[valid].reset_index(drop=True)
            if len(xv) >= 5:
                runs.append((int(cycle), xv, yv))
        return runs
    else:
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[prox_col], errors='coerce')
        valid = ~(x.isna() | y.isna())
        xv = x[valid].reset_index(drop=True)
        yv = y[valid].reset_index(drop=True)
        return [(1, xv, yv)] if len(xv) >= 5 else []

# -----------------------------
# SF pause detection (histogram + index grouping)
# -----------------------------
def _dense_x_centers(x, bin_width=0.05, topk=4):
    x = np.asarray(x, dtype=float)
    if x.size < 20:
        return []
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return []
    n_bins = max(20, int(np.ceil((xmax - xmin) / bin_width)))
    edges = np.linspace(xmin, xmax, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.argsort(counts)[::-1][:topk]
    candidates = sorted(centers[idx])
    # merge centers closer than 0.3 cm
    merged = []
    for c in candidates:
        if not merged or abs(c - merged[-1]) > 0.3:
            merged.append(c)
    return merged

def _indices_near(x, center, tol=0.2):
    x = np.asarray(x, dtype=float)
    mask = np.abs(x - center) <= tol
    return np.where(mask)[0]

def _split_by_gaps(idxs, min_gap=100):
    if idxs.size == 0:
        return []
    chunks = []
    start = idxs[0]
    prev = idxs[0]
    for k in idxs[1:]:
        if k - prev > min_gap:
            chunks.append((start, prev))
            start = k
        prev = k
    chunks.append((start, prev))
    return chunks

def detect_three_pauses(x):
    """
    Returns list of dicts with pause_num, idx_center, pos, direction guess (1 for adv, -1 for retract).
    Strategy:
      - find dense centers in x
      - choose the smallest-x center (near ~20) and enforce TWO time-separated chunks -> Pause1 (early), Pause3 (late)
      - choose the largest-x center (near ~35) -> Pause2 (single chunk)
    """
    x = np.asarray(x, dtype=float)
    centers = _dense_x_centers(x, bin_width=0.05, topk=6)
    if not centers:
        return []
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    # pick near-min and near-max
    c_min = min(centers, key=lambda c: abs(c - xmin - 0.0*(xmax-xmin)))  # just nearest to xmin
    c_max = max(centers, key=lambda c: c)

    # chunks near min center
    idxs_min = _indices_near(x, c_min, tol=0.25)
    chunks_min = _split_by_gaps(idxs_min, min_gap=100)
    # pick earliest and latest chunks for Pause1 and Pause3
    p1 = None; p3 = None
    if len(chunks_min) >= 2:
        s1, e1 = chunks_min[0]
        s3, e3 = chunks_min[-1]
        p1_idx = (s1 + e1)//2; p3_idx = (s3 + e3)//2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}
        p3 = {"pause_num": 3, "idx": int(p3_idx), "pos": float(x[p3_idx]), "dir": -1}
    elif len(chunks_min) == 1:
        # only one occurrence near min-x; use it for Pause1 and synthesize Pause3 later by searching next nearest center
        s1, e1 = chunks_min[0]
        p1_idx = (s1 + e1)//2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}

    # pause 2 near max center
    idxs_max = _indices_near(x, c_max, tol=0.25)
    chunks_max = _split_by_gaps(idxs_max, min_gap=50)
    p2 = None
    if len(chunks_max) >= 1:
        s2, e2 = chunks_max[0]
        p2_idx = (s2 + e2)//2
        p2 = {"pause_num": 2, "idx": int(p2_idx), "pos": float(x[p2_idx]), "dir": -1}

    pauses = [p for p in [p1, p2, p3] if p is not None]
    # try to synthesize a third if needed: pick another center near xmin
    if len(pauses) < 3:
        others = [c for c in centers if abs(c - c_min) < 0.6 and abs(c - c_max) > 0.6]
        for cc in others:
            idxs = _indices_near(x, cc, tol=0.25)
            chunks = _split_by_gaps(idxs, min_gap=100)
            if len(chunks) >= 1:
                s, e = chunks[-1]
                idxc = (s + e)//2
                pauses.append({"pause_num": 3, "idx": int(idxc), "pos": float(x[idxc]), "dir": -1})
                break

    # Ensure exactly three ordered by index
    pauses = sorted(pauses, key=lambda d: d["idx"])[:3]
    # Reassign pause_num by order to match 1,2,3 semantics
    for i, p in enumerate(pauses, 1):
        p["pause_num"] = i
        if i == 1: p["dir"] = 1
        else: p["dir"] = -1
    return pauses

def _peak_in_window(x, y, pos, idx_center, direction, half_cm=0.5):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
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
    out = {
        "File": file_name,
        "Pause1 Peak (g)": np.nan, "Pause1 Baseline (g)": np.nan, "Pause1 %Inc": np.nan,
        "Pause2 Peak (g)": np.nan, "Pause2 Baseline (g)": np.nan, "Pause2 %Inc": np.nan,
        "Pause3 Peak (g)": np.nan, "Pause3 Baseline (g)": np.nan, "Pause3 %Inc": np.nan,
    }
    if len(x) < 20 or len(y) < 20:
        return out
    events = detect_three_pauses(x)
    if not events or len(events) < 3:
        return out
    for e in events:
        peak = _peak_in_window(x, y, e["pos"], e["idx"], e["dir"], half_cm=0.5)
        base = _baseline_same_trace(x, y, e["pause_num"], e["pos"], e["idx"])
        pct = np.nan
        if np.isfinite(base) and abs(base) >= 1e-9 and np.isfinite(peak):
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
# DUR utilities (peak force = max positive y, NaN if none)
# -----------------------------
def analyze_dur_runs(runs, file_name):
    if len(runs) == 0:
        return []
    sorted_runs = sorted(runs, key=lambda t: t[0])
    first = sorted_runs[0]; last = sorted_runs[-1]
    _, x1, y1 = first; _, xL, yL = last
    def peak_pos(x, y):
        yv = y[y > 0]
        if len(yv) == 0:
            return np.nan, np.nan
        idx = int(y[y.idxmax()].index if hasattr(y, "idxmax") else np.argmax(y))
        # for pandas Series y, above line not ideal; simpler:
        idx = int(np.nanargmax(np.array(y)))
        return float(y.iloc[idx]), float(x.iloc[idx])
    p1, x_at1 = peak_pos(x1, y1); pL, x_atL = peak_pos(xL, yL)
    return [{
        "File": file_name,
        "First Cycle Peak Force (g)": round(p1, 3) if np.isfinite(p1) else np.nan,
        "First Cycle Position (cm)": round(x_at1, 3) if np.isfinite(x_at1) else np.nan,
        "Last Cycle Peak Force (g)": round(pL, 3) if np.isfinite(pL) else np.nan,
        "Last Cycle Position (cm)": round(x_atL, 3) if np.isfinite(x_atL) else np.nan,
    }]

def dur_cycle_peak_force(runs, file_name):
    rows = []
    for cycle_label, x, y in sorted(runs, key=lambda t: t[0]):
        yv = y[y > 0]
        if len(y) == 0 or len(yv) == 0:
            rows.append({"File": file_name, "Cycle": int(cycle_label),
                         "Peak Force (g)": np.nan, "Position (cm)": np.nan})
            continue
        idx = int(np.nanargmax(np.array(y)))
        rows.append({"File": file_name, "Cycle": int(cycle_label),
                     "Peak Force (g)": float(y.iloc[idx]), "Position (cm)": float(x.iloc[idx])})
    return pd.DataFrame(rows)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="CSV Overlay Plotter", layout="wide")
st.title("CSV Overlay Plotter")
uploaded = st.file_uploader("Select CSV files", type="csv", accept_multiple_files=True)

if uploaded:
    colors = generate_n_colors(len(uploaded))
    fig = go.Figure()

    sf_rows_all = []
    dur_rows_all = []
    dur_cycle_tables = {}
    color_map = {}

    for idx, file in enumerate(uploaded):
        try:
            df = read_csv_with_flexible_header(StringIO(file.getvalue().decode("utf-8-sig")))
            if df is None or df.empty:
                st.warning(f"Skipped {file.name}: could not find expected header.")
                continue

            runs = extract_runs_from_df(df)
            if not runs:
                st.warning(f"Skipped {file.name}: no valid numeric rows.")
                continue

            base = colors[idx]; color_map[file.name] = base

            # Overlay plot
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
                dur_cycle_tables[file.name] = dur_cycle_peak_force(runs, file.name)

        except Exception as e:
            st.error(f"Error in file {file.name}: {e}")

    fig.update_layout(template="plotly_white", xaxis_title="Encoder/Motor", yaxis_title="Prox", title="Overlay Plot")
    st.plotly_chart(fig, use_container_width=True, height=900)

    if sf_rows_all:
        st.subheader("SF Static-Friction Summary")
        df_sf = pd.DataFrame(sf_rows_all)
        st.dataframe(df_sf, use_container_width=True)
        st.download_button("Download SF summary (CSV)",
                           data=df_sf.to_csv(index=False).encode("utf-8"),
                           file_name="sf_static_friction_summary.csv",
                           mime="text/csv")

    if dur_rows_all:
        st.subheader("DUR First/Last Cycle Peak Force")
        df_dur = pd.DataFrame(dur_rows_all)
        st.dataframe(df_dur, use_container_width=True)
        st.download_button("Download DUR summary (CSV)",
                           data=df_dur.to_csv(index=False).encode("utf-8"),
                           file_name="dur_first_last_peak_force.csv",
                           mime="text/csv")

    if dur_cycle_tables:
        st.subheader("DUR Per-Cycle Peak Force (positive only)")
        all_files = list(dur_cycle_tables.keys())
        selected = st.multiselect("Select DUR files to compare", all_files, default=all_files)
        fig2 = go.Figure()
        for name in selected:
            dfc = dur_cycle_tables[name].copy()
            dfc['Cycle'] = pd.to_numeric(dfc['Cycle'], errors='coerce')
            dfc = dfc.sort_values('Cycle')
            r, g, b = color_map.get(name, (50,50,50))
            fig2.add_trace(go.Scatter(x=dfc['Cycle'], y=dfc['Peak Force (g)'],
                                      mode="lines+markers", name=name,
                                      line=dict(color=f"rgba({r},{g},{b},1)"),
                                      marker=dict(color=f"rgba({r},{g},{b},1)")))
        fig2.update_layout(template="plotly_white",
                           xaxis_title="Cycle",
                           yaxis_title="Peak Force (g)",
                           title="Per-Cycle Peak Force")
        st.plotly_chart(fig2, use_container_width=True, height=500)

        combined = pd.concat([v for v in dur_cycle_tables.values()], ignore_index=True)
        st.download_button("Download DUR per-cycle peak force (CSV)",
                           data=combined.to_csv(index=False).encode("utf-8"),
                           file_name="dur_per_cycle_peak_force.csv",
                           mime="text/csv")
