
# -*- coding: utf-8 -*-
"""
CSV Overlay Plotter (v3.3)
- SF: robust pause detection + same-trace baselines (from v3.2)
- DUR: 
    * First/Last summary uses max magnitude (|y|)
    * Per-cycle table uses max magnitude (|y|)
    * Per-cycle plot uses peak force (max positive y). If no positive values in a cycle, plot NaN.
- Fix: remove any y.idxmax/index usage to avoid numpy.float64 errors.
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
# SF pause detection (from v3.2)
# -----------------------------
def _dense_x_centers(x, bin_width=0.05, topk=6):
    x = np.asarray(x, dtype=float)
    if x.size < 20: return []
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return []
    n_bins = max(20, int(np.ceil((xmax - xmin) / bin_width)))
    edges = np.linspace(xmin, xmax, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.argsort(counts)[::-1][:topk]
    candidates = sorted(centers[idx])
    merged = []
    for c in candidates:
        if not merged or abs(c - merged[-1]) > 0.3:
            merged.append(c)
    return merged

def _indices_near(x, center, tol=0.25):
    x = np.asarray(x, dtype=float)
    mask = np.abs(x - center) <= tol
    return np.where(mask)[0]

def _split_by_gaps(idxs, min_gap=100):
    if idxs.size == 0: return []
    chunks = []
    start = idxs[0]; prev = idxs[0]
    for k in idxs[1:]:
        if k - prev > min_gap:
            chunks.append((start, prev)); start = k
        prev = k
    chunks.append((start, prev))
    return chunks

def detect_three_pauses(x):
    x = np.asarray(x, dtype=float)
    centers = _dense_x_centers(x, bin_width=0.05, topk=6)
    if not centers: return []
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    c_min = min(centers, key=lambda c: abs(c - xmin))
    c_max = max(centers, key=lambda c: c)
    idxs_min = _indices_near(x, c_min, tol=0.25)
    chunks_min = _split_by_gaps(idxs_min, min_gap=100)
    p1 = p3 = None
    if len(chunks_min) >= 2:
        s1, e1 = chunks_min[0]; s3, e3 = chunks_min[-1]
        p1_idx = (s1 + e1)//2; p3_idx = (s3 + e3)//2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}
        p3 = {"pause_num": 3, "idx": int(p3_idx), "pos": float(x[p3_idx]), "dir": -1}
    elif len(chunks_min) == 1:
        s1, e1 = chunks_min[0]; p1_idx = (s1 + e1)//2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}
    idxs_max = _indices_near(x, c_max, tol=0.25)
    chunks_max = _split_by_gaps(idxs_max, min_gap=50)
    p2 = None
    if len(chunks_max) >= 1:
        s2, e2 = chunks_max[0]; p2_idx = (s2 + e2)//2
        p2 = {"pause_num": 2, "idx": int(p2_idx), "pos": float(x[p2_idx]), "dir": -1}
    pauses = [p for p in [p1, p2, p3] if p is not None]
    pauses = sorted(pauses, key=lambda d: d["idx"])[:3]
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
# DUR utilities
# -----------------------------
def _max_abs_with_pos(x, y):
    """Return (max_abs, x_at_abs, max_pos, x_at_pos). All NaN-safe; if no positive values, max_pos is NaN."""
    arr_y = np.asarray(y, dtype=float)
    arr_x = np.asarray(x, dtype=float)
    if arr_y.size == 0 or arr_x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    # max magnitude
    idx_abs = int(np.nanargmax(np.abs(arr_y)))
    max_abs = float(np.abs(arr_y[idx_abs]))
    x_at_abs = float(arr_x[idx_abs])
    # max positive peak force
    pos_mask = arr_y > 0
    if not np.any(pos_mask):
        return (max_abs, x_at_abs, np.nan, np.nan)
    # we want the location of the maximum positive y
    idx_pos = int(np.nanargmax(np.where(pos_mask, arr_y, -np.inf)))
    max_pos = float(arr_y[idx_pos])
    x_at_pos = float(arr_x[idx_pos])
    return (max_abs, x_at_abs, max_pos, x_at_pos)

def analyze_dur_first_last(runs, file_name):
    """First/Last summary using max magnitude (|y|)."""
    if len(runs) == 0:
        return []
    runs_sorted = sorted(runs, key=lambda t: t[0])
    _, x1, y1 = runs_sorted[0]
    _, xL, yL = runs_sorted[-1]
    max_abs1, x_abs1, _, _ = _max_abs_with_pos(x1, y1)
    max_absL, x_absL, _, _ = _max_abs_with_pos(xL, yL)
    return [{
        "File": file_name,
        "First Cycle Max |Force| (g)": round(max_abs1, 3) if np.isfinite(max_abs1) else np.nan,
        "First Cycle Position (cm)": round(x_abs1, 3) if np.isfinite(x_abs1) else np.nan,
        "Last Cycle Max |Force| (g)": round(max_absL, 3) if np.isfinite(max_absL) else np.nan,
        "Last Cycle Position (cm)": round(x_absL, 3) if np.isfinite(x_absL) else np.nan,
    }]

def dur_cycle_tables(runs, file_name):
    """Return two per-cycle tables: magnitude table and positive-peak table (for plotting)."""
    rows_mag = []
    rows_pos = []
    for cycle_label, x, y in sorted(runs, key=lambda t: t[0]):
        max_abs, x_abs, max_pos, x_pos = _max_abs_with_pos(x, y)
        rows_mag.append({
            "File": file_name,
            "Cycle": int(cycle_label),
            "Max |Force| (g)": max_abs if np.isfinite(max_abs) else np.nan,
            "Position (cm)": x_abs if np.isfinite(x_abs) else np.nan,
        })
        rows_pos.append({
            "File": file_name,
            "Cycle": int(cycle_label),
            "Peak Force (g)": max_pos if np.isfinite(max_pos) else np.nan,
            "Position (cm)": x_pos if np.isfinite(x_pos) else np.nan,
        })
    return pd.DataFrame(rows_mag), pd.DataFrame(rows_pos)

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
    dur_first_last_rows = []
    dur_cycle_mag_tables = {}
    dur_cycle_pos_tables = {}
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
                dur_first_last_rows.extend(analyze_dur_first_last(runs, file.name))
                df_mag, df_pos = dur_cycle_tables(runs, file.name)
                dur_cycle_mag_tables[file.name] = df_mag
                dur_cycle_pos_tables[file.name] = df_pos

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

    if dur_first_last_rows:
        st.subheader("DUR First/Last Cycle (Max |Force|)")
        df_dur = pd.DataFrame(dur_first_last_rows)
        st.dataframe(df_dur, use_container_width=True)
        st.download_button("Download DUR first/last summary (CSV)",
                           data=df_dur.to_csv(index=False).encode("utf-8"),
                           file_name="dur_first_last_max_magnitude.csv",
                           mime="text/csv")

    if dur_cycle_mag_tables:
        st.subheader("DUR Per-Cycle Tables")
        combined_mag = pd.concat([v for v in dur_cycle_mag_tables.values()], ignore_index=True)
        st.markdown("**Per-cycle Max |Force| (table)**")
        st.dataframe(combined_mag, use_container_width=True)
        st.download_button("Download per-cycle Max |Force| (CSV)",
                           data=combined_mag.to_csv(index=False).encode("utf-8"),
                           file_name="dur_per_cycle_max_magnitude.csv",
                           mime="text/csv")

    if dur_cycle_pos_tables:
        st.subheader("DUR Per-Cycle Peak Force Plot (positive only)")
        all_files = list(dur_cycle_pos_tables.keys())
        selected = st.multiselect("Select DUR files to compare", all_files, default=all_files)
        fig2 = go.Figure()
        for name in selected:
            dfc = dur_cycle_pos_tables[name].copy()
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
                           title="Per-Cycle Peak Force (positive only)")
        st.plotly_chart(fig2, use_container_width=True, height=500)

        combined_pos = pd.concat([v for v in dur_cycle_pos_tables.values()], ignore_index=True)
        st.download_button("Download per-cycle Peak Force (CSV)",
                           data=combined_pos.to_csv(index=False).encode("utf-8"),
                           file_name="dur_per_cycle_peak_force.csv",
                           mime="text/csv")
