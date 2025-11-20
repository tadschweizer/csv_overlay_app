# -*- coding: utf-8 -*-
"""
CSV Overlay Plotter (v3.9.1)

SF improvements (combined):
- Robust dwell-based pause detection: identifies low-x (~20 cm) twice and high-x (~35 cm) once.
- Pre value sampled at EXACT Xp ± 0.100 cm (snapped to 3 decimals):
  • Choose the latest sample BEFORE the pause on the approach pass (tight band ±0.02 → ±0.05 → ±0.10).
  • If nothing in band, linearly interpolate on the approach side.
- Post ‘peak’ = extreme within the first 1.0 cm AFTER the pause in the motion direction (max if +, min if −).
- Turnaround (Pause #2) also reports y at (Xp − 1.000 cm) AFTER the pause and Δ = peak − that value.

DUR behavior unchanged:
- Overlay plot (no forced y = 0).
- Per-cycle positive-only peak plot with y-axis starting at 0.
- No idxmax/index usage on user DataFrames.
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
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
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
    csv_data = "\n".join(lines[header_index:])  # FIXED
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
# Pause detection (dwell-based, position-grouped)
# -----------------------------
def _dense_x_centers(x, bin_width=0.05, topk=20):
    x = np.asarray(x, dtype=float)
    if x.size < 20:
        return np.array([]), np.array([])
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.array([]), np.array([])
    n_bins = max(20, int(np.ceil((xmax - xmin) / bin_width)))
    edges = np.linspace(xmin, xmax, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.argsort(counts)[::-1][:topk]
    return centers[idx], counts[idx]

def _pause_blocks_from_center(x, center, tol=0.25, min_gap=80):
    x = np.asarray(x, dtype=float)
    mask = np.abs(x - center) <= tol
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return []
    blocks = []
    start = idxs[0]
    prev = idxs[0]
    for k in idxs[1:]:
        if k - prev > min_gap:
            blocks.append((start, prev))
            start = k
        prev = k
    blocks.append((start, prev))
    return blocks

def detect_pauses_three_strict(x):
    """
    Return three pauses: low-x twice (~20 cm) and high-x once (~35 cm).
    Each pause dict has: pause_num, idx, pos, dir (post-pause motion sign).
    """
    x = np.asarray(x, dtype=float)
    centers, _ = _dense_x_centers(x, bin_width=0.05, topk=20)
    if len(centers) == 0:
        return []

    cand_blocks = []
    for c in centers:
        for s, e in _pause_blocks_from_center(x, c, tol=0.25, min_gap=80):
            xm = float(np.nanmedian(x[s:e+1]))
            cand_blocks.append({"start": int(s), "end": int(e), "x_med": xm, "len": int(e - s + 1)})
    if not cand_blocks:
        return []

    # Group by position using 0.5 cm buckets
    groups = {}
    for b in cand_blocks:
        key = round(b["x_med"] * 2) / 2.0
        groups.setdefault(key, []).append(b)

    # Pick the two strongest dwell positions (expected: ~20 and ~35)
    items = [(k, sum(bb["len"] for bb in v), v) for k, v in groups.items()]
    items.sort(key=lambda t: t[1], reverse=True)
    if len(items) < 2:
        # Fallback: earliest 3 by time
        blocks = sorted(cand_blocks, key=lambda b: b["start"])[:3]
        evts = []
        for i, b in enumerate(blocks, 1):
            idxc = (b["start"] + b["end"]) // 2
            post_slope = np.nanmedian(np.diff(x[idxc:idxc+10]))
            dir_sign = 1 if (np.isfinite(post_slope) and post_slope >= 0) else -1
            evts.append({"pause_num": i, "idx": idxc, "pos": b["x_med"], "dir": dir_sign})
        return evts

    (pA, _, blocksA), (pB, _, blocksB) = items[0], items[1]
    if pA > pB:
        high_blocks = blocksA
        low_blocks = blocksB
    else:
        high_blocks = blocksB
        low_blocks = blocksA

    low_blocks_sorted = sorted(low_blocks, key=lambda b: b["start"])
    high_blocks_sorted = sorted(high_blocks, key=lambda b: b["start"])

    p1b = low_blocks_sorted[0]
    p3b = low_blocks_sorted[-1]
    p2b = high_blocks_sorted[0]

    p1 = {"pause_num": 1, "idx": (p1b["start"] + p1b["end"]) // 2,
          "pos": float(np.nanmedian(x[p1b["start"]:p1b["end"]+1])), "dir": 1}
    p2 = {"pause_num": 2, "idx": (p2b["start"] + p2b["end"]) // 2,
          "pos": float(np.nanmedian(x[p2b["start"]:p2b["end"]+1])), "dir": -1}
    p3 = {"pause_num": 3, "idx": (p3b["start"] + p3b["end"]) // 2,
          "pos": float(np.nanmedian(x[p3b["start"]:p3b["end"]+1])), "dir": -1}
    return [p1, p2, p3]

# -----------------------------
# SF helpers (pre/post rules)
# -----------------------------
def _approach_dir(x, idx_center, lookback=40):
    i0 = max(1, idx_center - lookback)
    diffs = np.diff(np.asarray(x[i0:idx_center+1], dtype=float))
    if diffs.size == 0:
        return 1
    return 1 if np.nanmedian(diffs) >= 0 else -1

def _nearest_in_band_index(x, band_lo, band_hi, idx_limit=None, side="before"):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    mask = np.isfinite(x) & (x >= band_lo) & (x <= band_hi)
    if idx_limit is not None:
        mask &= (idx < idx_limit) if side == "before" else (idx > idx_limit)
    cand = np.where(mask)[0]
    if cand.size == 0:
        return None
    # choose latest before or earliest after relative to the pause index
    return int(cand[-1] if side == "before" else cand[0])

def _interpolate_at_target(x, y, x_target, idx_limit=None, side="before", look=800):
    # Linear interpolation on the specified side of idx_center
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if idx_limit is None:
        return np.nan
    if side == "before":
        i0 = max(0, idx_limit - look); i1 = idx_limit - 1
        rng = range(i1, i0, -1)  # walk backward
    else:
        i0 = idx_limit + 1; i1 = min(len(x) - 1, idx_limit + look)
        rng = range(i0, i1)     # walk forward
    prev_i = None
    for i in rng:
        if prev_i is None:
            prev_i = i
            continue
        xa, xb = x[prev_i], x[i]
        if (xa - x_target) * (xb - x_target) <= 0 and np.isfinite(xa) and np.isfinite(xb):
            ya, yb = y[prev_i], y[i]
            if not (np.isfinite(ya) and np.isfinite(yb)):
                prev_i = i; continue
            if abs(xb - xa) < 1e-12:
                return float(ya)
            t = (x_target - xa) / (xb - xa)
            return float(ya + t * (yb - ya))
        prev_i = i
    return np.nan

def _value_at_exact_offset(x, y, pos, idx_center, offset, side="before"):
    """
    Sample y at x_target = round(pos + offset, 3).
    Prefer the latest sample BEFORE the pause (or earliest AFTER) in a tight band, else widen, else interpolate.
    This avoids grabbing the other pass at the same x.
    """
    x_target = round(pos + offset, 3)  # e.g., 20.000 - 0.100 => 19.900
    for tol in (0.02, 0.05, 0.10):
        lo, hi = x_target - tol, x_target + tol
        j = _nearest_in_band_index(x, lo, hi, idx_limit=idx_center, side=side)
        if j is not None:
            return float(y[j]), float(x[j]), j
    val = _interpolate_at_target(x, y, x_target, idx_limit=idx_center, side=side, look=800)
    return (float(val) if np.isfinite(val) else np.nan), float(x_target), None

def _peak_after_pause_1cm(x, y, pos, idx_center, direction, win_cm=1.0):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    idx = np.arange(len(x))
    mask = idx > idx_center
    if direction >= 0:
        mask &= (x >= pos) & (x <= pos + win_cm)
        vals = y[mask]
        return np.nan if vals.size == 0 else float(np.nanmax(vals))
    else:
        mask &= (x <= pos) & (x >= pos - win_cm)
        vals = y[mask]
        return np.nan if vals.size == 0 else float(np.nanmin(vals))

essential_sf_cols = [
    "File","Pause #","Pause X (cm)","Approach Dir","Post Dir",
    "Pre X used (cm)","Pre @ X±0.1 (g)","Peak ≤1cm After (g)","% Change vs Pre",
    "Post X used (cm)","Post @ (X-1.0) (g)","Δ Peak - Post@1cm (g)"
]

def analyze_sf_precise(x, y, file_name):
    rows = []
    events = detect_pauses_three_strict(x)
    for e in events:
        appr = _approach_dir(x, e["idx"])           # advance vs pullback approach
        pre_off = -0.1 if appr >= 0 else +0.1        # 19.900 on advance; 20.100 on pullback (snapped)
        pre_val, pre_x_used, _ = _value_at_exact_offset(x, y, e["pos"], e["idx"], pre_off, side="before")

        peak = _peak_after_pause_1cm(x, y, e["pos"], e["idx"], e["dir"], win_cm=1.0)

        pct = np.nan
        if np.isfinite(pre_val) and abs(pre_val) >= 1e-12 and np.isfinite(peak):
            pct = (abs(peak) - abs(pre_val)) / abs(pre_val) * 100.0

        post1_val = np.nan; post1_x_used = np.nan; delta = np.nan
        if e["pause_num"] == 2:                      # turnaround special
            post1_val, post1_x_used, _ = _value_at_exact_offset(x, y, e["pos"], e["idx"], -1.0, side="after")
            if np.isfinite(peak) and np.isfinite(post1_val):
                delta = float(peak - post1_val)

        rows.append({
            "File": file_name,
            "Pause #": e["pause_num"],
            "Pause X (cm)": round(float(e["pos"]), 3),
            "Approach Dir": "pos" if appr >= 0 else "neg",
            "Post Dir": "pos" if e["dir"] >= 0 else "neg",
            "Pre X used (cm)": round(pre_x_used, 3) if np.isfinite(pre_x_used) else np.nan,
            "Pre @ X±0.1 (g)": round(pre_val, 3) if np.isfinite(pre_val) else np.nan,
            "Peak ≤1cm After (g)": round(peak, 3) if np.isfinite(peak) else np.nan,
            "% Change vs Pre": round(pct, 2) if np.isfinite(pct) else np.nan,
            "Post X used (cm)": round(post1_x_used, 3) if np.isfinite(post1_x_used) else np.nan,
            "Post @ (X-1.0) (g)": round(post1_val, 3) if np.isfinite(post1_val) else np.nan,
            "Δ Peak - Post@1cm (g)": round(delta, 3) if np.isfinite(delta) else np.nan,
        })
    return rows

# -----------------------------
# DUR utilities (unchanged)
# -----------------------------
def _max_abs_with_pos(x, y):
    arr_y = np.asarray(y, dtype=float)
    arr_x = np.asarray(x, dtype=float)
    if arr_y.size == 0 or arr_x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    idx_abs = int(np.nanargmax(np.abs(arr_y)))
    max_abs = float(np.abs(arr_y[idx_abs]))
    x_at_abs = float(arr_x[idx_abs])
    pos_mask = arr_y > 0
    if not np.any(pos_mask):
        return (max_abs, x_at_abs, np.nan, np.nan)
    idx_pos = int(np.nanargmax(np.where(pos_mask, arr_y, -np.inf)))
    max_pos = float(arr_y[idx_pos])
    x_at_pos = float(arr_x[idx_pos])
    return (max_abs, x_at_abs, max_pos, x_at_pos)

def analyze_dur_first_last(runs, file_name):
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

    sf_rows_precise = []
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

            base = colors[idx]
            color_map[file.name] = base

            # Overlay plot (no forced y=0)
            if len(runs) > 1:
                for i, (cycle_label, x, y) in enumerate(runs):
                    alpha = 0.1 + 0.9 * i / (len(runs) - 1) if len(runs) > 1 else 1.0
                    rgba = f"rgba({base[0]},{base[1]},{base[2]},{alpha})"
                    show = (i == len(runs) - 1)
                    name = file.name if show else None
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=y, mode="lines",
                            line=dict(color=rgba),
                            name=name, showlegend=show, legendgroup=file.name
                        )
                    )
            else:
                _, x, y = runs[0]
                rgba = f"rgba({base[0]},{base[1]},{base[2]},1)"
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba), name=file.name))

            fname_lower = file.name.lower()

            # SF processing (use last run)
            if "sf" in fname_lower:
                _, x_sf, y_sf = runs[-1]
                sf_rows_precise.extend(analyze_sf_precise(x_sf, y_sf, file.name))

            # DUR processing
            if "dur" in fname_lower:
                dur_first_last_rows.extend(analyze_dur_first_last(runs, file.name))
                df_mag, df_pos = dur_cycle_tables(runs, file_name=file.name)
                dur_cycle_mag_tables[file.name] = df_mag
                dur_cycle_pos_tables[file.name] = df_pos

        except Exception as e:
            st.error(f"Error in file {file.name}: {e}")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Encoder/Motor",
        yaxis_title="Prox",
        title="Overlay Plot",
        height=900,
    )
    st.plotly_chart(fig, use_container_width=True)

    # SF output (precise rules)
    if sf_rows_precise:
        st.subheader("SF Pause → 1 cm Peak (precise rules)")
        df_sf = pd.DataFrame(sf_rows_precise)[essential_sf_cols]
        st.dataframe(df_sf, use_container_width=True)
        st.download_button(
            "Download SF pause→1 cm peaks (CSV)",
            data=df_sf.to_csv(index=False).encode("utf-8"),
            file_name="sf_pause_1cm_peaks_precise.csv",
            mime="text/csv",
        )

    # DUR outputs
    if dur_first_last_rows:
        st.subheader("DUR First/Last Cycle (Max |Force|)")
        df_dur = pd.DataFrame(dur_first_last_rows)
        st.dataframe(df_dur, use_container_width=True)
        st.download_button(
            "Download DUR first/last summary (CSV)",
            data=df_dur.to_csv(index=False).encode("utf-8"),
            file_name="dur_first_last_max_magnitude.csv",
            mime="text/csv",
        )

    if dur_cycle_mag_tables:
        st.subheader("DUR Per-Cycle Tables")
        combined_mag = pd.concat([v for v in dur_cycle_mag_tables.values()], ignore_index=True)
        st.markdown("**Per-cycle Max |Force| (table)**")
        st.dataframe(combined_mag, use_container_width=True)
        st.download_button(
            "Download per-cycle Max |Force| (CSV)",
            data=combined_mag.to_csv(index=False).encode("utf-8"),
            file_name="dur_per_cycle_max_magnitude.csv",
            mime="text/csv",
        )

    if dur_cycle_pos_tables:
        st.subheader("DUR Per-Cycle Peak Force Plot (positive only)")
        all_files = list(dur_cycle_pos_tables.keys())
        selected = st.multiselect("Select DUR files to compare", all_files, default=all_files)
        fig2 = go.Figure()
        for name in selected:
            dfc = dur_cycle_pos_tables[name].copy()
            dfc['Cycle'] = pd.to_numeric(dfc['Cycle'], errors='coerce')
            dfc = dfc.sort_values('Cycle')
            r, g, b = color_map.get(name, (50, 50, 50))
            fig2.add_trace(
                go.Scatter(
                    x=dfc['Cycle'], y=dfc['Peak Force (g)'],
                    mode="lines+markers", name=name,
                    line=dict(color=f"rgba({r},{g},{b},1)"),
                    marker=dict(color=f"rgba({r},{g},{b},1)"),
                )
            )
        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Cycle",
            yaxis_title="Peak Force (g)",
            title="Per-Cycle Peak Force (positive only)",
            yaxis=dict(range=[0, None], zeroline=True, zerolinewidth=1),
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

        combined_pos = pd.concat([v for v in dur_cycle_pos_tables.values()], ignore_index=True)
        st.download_button(
            "Download per-cycle Peak Force (CSV)",
            data=combined_pos.to_csv(index=False).encode("utf-8"),
            file_name="dur_per_cycle_peak_force.csv",
            mime="text/csv",
        )
