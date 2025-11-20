# -*- coding: utf-8 -*-
"""
CSV Overlay Plotter (v3.5)
- NEW: For SF traces, detect 3 pauses and report the peak force within 1 cm *after* each pause in the correct motion direction.
- Baseline for % change: value immediately *before* the pause (local median in a narrow window).
- Also reports an "Alt % Change vs Post" using the median of the 1 cm window after the pause (useful for cases like x=35 → -20 spike then -5 settle).
- DUR plots unchanged (overlay: no forced y=0; per-cycle peak plot y starts at 0).
- Keeps:
  * SF robust pause detection + same-trace analysis (no cross-trace baselines)
  * DUR first/last & per-cycle tables use max |force|; per-cycle plot uses positive-only peak force
  * No idxmax/index usage
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
# SF pause detection
# -----------------------------

def _dense_x_centers(x, bin_width=0.05, topk=6):
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
    """Return up to three pause dicts with fields: pause_num, idx, pos, dir(+1/-1)."""
    x = np.asarray(x, dtype=float)
    centers = _dense_x_centers(x, bin_width=0.05, topk=6)
    if not centers:
        return []
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    # Treat lowest-x center as the "20 cm" stop (appears twice); highest-x as the mid "35 cm" stop.
    c_min = min(centers, key=lambda c: abs(c - xmin))
    c_max = max(centers, key=lambda c: c)

    # Low-x stop usually happens twice (outward and return)
    idxs_min = _indices_near(x, c_min, tol=0.25)
    chunks_min = _split_by_gaps(idxs_min, min_gap=100)
    p1 = p3 = None
    if len(chunks_min) >= 2:
        s1, e1 = chunks_min[0]
        s3, e3 = chunks_min[-1]
        p1_idx = (s1 + e1) // 2
        p3_idx = (s3 + e3) // 2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}
        p3 = {"pause_num": 3, "idx": int(p3_idx), "pos": float(x[p3_idx]), "dir": -1}
    elif len(chunks_min) == 1:
        s1, e1 = chunks_min[0]
        p1_idx = (s1 + e1) // 2
        p1 = {"pause_num": 1, "idx": int(p1_idx), "pos": float(x[p1_idx]), "dir": 1}

    # High-x stop happens once and is followed by negative direction
    idxs_max = _indices_near(x, c_max, tol=0.25)
    chunks_max = _split_by_gaps(idxs_max, min_gap=50)
    p2 = None
    if len(chunks_max) >= 1:
        s2, e2 = chunks_max[0]
        p2_idx = (s2 + e2) // 2
        p2 = {"pause_num": 2, "idx": int(p2_idx), "pos": float(x[p2_idx]), "dir": -1}

    pauses = [p for p in [p1, p2, p3] if p is not None]
    pauses = sorted(pauses, key=lambda d: d["idx"])[:3]

    # Normalize numbering/dir: first pause dir=+1, the rest dir=-1
    for i, p in enumerate(pauses, 1):
        p["pause_num"] = i
        if i == 1:
            p["dir"] = 1
        else:
            p["dir"] = -1
    return pauses

# -----------------------------
# SF peak/baseline helpers
# -----------------------------

def _peak_after_pause(x, y, pos, idx_center, direction, win_cm=1.0):
    """Extreme value within 1 cm AFTER the pause along motion direction.
    dir>=0 → search [pos, pos+1] and take max; dir<0 → search [pos-1, pos] and take min.
    Only uses samples with index > idx_center (post-pause in time)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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


def _baseline_pre_pause(x, y, pos, idx_center, w=0.2):
    """Value just BEFORE the pause: median in a narrow window [pos-w, pos] using idx < idx_center.
    Falls back to a wider window if needed; last sample before idx_center if still empty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(x))
    mask = (idx < idx_center) & (x >= pos - w) & (x <= pos + 1e-9)
    vals = y[mask]
    if vals.size == 0:
        w2 = 0.4
        mask = (idx < idx_center) & (x >= pos - w2) & (x <= pos + 1e-9)
        vals = y[mask]
    if vals.size == 0:
        before = np.where(idx < idx_center)[0]
        if before.size == 0:
            return np.nan
        j = before[-1]
        return float(y[j])
    return float(np.nanmedian(vals))


def _post_window_median(x, y, pos, idx_center, direction, win_cm=1.0):
    """Median of y *after* the pause inside the 1 cm window (robust to spikes).
    We trim 10% on both ends to de-weight the spike itself.
    Useful for cases where the trace quickly settles after the spike (e.g., -20 → -5)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(x))
    mask = idx > idx_center
    if direction >= 0:
        mask &= (x >= pos) & (x <= pos + win_cm)
    else:
        mask &= (x <= pos) & (x >= pos - win_cm)
    vals = y[mask]
    if vals.size == 0:
        return np.nan
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    lo = int(np.floor(0.1 * n))
    hi = int(np.ceil(0.9 * n))
    if hi <= lo:
        return float(np.nanmedian(vals_sorted))
    trimmed = vals_sorted[lo:hi]
    return float(np.nanmedian(trimmed))


def analyze_sf_pause_peaks(x, y, file_name):
    """Return per-pause rows with: pre-pause value, peak ≤1 cm after, and % changes."""
    out_rows = []
    if len(x) < 20 or len(y) < 20:
        return out_rows
    events = detect_three_pauses(x)
    if not events:
        return out_rows

    for e in events:
        base_pre = _baseline_pre_pause(x, y, e["pos"], e["idx"])  # just before pause
        peak = _peak_after_pause(x, y, e["pos"], e["idx"], e["dir"], win_cm=1.0)
        post_med = _post_window_median(x, y, e["pos"], e["idx"], e["dir"], win_cm=1.0)

        pct = np.nan
        if np.isfinite(base_pre) and abs(base_pre) >= 1e-9 and np.isfinite(peak):
            pct = (abs(peak) - abs(base_pre)) / abs(base_pre) * 100.0

        alt_pct = np.nan
        if np.isfinite(post_med) and abs(post_med) >= 1e-9 and np.isfinite(peak):
            alt_pct = (abs(peak) - abs(post_med)) / abs(post_med) * 100.0

        out_rows.append({
            "File": file_name,
            "Pause #": e["pause_num"],
            "Pause X (cm)": round(float(e["pos"]), 3),
            "Direction": "pos" if e["dir"] >= 0 else "neg",
            "Pre-Pause Value (g)": round(float(base_pre), 3) if np.isfinite(base_pre) else np.nan,
            "Peak ≤1cm After (g)": round(float(peak), 3) if np.isfinite(peak) else np.nan,
            "% Change vs Pre-Pause": round(float(pct), 2) if np.isfinite(pct) else np.nan,
            "Post-Window Median (g)": round(float(post_med), 3) if np.isfinite(post_med) else np.nan,
            "Alt % Change vs Post": round(float(alt_pct), 2) if np.isfinite(alt_pct) else np.nan,
        })
    return out_rows

# -----------------------------
# Durability (DUR) utilities (v3.3 behavior)
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

    sf_rows_all = []  # existing SF summary (peaks/baselines at 3 pauses)
    sf_pause_rows_all = []  # NEW: pause→1 cm peak table rows
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
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(color=rgba),
                            name=name,
                            showlegend=show,
                            legendgroup=file.name,
                        )
                    )
            else:
                _, x, y = runs[0]
                rgba = f"rgba({base[0]},{base[1]},{base[2]},1)"
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba), name=file.name))

            fname_lower = file.name.lower()

            # --- SF processing ---
            if "sf" in fname_lower:
                # analyze last run for SF summaries
                _, x_sf, y_sf = runs[-1]

                # Existing corrected SF summary (pause peaks & baselines from same trace)
                # Leaving this as-is for continuity with earlier versions
                def analyze_sf_run_corrected(x, y, file_name):
                    out = {
                        "File": file_name,
                        "Pause1 Peak (g)": np.nan,
                        "Pause1 Baseline (g)": np.nan,
                        "Pause1 %Inc": np.nan,
                        "Pause2 Peak (g)": np.nan,
                        "Pause2 Baseline (g)": np.nan,
                        "Pause2 %Inc": np.nan,
                        "Pause3 Peak (g)": np.nan,
                        "Pause3 Baseline (g)": np.nan,
                        "Pause3 %Inc": np.nan,
                    }
                    events = detect_three_pauses(x)
                    if not events or len(events) < 3:
                        return out
                    for e in events:
                        # For backward compatibility only (±0.5 cm symmetric window)
                        # Not used for the new 1 cm directional peak report
                        # Baseline: same-trace neighborhood
                        def _baseline_same_trace(x, y, pause_num, pos, idx_center):
                            x = np.asarray(x, dtype=float)
                            y = np.asarray(y, dtype=float)
                            idx = np.arange(len(x))
                            if pause_num in (1, 3):
                                before = (idx < idx_center) & (x >= pos - 1.0) & (x < pos)
                                after = (idx > idx_center) & (x > pos) & (x <= pos + 1.5)
                                sample = np.concatenate([y[before], y[after]])
                            else:
                                after_only = (idx > idx_center) & (x <= pos) & (x >= pos - 1.0)
                                sample = y[after_only]
                            if sample.size == 0:
                                return np.nan
                            return float(np.nanmean(sample))

                        # Symmetric ±0.5 cm peak window for legacy output
                        def _legacy_peak(x, y, pos, idx_center, direction):
                            x = np.asarray(x, dtype=float)
                            y = np.asarray(y, dtype=float)
                            mask = (x >= pos - 0.5) & (x <= pos + 0.5)
                            vals = y[mask]
                            return float(np.nanmax(vals)) if direction >= 0 else float(np.nanmin(vals))

                        peak = _legacy_peak(x, y, e["pos"], e["idx"], e["dir"]) 
                        base = _baseline_same_trace(x, y, e["pause_num"], e["pos"], e["idx"]) 
                        pct = np.nan
                        if np.isfinite(base) and abs(base) >= 1e-9 and np.isfinite(peak):
                            pct = (abs(peak) - abs(base)) / abs(base) * 100.0
                        k = e["pause_num"]
                        out[f"Pause{k} Peak (g)"] = round(float(peak), 3) if np.isfinite(peak) else np.nan
                        out[f"Pause{k} Baseline (g)"] = round(float(base), 3) if np.isfinite(base) else np.nan
                        out[f"Pause{k} %Inc"] = round(float(pct), 2) if np.isfinite(pct) else np.nan
                    return out

                sf_rows_all.append(analyze_sf_run_corrected(x_sf, y_sf, file.name))

                # NEW: 1 cm directional peak report per pause
                sf_pause_rows_all.extend(analyze_sf_pause_peaks(x_sf, y_sf, file.name))

            # --- DUR processing ---
            if "dur" in fname_lower:
                dur_first_last_rows.extend(analyze_dur_first_last(runs, file.name))
                df_mag, df_pos = dur_cycle_tables(runs, file.name)
                dur_cycle_mag_tables[file.name] = df_mag
                dur_cycle_pos_tables[file.name] = df_pos

        except Exception as e:
            st.error(f"Error in file {file.name}: {e}")

    # Overlay figure
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Encoder/Motor",
        yaxis_title="Prox",
        title="Overlay Plot",
    )
    st.plotly_chart(fig, use_container_width=True, height=900)

    # --- SF outputs ---
    if sf_pause_rows_all:
        st.subheader("SF Pause → 1 cm Peak (direction-aware)")
        df_sf_peaks = pd.DataFrame(sf_pause_rows_all)
        st.dataframe(df_sf_peaks, use_container_width=True)
        st.download_button(
            "Download SF pause→1 cm peaks (CSV)",
            data=df_sf_peaks.to_csv(index=False).encode("utf-8"),
            file_name="sf_pause_1cm_peaks.csv",
            mime="text/csv",
        )

    if sf_rows_all:
        st.subheader("SF Static-Friction Summary (legacy window)")
        df_sf = pd.DataFrame(sf_rows_all)
        st.dataframe(df_sf, use_container_width=True)
        st.download_button(
            "Download SF legacy summary (CSV)",
            data=df_sf.to_csv(index=False).encode("utf-8"),
            file_name="sf_static_friction_summary.csv",
            mime="text/csv",
        )

    # --- DUR outputs ---
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
                    x=dfc['Cycle'],
                    y=dfc['Peak Force (g)'],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=f"rgba({r},{g},{b},1)"),
                    marker=dict(color=f"rgba({r},{g},{b},1)"),
                )
            )
        # Force y-axis to start at 0 for DUR plot only
        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Cycle",
            yaxis_title="Peak Force (g)",
            title="Per-Cycle Peak Force (positive only)",
            yaxis=dict(range=[0, None], zeroline=True, zerolinewidth=1),
        )
        st.plotly_chart(fig2, use_container_width=True, height=500)

        combined_pos = pd.concat([v for v in dur_cycle_pos_tables.values()], ignore_index=True)
        st.download_button(
            "Download per-cycle Peak Force (CSV)",
            data=combined_pos.to_csv(index=False).encode("utf-8"),
            file_name="dur_per_cycle_peak_force.csv",
            mime="text/csv",
        )
