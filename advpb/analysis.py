"""
Analysis for SF / LUB / DUR track-tester files.

Sign convention (tester native): + = advance (push), - = pullback (pull).
"Max pull" is reported as the most negative force.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import TestFile

FOOTER_TOL_G = 0.01        # tolerance for cross-check vs tester footer
PEAK_WINDOW_CM = 1.0       # breakaway peak searched within this travel after restart
STEADY_WINDOW_CM = (1.0, 3.0)  # steady (kinetic) force window after restart
HOLD_WINDOW_S = 2.0        # dwell force = median of last N s before restart
MIN_DWELL_S = 1.0          # stop shorter than this is not treated as a dwell
STATIONARY_CM = 0.02


def _r(v, n=3):
    return round(float(v), n) if v is not None and np.isfinite(v) else np.nan


# ------------------------------------------------------------------
# Overall max push / pull (all tests)
# ------------------------------------------------------------------
def max_push_pull(x: np.ndarray, y: np.ndarray) -> dict:
    i_max, i_min = int(np.nanargmax(y)), int(np.nanargmin(y))
    return {
        "Max Push (g)": _r(y[i_max]), "Max Push Pos (cm)": _r(x[i_max]),
        "Max Pull (g)": _r(y[i_min]), "Max Pull Pos (cm)": _r(x[i_min]),
        "Max |Force| (g)": _r(max(abs(y[i_max]), abs(y[i_min]))),
    }


def footer_check(tf: TestFile, mx: float, mn: float) -> str:
    f = tf.footer
    if not f or "Max Force" not in f:
        return "No footer"
    ok = abs(f["Max Force"] - mx) <= FOOTER_TOL_G and abs(f.get("Min Force", mn) - mn) <= FOOTER_TOL_G
    return "OK" if ok else f"MISMATCH (tester max {f['Max Force']:.3f} / min {f.get('Min Force', np.nan):.3f})"


def file_summary(tf: TestFile) -> dict:
    x = tf.data[tf.x_col].to_numpy(float)
    y = tf.data[tf.y_col].to_numpy(float)
    mp = max_push_pull(x, y)
    return {
        "File": tf.name, "Lot": tf.lot, "Sample": tf.sample, "Test": tf.test_type,
        **mp,
        "Footer Check": footer_check(tf, mp["Max Push (g)"], mp["Max Pull (g)"]),
    }


def file_info(tf: TestFile) -> dict:
    runs = tf.runs()
    return {
        "File": tf.name, "Lot": tf.lot, "Sample": tf.sample, "Test": tf.test_type,
        "Recipe": tf.meta.get("Recipe", ""), "Date": tf.meta.get("Date", ""),
        "Time": tf.meta.get("Time", ""), "User ID": tf.meta.get("User ID", ""),
        "Recipe Cycles": tf.meta.get("Cycles", ""), "Cycles Found": len(runs),
        "Sample Rate": tf.meta.get("Sam. Rate", ""), "Rows": len(tf.data),
        "Tare (g)": tf.footer.get("Tare Value", np.nan),
        "Warnings": " | ".join(tf.warnings),
    }


# ------------------------------------------------------------------
# SF: step-based breakaway analysis
# ------------------------------------------------------------------
def _direction(x, start, end, x0=None):
    """Sign of travel after index `start` relative to x0 (first move > 0.05 cm)."""
    x0 = x[start] if x0 is None else x0
    for j in range(start, end):
        if abs(x[j] - x0) > 0.05:
            return 1 if x[j] > x0 else -1
    return 0


def sf_breakaways(tf: TestFile) -> list[dict]:
    """
    One row per restart-after-dwell. Restarts are located from the tester's Step
    column (step increments on the sample motion resumes), so no position guessing.
    """
    rows = []
    for run in tf.runs():
        x, y, t, s = run.x, run.y, run.t, run.step
        if s is None:
            continue
        if t is None:
            t = np.arange(len(x)) * 100.0
        trans = np.where(np.diff(s) != 0)[0] + 1
        bounds = list(trans) + [len(x)]
        for n, i in enumerate(trans):
            seg_end = bounds[n + 1]
            x0 = x[i - 1]          # dwell position = last sample of the previous step
            # dwell: contiguous stationary samples before restart
            k = i - 1
            while k > 0 and abs(x[k - 1] - x0) <= STATIONARY_CM:
                k -= 1
            dwell_s = (t[i] - t[k]) / 1000.0
            if dwell_s < MIN_DWELL_S:
                continue
            d_after = _direction(x, i, seg_end, x0)
            d_before = -_direction(x[::-1], len(x) - 1 - k, len(x)) if k > 0 else 0
            if d_after == 0:
                continue
            kind = "Reversal" if d_before and d_before != d_after else (
                "Advance restart" if d_after > 0 else "Pullback restart")

            idx = np.arange(i, seg_end)
            travel = (x[idx] - x0) * d_after
            pk_idx = idx[(travel >= -STATIONARY_CM) & (travel <= PEAK_WINDOW_CM)]
            st_idx = idx[(travel >= STEADY_WINDOW_CM[0]) & (travel <= STEADY_WINDOW_CM[1])]
            hold_idx = np.arange(k, i)[t[k:i] >= t[i] - HOLD_WINDOW_S * 1000]

            if pk_idx.size:
                j = pk_idx[np.argmax(y[pk_idx])] if d_after > 0 else pk_idx[np.argmin(y[pk_idx])]
                peak, peak_x = y[j], x[j]
            else:
                peak = peak_x = np.nan
            steady = np.median(y[st_idx]) if st_idx.size else np.nan
            hold = np.median(y[hold_idx]) if hold_idx.size else np.nan

            rows.append({
                "File": tf.name, "Lot": tf.lot, "Sample": tf.sample,
                "Event #": len([r for r in rows if r["File"] == tf.name]) + 1,
                "Event": f"{kind} @ {x0:.0f} cm",
                "Direction": "Advance (+)" if d_after > 0 else "Pullback (-)",
                "Dwell X (cm)": _r(x0), "Dwell (s)": _r(dwell_s, 1),
                "Dwell Force (g)": _r(hold),
                "Breakaway Peak (g)": _r(peak), "Peak X (cm)": _r(peak_x),
                "|Breakaway Peak| (g)": _r(abs(peak)) if np.isfinite(peak) else np.nan,
                f"Steady {STEADY_WINDOW_CM[0]:g}-{STEADY_WINDOW_CM[1]:g} cm (g)": _r(steady),
                "Peak - Steady (g)": _r(peak - steady),
            })
    return rows


# ------------------------------------------------------------------
# LUB / DUR: per-cycle push & pull
# ------------------------------------------------------------------
def per_cycle(tf: TestFile) -> pd.DataFrame:
    rows = []
    for run in tf.runs():
        mp = max_push_pull(run.x, run.y)
        rows.append({"File": tf.name, "Lot": tf.lot, "Sample": tf.sample,
                     "Test": tf.test_type, "Cycle": run.cycle, **mp})
    return pd.DataFrame(rows)


def dur_first_last(cyc: pd.DataFrame) -> dict:
    if cyc.empty:
        return {}
    cyc = cyc.sort_values("Cycle")
    f, l = cyc.iloc[0], cyc.iloc[-1]

    def pct(a, b):
        return _r((b - a) / abs(a) * 100, 2) if abs(a) > 1e-9 else np.nan

    return {
        "File": f["File"], "Lot": f["Lot"], "Sample": f["Sample"],
        "First Cycle": int(f["Cycle"]), "Last Cycle": int(l["Cycle"]),
        "First Max Push (g)": f["Max Push (g)"], "Last Max Push (g)": l["Max Push (g)"],
        "Push % Change": pct(f["Max Push (g)"], l["Max Push (g)"]),
        "First Max Pull (g)": f["Max Pull (g)"], "Last Max Pull (g)": l["Max Pull (g)"],
        "Pull % Change": pct(abs(f["Max Pull (g)"]), abs(l["Max Pull (g)"])),
        "Max Push All Cycles (g)": cyc["Max Push (g)"].max(),
        "Max Pull All Cycles (g)": cyc["Max Pull (g)"].min(),
    }


# ------------------------------------------------------------------
# Run everything
# ------------------------------------------------------------------
def analyze(files: list[TestFile]) -> dict[str, pd.DataFrame]:
    summary, info, sf, cyc_tables, fl = [], [], [], [], []
    for tf in files:
        info.append(file_info(tf))
        if tf.test_type not in ("SF", "LUB", "DUR"):
            continue
        summary.append(file_summary(tf))
        if tf.test_type == "SF":
            sf.extend(sf_breakaways(tf))
        else:
            c = per_cycle(tf)
            cyc_tables.append(c)
            if tf.test_type == "DUR":
                fl.append(dur_first_last(c))
    cyc = pd.concat(cyc_tables, ignore_index=True) if cyc_tables else pd.DataFrame()
    sort = ["Test", "Lot", "Sample"]
    out = {
        "Max Force Summary": pd.DataFrame(summary),
        "SF Breakaway": pd.DataFrame(sf),
        "LUB": cyc[cyc["Test"] == "LUB"].drop(columns=["Cycle"]) if not cyc.empty else pd.DataFrame(),
        "DUR First-Last": pd.DataFrame(fl),
        "DUR Per-Cycle": cyc[cyc["Test"] == "DUR"] if not cyc.empty else pd.DataFrame(),
        "File Info": pd.DataFrame(info),
    }
    if not out["Max Force Summary"].empty:
        out["Max Force Summary"] = out["Max Force Summary"].sort_values(sort).reset_index(drop=True)
    return out
