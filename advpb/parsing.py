"""
Parsing for IDTE track-tester CSV exports.

File layout (all three recipes):
    metadata header  (User ID, Date, Time, Recipe, Cycles, Sam. Rate, ...)
    step table       (Step,Speed,Distance,Delay)
    '#'
    data block       (Encoder,Motor,Prox,Time(ms),Cycle,Step)
    footer           (Load Cell,Max Force,Max Pos,Min Force,Min Pos,Avg Force,Tare Value)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO

import numpy as np
import pandas as pd

TEST_TYPES = ("SF", "LUB", "DUR")


@dataclass
class Run:
    cycle: int
    x: np.ndarray          # position (cm)
    y: np.ndarray          # force (g)
    t: np.ndarray | None   # time (ms)
    step: np.ndarray | None


@dataclass
class TestFile:
    name: str
    data: pd.DataFrame
    meta: dict = field(default_factory=dict)
    steps: pd.DataFrame = field(default_factory=pd.DataFrame)   # recipe step table
    footer: dict = field(default_factory=dict)
    x_col: str = ""
    y_col: str = ""
    test_type: str = "Unknown"
    lot: str = ""
    sample: str = ""
    warnings: list = field(default_factory=list)

    def runs(self) -> list[Run]:
        """Split data by Cycle; drops fragments with < 5 samples (e.g. a trailing cycle-25 row)."""
        df = self.data
        t = "Time(ms)" if "Time(ms)" in df.columns else None
        s = "Step" if "Step" in df.columns else None
        groups = df.groupby("Cycle", sort=True) if "Cycle" in df.columns else [(1, df)]
        out = []
        for cyc, sub in groups:
            if len(sub) < 5:
                continue
            out.append(Run(
                cycle=int(cyc),
                x=sub[self.x_col].to_numpy(float),
                y=sub[self.y_col].to_numpy(float),
                t=sub[t].to_numpy(float) if t else None,
                step=sub[s].to_numpy(int) if s else None,
            ))
        return out


def _num(v):
    try:
        return float(str(v).strip().lstrip("'"))
    except (TypeError, ValueError):
        return np.nan


def type_from_recipe(recipe: str) -> str:
    r = (recipe or "").lower()
    if "static" in r or "friction" in r:
        return "SF"
    if "lubric" in r:
        return "LUB"
    if "durab" in r:
        return "DUR"
    return "Unknown"


def type_from_filename(name: str) -> str:
    """Token-based so words like 'Sf' inside other words don't false-match."""
    tokens = re.split(r"[^a-z]+", name.lower())
    for tt in TEST_TYPES:
        if tt.lower() in tokens:
            return tt
    return "Unknown"


def lot_sample_from_filename(name: str) -> tuple[str, str]:
    lot = re.search(r"(?:ln|lot)\s*#?\s*(\d+)", name, re.I)
    smp = re.search(r"S\s*#\s*(\d+)", name, re.I)
    return (lot.group(1) if lot else ""), (smp.group(1) if smp else "")


def parse_file(name: str, raw: bytes | str) -> TestFile:
    text = raw.decode("utf-8-sig", errors="ignore") if isinstance(raw, bytes) else raw
    lines = text.splitlines()

    # ---- locate data header ----
    hdr = next((i for i, ln in enumerate(lines)
                if "prox" in ln.lower() and ("encoder" in ln.lower() or "motor" in ln.lower())), None)
    if hdr is None:
        raise ValueError("Could not find a data header containing 'Prox' and 'Encoder'/'Motor'.")

    # ---- metadata + recipe step table ----
    meta, step_rows, in_steps = {}, [], False
    for ln in lines[:hdr]:
        cells = [c.strip() for c in ln.split(",")]
        key = cells[0].lstrip("#").rstrip(":").strip()
        if key.lower() == "step" and len(cells) >= 4:
            in_steps = True
            continue
        if in_steps:
            if cells[0] in ("#", "-", ""):
                in_steps = False
                continue
            step_rows.append(cells[:4])
            continue
        if len(cells) >= 2 and key:
            meta[key] = cells[1].lstrip("'")
            # "Cycles:,24,Steps:,2" carries two pairs
            for j in range(2, len(cells) - 1, 2):
                k2 = cells[j].rstrip(":").strip()
                if k2:
                    meta[k2] = cells[j + 1].lstrip("'")
    steps = pd.DataFrame(step_rows, columns=["Step", "Speed", "Distance", "Delay"]).apply(pd.to_numeric, errors="coerce") \
        if step_rows else pd.DataFrame(columns=["Step", "Speed", "Distance", "Delay"])

    # ---- data block (stop at footer) ----
    columns = [c.strip() for c in lines[hdr].split(",")]
    data_lines, footer = [], {}
    k = hdr + 1
    while k < len(lines):
        ln = lines[k]
        if ln.lower().startswith("load cell"):
            fkeys = [c.strip() for c in ln.split(",")]
            if k + 1 < len(lines):
                fvals = [c.strip() for c in lines[k + 1].split(",")]
                footer = {fk: _num(fv) for fk, fv in zip(fkeys[1:], fvals[1:])}
            break
        if ln.strip() and not ln.startswith("#"):
            data_lines.append(ln)
        k += 1
    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=columns,
                     usecols=range(len(columns)), on_bad_lines="skip", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")

    enc = [c for c in df.columns if "encoder" in c.lower()]
    mot = [c for c in df.columns if "motor" in c.lower()]
    prox = [c for c in df.columns if "prox" in c.lower()]
    x_col = enc[0] if enc else mot[0]
    y_col = prox[0]
    df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)

    tf = TestFile(name=name, data=df, meta=meta, steps=steps, footer=footer, x_col=x_col, y_col=y_col)
    tf.lot, tf.sample = lot_sample_from_filename(name)
    tf.test_type = type_from_recipe(meta.get("Recipe", ""))
    if tf.test_type == "Unknown":
        tf.test_type = type_from_filename(name)
    elif type_from_filename(name) not in ("Unknown", tf.test_type):
        tf.warnings.append(f"Filename suggests {type_from_filename(name)} but recipe is "
                           f"'{meta.get('Recipe')}'; using recipe ({tf.test_type}).")
    if "Step" not in df.columns and tf.test_type == "SF":
        tf.warnings.append("No 'Step' column - SF breakaway detection needs it.")
    return tf
