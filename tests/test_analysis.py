"""
Run:  pytest -q
Uses synthetic files so the repo needs no real test data.
Optional: drop real tester CSVs in tests/data/ (git-ignored) and they get smoke-tested too.
"""
from pathlib import Path

import numpy as np
import pytest

from advpb.analysis import analyze, sf_breakaways
from advpb.parsing import lot_sample_from_filename, parse_file, type_from_filename

HEADER = """#Unique ID:,'
User ID:,operator
Date:,'6/9/26
Time:,14:42
Recipe:,{recipe}
Recipe Parameters
Units,Force:,g,Distance:,cm,Speed:,cm/min
Encoder:,Enabled
Cycles:,{cycles},Steps:,{nsteps}
Sam. Rate:,100 ms
-
Step,Speed,Distance,Delay
{steps}
#
Encoder,Motor,Prox,Time(ms),Cycle,Step
"""


def _write(rows, recipe, steps, cycles=1):
    y = np.array([r[1] for r in rows])
    txt = HEADER.format(recipe=recipe, cycles=cycles, nsteps=len(steps),
                        steps="\n".join(",".join(map(str, s)) for s in steps))
    txt += "\n".join(f"{x:.5f},{x:.5f},{f:.4f},{i*100},{c},{s}" for i, (x, f, c, s) in enumerate(rows))
    i_max, i_min = int(y.argmax()), int(y.argmin())
    txt += (f"\nLoad Cell,Max Force,Max Pos,Min Force,Min Pos,Avg Force,Tare Value\n"
            f"Prox 1,{y.max():.4f},{rows[i_max][0]:.5f},{y.min():.4f},{rows[i_min][0]:.5f},0,0\n#\n")
    return txt


def synthetic_sf():
    """0->20 cm, 30 s dwell, ->31 cm, 30 s dwell, ->20 cm, 30 s dwell, ->0. 0.05 cm/sample."""
    rows, x = [], 0.0
    plan = [(1, 20.0, +1, 20.0, 100.0), (2, 31.0, +1, 30.0, -30.0), (3, 20.0, -1, -8.0, -50.0), (4, 0.0, -1, None, None)]
    steady = {1: 20.0, 2: 30.0, 3: -8.0, 4: -8.0}
    peaks = {2: 100.0, 3: -30.0, 4: -50.0}
    for step, target, d, _, _ in plan:
        first = True
        while (x < target) if d > 0 else (x > target):
            x = min(x + 0.05, target) if d > 0 else max(x - 0.05, target)
            f = peaks[step] if (first and step in peaks) else steady[step] * (1 if step != 2 else 1)
            if step == 3 and not first:
                f = -10.0
            rows.append((x, f, 1, step))
            first = False
        if step < 4:
            rows += [(x, steady[step] if step != 2 else 30.0, 1, step)] * 300  # 30 s dwell
    steps = [(1, 30.0, 20.0, 30.0), (2, 30.0, 11.0, 30.0), (3, 30.0, -11.0, 30.0), (4, 30.0, -20.0, 0.0)]
    return _write(rows, "Guidewire Static Friction", steps)


def synthetic_dur(cycles=3):
    rows = []
    for c in range(1, cycles + 1):
        xs = np.arange(0, 31.0001, 0.05)
        rows += [(x, 50 - c + 5 * np.sin(x), c, 1) for x in xs]
        rows += [(x, -5 - 0.1 * c, c, 2) for x in xs[::-1]]
    rows.append((0.0, -2.0, cycles + 1, 1))  # trailing fragment like real exports
    return _write(rows, "Guidewire Durability", [(1, 30.0, 31.0, 0.0), (2, 30.0, -31.0, 0.0)], cycles)


def test_filename_helpers():
    assert lot_sample_from_filename("Synchro ln 999524 S#6 SF characterization.csv") == ("999524", "6")
    assert type_from_filename("x ln 1 S#2 DUR.csv") == "DUR"
    assert type_from_filename("Sfoo.csv") == "Unknown"


def test_sf_three_breakaways():
    tf = parse_file("Wire ln 111 S#1 SF.csv", synthetic_sf())
    assert tf.test_type == "SF" and tf.footer["Max Force"] == pytest.approx(100.0)
    rows = sf_breakaways(tf)
    assert [r["Direction"][0] for r in rows] == ["A", "P", "P"]
    assert [r["Event"].split(" @")[0] for r in rows] == ["Advance restart", "Reversal", "Pullback restart"]
    assert [r["Breakaway Peak (g)"] for r in rows] == pytest.approx([100.0, -30.0, -50.0])
    assert all(r["Dwell (s)"] >= 29 for r in rows)
    assert rows[2]["Dwell Force (g)"] == pytest.approx(-8.0)   # pullback pass, not the advance pass


def test_dur_cycles_and_footer():
    tf = parse_file("Wire ln 111 S#1 DUR.csv", synthetic_dur(3))
    out = analyze([tf])
    assert len(out["DUR Per-Cycle"]) == 3                      # trailing fragment dropped
    assert out["Max Force Summary"]["Footer Check"].iloc[0] == "OK"
    fl = out["DUR First-Last"].iloc[0]
    assert fl["First Cycle"] == 1 and fl["Last Cycle"] == 3


REAL = sorted((Path(__file__).parent / "data").glob("*.csv"))


@pytest.mark.skipif(not REAL, reason="no real files in tests/data")
@pytest.mark.parametrize("path", REAL, ids=lambda p: p.name)
def test_real_files_match_footer(path):
    tf = parse_file(path.name, path.read_bytes())
    out = analyze([tf])
    assert out["Max Force Summary"]["Footer Check"].iloc[0] == "OK"
    if tf.test_type == "SF":
        assert len(out["SF Breakaway"]) == 3
