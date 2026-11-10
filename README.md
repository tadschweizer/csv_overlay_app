# ADV/PB Track Tester Analyzer

Streamlit app for IDTE track-tester exports: **Static Friction (SF)**, **Lubricity (LUB)**, **Durability (DUR)**.

## What it reports

| Test | Output |
|---|---|
| All | Max push (+) and max pull (−) with positions, max \|force\|, cross-check vs tester footer |
| SF | Per restart-after-dwell: breakaway peak (≤1 cm travel), dwell force (last 2 s), steady force (1–3 cm), Peak − Steady |
| LUB | Max push / pull with positions |
| DUR | Per-cycle max push / pull, first vs last cycle % change, overall max |

Exports: **Excel workbook** (one sheet per table + About) and a **standalone HTML report** (tables + plots).

### How it works
- **Test type** comes from the `Recipe:` line (filename fallback, manual override for unknowns).
- **Lot / sample** come from the filename, e.g. `Synchro ln 999524 S#6 SF characterization.csv`.
- **SF breakaways** are located from the tester's `Step` column; the step increments on the sample where motion resumes, so no position guessing.
- **Footer Check** compares computed max/min to the tester's own `Max Force` / `Min Force`.
- Sign convention is the tester's: **+ = advance/push, − = pullback/pull**.

Tunable constants (peak window, steady window, dwell window) are at the top of `advpb/analysis.py`.

## Run
**Codespaces:** open the repo → *Code → Codespaces → Create*. The app starts automatically on port 8501.

**Local:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test
```bash
pytest -q
```
Synthetic files are built in the tests. To also check real exports, copy CSVs into `tests/data/` (git-ignored).

## Layout
```
app.py                 Streamlit UI only
advpb/parsing.py       CSV header / data / footer parsing
advpb/analysis.py      SF breakaway, max force, per-cycle
advpb/plots.py         Plotly figures
advpb/export.py        Excel + HTML export
tests/                 pytest
.devcontainer/         Codespaces config
```

## Changelog
**4.0.0**
- SF analysis rebuilt on the `Step` column; replaced the Pre/Post table (Pause #3 sampled the wrong pass).
- Added dwell force, steady force, Peak − Steady, and per-file max push/pull summary for all tests.
- Test type from recipe; lot/sample parsed from filename; footer cross-check.
- Excel + HTML export.
- Parsing/analysis cached (no re-reading files on every click); long DUR plots decimated.
- Split into `advpb` package with pytest suite.
