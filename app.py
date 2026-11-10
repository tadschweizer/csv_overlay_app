# -*- coding: utf-8 -*-
"""
ADV/PB Track Tester Analyzer (v4.0.0)  -  Streamlit UI

Analysis logic lives in the `advpb` package; this file is UI only.
Run:  streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from advpb import __version__
from advpb.analysis import analyze
from advpb.export import to_excel, to_html
from advpb.parsing import parse_file
from advpb import plots

st.set_page_config(page_title="ADV/PB Analyzer", layout="wide")
st.title("ADV/PB Track Tester Analyzer")
st.caption(f"v{__version__} · Force sign: + = advance/push, − = pullback/pull")


# ------------------------------------------------------------------
# Cached work - files are parsed/analyzed once, not on every widget click
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=200)
def cached_parse(name: str, raw: bytes):
    return parse_file(name, raw)


@st.cache_data(show_spinner="Analyzing...")
def cached_analyze(payload: tuple):
    """payload = tuple of (name, raw bytes, test type). Hashable -> cache key."""
    files = []
    for name, raw, ttype in payload:
        tf = cached_parse(name, raw)
        tf.test_type = ttype
        files.append(tf)
    return files, analyze(files)


@st.cache_data(show_spinner="Building export files...")
def cached_exports(payload: tuple):
    files, tables = cached_analyze(payload)
    labels = {f.name: f"ln {f.lot} S#{f.sample}" for f in files}
    figs = [
        plots.sf_breakaway_bar(tables["SF Breakaway"]),
        plots.dur_trend(tables["DUR Per-Cycle"], labels),
        plots.overlay([f for f in files if f.test_type == "SF"], "SF Force vs Position"),
        plots.overlay([f for f in files if f.test_type == "LUB"], "LUB Force vs Position"),
    ]
    return to_excel(tables), to_html(tables, figs)


def show(df: pd.DataFrame, **kw):
    if df is None or df.empty:
        st.info("No data for this test type.")
    else:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True, **kw)


# ------------------------------------------------------------------
# Upload + type detection
# ------------------------------------------------------------------
uploaded = st.file_uploader("Select tester CSV files (SF, LUB, DUR)", type="csv", accept_multiple_files=True)

with st.expander("How to use", expanded=not uploaded):
    st.markdown(
        "- Upload any mix of Static Friction, Lubricity and Durability CSV exports.\n"
        "- Test type is read from the **Recipe** line in each file (filename is the fallback).\n"
        "- Lot and sample are read from the filename (`ln 999524 S#6`).\n"
        "- **SF breakaways** are located from the tester's `Step` column: each restart after a dwell "
        "gives the peak within 1 cm of travel, the dwell force, the steady force 1–3 cm after, and Peak − Steady.\n"
        "- **Footer Check** compares the computed max/min to the tester's own footer values.\n"
        "- Download the Excel workbook and HTML report at the bottom."
    )

if not uploaded:
    st.stop()

parsed, errors = [], []
for f in uploaded:
    try:
        parsed.append((f.name, f.getvalue(), cached_parse(f.name, f.getvalue())))
    except Exception as e:  # noqa: BLE001
        errors.append({"File": f.name, "Error": str(e)})
if errors:
    st.error("Some files could not be read:")
    st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)

st.subheader("Files")
types = {}
unknown = [p for p in parsed if p[2].test_type == "Unknown"]
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(pd.DataFrame([{"File": n, "Lot": tf.lot, "Sample": tf.sample, "Detected": tf.test_type,
                                "Recipe": tf.meta.get("Recipe", ""), "Warnings": " | ".join(tf.warnings)}
                               for n, _, tf in parsed]), hide_index=True, use_container_width=True)
with c2:
    if unknown:
        st.markdown("**Assign type for unrecognized files**")
    for n, _, tf in parsed:
        types[n] = tf.test_type
        if tf.test_type == "Unknown":
            types[n] = st.selectbox(n, ["Unknown", "SF", "LUB", "DUR"], key=f"type_{n}")

payload = tuple((n, raw, types[n]) for n, raw, _ in parsed if types[n] != "Unknown")
if not payload:
    st.warning("No files with a recognized test type.")
    st.stop()

files, tables = cached_analyze(payload)
labels = {f.name: f"ln {f.lot} S#{f.sample}" for f in files}

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
st.divider()
t_sum, t_sf, t_lub, t_dur, t_plot, t_info = st.tabs(
    ["Max Force Summary", "Static Friction", "Lubricity", "Durability", "Overlay Plots", "File Info"])

with t_sum:
    st.markdown("Overall max push and max pull for every file, all test types.")
    summ = tables["Max Force Summary"]
    show(summ)
    if not summ.empty and (summ["Footer Check"] != "OK").any():
        st.warning("One or more files do not match the tester footer max/min - check the File Info tab.")

with t_sf:
    sf = tables["SF Breakaway"]
    show(sf)
    if not sf.empty:
        st.plotly_chart(plots.sf_breakaway_bar(sf), use_container_width=True)

with t_lub:
    show(tables["LUB"])

with t_dur:
    st.markdown("**First vs last cycle**")
    show(tables["DUR First-Last"])
    cyc = tables["DUR Per-Cycle"]
    if not cyc.empty:
        sel = st.multiselect("Files to plot", list(labels), default=[f for f in labels if f in set(cyc["File"])],
                             format_func=lambda n: labels[n], key="dur_sel")
        st.plotly_chart(plots.dur_trend(cyc[cyc["File"].isin(sel)], labels), use_container_width=True)
        with st.expander("Per-cycle table"):
            show(cyc)

with t_plot:
    present = sorted({f.test_type for f in files})
    ttype = st.radio("Test type", present, horizontal=True)
    choices = [f for f in files if f.test_type == ttype]
    sel = st.multiselect("Files", [f.name for f in choices], default=[f.name for f in choices],
                         format_func=lambda n: labels.get(n, n), key="ov_sel")
    st.plotly_chart(plots.overlay([f for f in choices if f.name in sel], f"{ttype} Force vs Position"),
                    use_container_width=True)

with t_info:
    show(tables["File Info"])

# ------------------------------------------------------------------
# Downloads
# ------------------------------------------------------------------
st.divider()
st.subheader("Downloads")
xlsx, html_bytes = cached_exports(payload)
stamp = datetime.now().strftime("%Y%m%d_%H%M")
d1, d2 = st.columns(2)
d1.download_button("Excel workbook (all tables)", xlsx, f"advpb_analysis_{stamp}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d2.download_button("HTML report (tables + plots)", html_bytes, f"advpb_report_{stamp}.html",
                   "text/html", use_container_width=True)
