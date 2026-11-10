"""Excel workbook and standalone HTML report export."""
from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import __version__

HEADER_FILL = PatternFill("solid", start_color="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def to_excel(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet, df in tables.items():
            if df is None or df.empty:
                continue
            df.reset_index(drop=True).to_excel(xw, sheet_name=sheet[:31], index=False)
            ws = xw.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for c, col in enumerate(df.columns, 1):
                cell = ws.cell(row=1, column=c)
                cell.fill, cell.font = HEADER_FILL, HEADER_FONT
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                width = max(len(str(col)) * 0.9, df[col].astype(str).str.len().max() if len(df) else 0)
                ws.column_dimensions[get_column_letter(c)].width = min(max(10, width + 2), 55)
                if pd.api.types.is_float_dtype(df[col]):
                    for r in range(2, len(df) + 2):
                        ws.cell(row=r, column=c).number_format = "0.000"
        meta = pd.DataFrame({"Item": ["Generated", "Tool version", "Sign convention"],
                             "Value": [datetime.now().strftime("%Y-%m-%d %H:%M"), __version__,
                                       "+ = advance/push, - = pullback/pull (tester native)"]})
        meta.to_excel(xw, sheet_name="About", index=False)
    return buf.getvalue()


def to_html(tables: dict[str, pd.DataFrame], figures: list, title="ADV/PB Analysis Report") -> bytes:
    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}}
h1{{color:#1F3864}} h2{{color:#1F3864;border-bottom:2px solid #1F3864;padding-bottom:4px;margin-top:32px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}}
th{{background:#1F3864;color:#fff;padding:4px 8px;text-align:left}}
td{{border:1px solid #ccc;padding:3px 8px}} tr:nth-child(even) td{{background:#f4f6fa}}
.meta{{color:#666;font-size:12px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; tool v{__version__} &middot;
Sign convention: + = advance/push, &minus; = pullback/pull</p>"""]
    first = True
    for fig in figures:
        if fig is not None and len(fig.data):
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
            first = False
    for name, df in tables.items():
        if df is None or df.empty or name == "DUR Per-Cycle":
            continue   # per-cycle detail lives in the Excel file
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(df.reset_index(drop=True).to_html(index=False, float_format=lambda v: f"{v:.3f}",
                                                       na_rep="", border=0))
    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")
