"""Generate a shareable, self-contained HTML estimate report."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html_report(*, project: str, author: str, prediction, sched: dict, pct: dict,
                      rate: float, inputs: pd.DataFrame, phases: pd.DataFrame,
                      drivers: pd.DataFrame, risks: pd.DataFrame,
                      risk_score: float, risk_level: str) -> str:
    cards = [
        ("Expected effort", f"{prediction.point:,.0f} h"),
        ("80% interval", f"{prediction.low:,.0f} - {prediction.high:,.0f} h"),
        ("Planning figure (P80)", f"{pct['P80']:,.0f} h"),
        ("Budget at P80", f"${pct['P80'] * rate:,.0f}"),
        ("Duration", f"{sched['duration_months']:.1f} months"),
        ("Risk", f"{risk_level} ({risk_score:.0f}/100)"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='lbl'>{escape(l)}</div><div class='val'>{escape(v)}</div></div>"
        for l, v in cards
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Effort estimate - {escape(project)}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;800&family=Inter:wght@300;400;600&display=swap');
body{{font-family:Inter,sans-serif;background:#f3fbf6;color:#12372a;margin:0;padding:2.5rem 1.5rem;}}
.wrap{{max-width:940px;margin:auto;}}
h1,h2{{font-family:Sora,sans-serif;color:#1b7449;}}
h1{{margin:0;font-size:2rem;}}
.meta{{color:#5b7c6d;font-size:.9rem;margin-bottom:1.8rem;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin-bottom:2rem;}}
.card{{background:#fff;border:1px solid #dcf5e3;border-radius:16px;padding:1.1rem;box-shadow:0 6px 18px rgba(34,147,90,.08);}}
.lbl{{font-size:.72rem;letter-spacing:1.2px;text-transform:uppercase;color:#5b7c6d;font-weight:600;}}
.val{{font-size:1.5rem;font-weight:700;color:#1b7449;margin-top:.25rem;}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden;
      box-shadow:0 6px 18px rgba(34,147,90,.08);margin-bottom:2rem;font-size:.9rem;}}
th{{background:#dcf5e3;text-align:left;padding:.65rem .8rem;color:#1b7449;}}
td{{padding:.55rem .8rem;border-top:1px solid #eef7f1;}}
.note{{background:#fff;border-left:4px solid #34b26b;padding:1rem 1.2rem;border-radius:0 12px 12px 0;color:#39594a;font-size:.9rem;}}
</style></head><body><div class="wrap">
<h1>Software Effort Estimate</h1>
<div class="meta">{escape(project)} &nbsp;|&nbsp; prepared by {escape(author)} &nbsp;|&nbsp;
{datetime.now():%d %B %Y, %H:%M}</div>
<div class="grid">{card_html}</div>
<h2>Delivery plan by phase</h2>{_table(phases)}
<h2>What drives this estimate</h2>{_table(drivers)}
<h2>Risk register</h2>{_table(risks)}
<h2>Model inputs</h2>{_table(inputs)}
<div class="note"><strong>How to read this.</strong> The expected effort is the model's point
estimate. The {escape(prediction.method)} gives the interval. The P80 figure is the value the
project stays under in 80% of simulated outcomes, so it is the number to commit to a sponsor -
the point estimate is roughly a coin flip. Cost assumes ${rate:,.0f} per hour.</div>
</div></body></html>"""
