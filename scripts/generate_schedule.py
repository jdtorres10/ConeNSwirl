#!/usr/bin/env python3
"""
Single source of truth: data/schedule.json
Generates:
  - knowledge_base/schedule.txt (chatbot RAG)
  - Replaces the schedule <table> in index.html (between HTML markers)

Run from repo root:
  python3 scripts/generate_schedule.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "schedule.json"
KB_PATH = REPO_ROOT / "knowledge_base" / "schedule.txt"
INDEX_PATH = REPO_ROOT / "index.html"

BEGIN = "<!-- CNS_SCHEDULE_TABLE_BEGIN -->"
END = "<!-- CNS_SCHEDULE_TABLE_END -->"
TABLE_START = '<table class="w-full text-left text-sm">'


def _required_stop_keys(row: dict, idx: int) -> None:
    keys = (
        "schedule_heading",
        "day_short",
        "date_short",
        "location",
        "address",
        "map_url",
        "hours",
    )
    for k in keys:
        if k not in row or not isinstance(row[k], str) or not str(row[k]).strip():
            sys.exit(f"schedule.json weekly_stops[{idx}]: missing or empty {k!r}")


def load_schedule() -> dict:
    if not DATA_PATH.is_file():
        sys.exit(f"Missing {DATA_PATH}")
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if "weekly_stops" not in data or not isinstance(data["weekly_stops"], list):
        sys.exit("schedule.json must contain a non-empty list: weekly_stops")
    for i, row in enumerate(data["weekly_stops"]):
        if not isinstance(row, dict):
            sys.exit(f"weekly_stops[{i}] must be an object")
        _required_stop_keys(row, i)
    return data


def build_table_html(stops: list[dict]) -> str:
    thead = (
        "<thead><tr class=\"border-b border-ink/15 bg-ink/5\">"
        "<th class=\"px-4 py-3 font-display text-xs font-semibold uppercase tracking-wider text-ink sm:px-5\">Day</th>"
        "<th class=\"px-4 py-3 font-display text-xs font-semibold uppercase tracking-wider text-ink sm:px-5\">Location</th>"
        "<th class=\"hidden px-4 py-3 font-display text-xs font-semibold uppercase tracking-wider text-ink sm:table-cell sm:px-5\">Hours</th>"
        "</tr></thead>"
    )
    rows: list[str] = []
    for idx, row in enumerate(stops):
        day = html.escape(row["day_short"])
        dshort = html.escape(row["date_short"])
        loc = html.escape(row["location"])
        addr = html.escape(row["address"])
        href = row["map_url"].strip()
        if '"' in href or "<" in href or ">" in href:
            sys.exit(f"weekly_stops[{idx}].map_url: must not contain \", <, or >")
        hrs = html.escape(row["hours"])
        rows.append(
            f'<tr class="border-b border-ink/10 last:border-0">'
            f'<td class="px-4 py-4 font-semibold text-primary sm:px-5">'
            f'<span class="block">{day}</span>'
            f'<span class="mt-0.5 block text-xs font-normal text-ink/55">{dshort}</span></td>'
            f'<td class="px-4 py-4 text-ink sm:px-5">'
            f'<div class="font-medium text-ink">{loc}</div>'
            f'<a href="{href}" class="mt-1 block text-sm text-primary underline-offset-4 hover:underline" '
            f'target="_blank" rel="noopener noreferrer">{addr}</a>'
            f'<span class="mt-1 block text-ink/60 sm:hidden">{hrs}</span></td>'
            f'<td class="hidden px-4 py-4 text-ink/70 sm:table-cell sm:px-5">{hrs}</td></tr>'
        )
    return f'{TABLE_START}{thead}<tbody>{"".join(rows)}</tbody></table>'


def build_schedule_txt(data: dict) -> str:
    lines: list[str] = [
        "FOOD TRUCK SCHEDULE & LOCATIONS",
        "",
        "=== THIS WEEK'S SCHEDULE ===",
        "",
    ]
    for row in data["weekly_stops"]:
        lines.append(f"{row['schedule_heading']}:")
        lines.append(f"- Location: {row['location']}")
        lines.append(f"- Address: {row['address']}")
        lines.append(f"- Map link (opens in maps app): {row['map_url']}")
        lines.append(f"- Time: {row['hours']}")
        lines.append("")

    ev = data.get("special_events") or []
    if isinstance(ev, list) and ev:
        lines.extend(["=== UPCOMING SPECIAL EVENTS ===", ""])
        for item in ev:
            if str(item).strip():
                lines.append(str(item).strip())
        lines.append("")

    notes = (data.get("seasonal_notes") or "").strip()
    if notes:
        lines.extend(["=== SEASONAL SCHEDULE NOTES ===", "", notes, ""])

    find_us = (data.get("find_us") or "").strip()
    if find_us:
        lines.extend(["=== HOW TO FIND US ===", "", find_us])

    return "\n".join(lines).rstrip() + "\n"


def patch_index(table_html: str) -> str:
    raw = INDEX_PATH.read_text(encoding="utf-8")
    block = BEGIN + table_html + END
    if BEGIN in raw and END in raw:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        if not pattern.search(raw):
            sys.exit("index.html: markers present but block not found")
        return pattern.sub(block, raw, count=1)

    i = raw.find(TABLE_START)
    if i < 0:
        sys.exit("index.html: schedule table not found (expected w-full schedule table)")
    j = raw.find("</table>", i)
    if j < 0:
        sys.exit("index.html: closing </table> not found")
    j += len("</table>")
    return raw[:i] + block + raw[j:]


def main() -> None:
    data = load_schedule()
    table = build_table_html(data["weekly_stops"])
    txt = build_schedule_txt(data)

    KB_PATH.write_text(txt, encoding="utf-8")
    INDEX_PATH.write_text(patch_index(table), encoding="utf-8")
    print(f"Wrote {KB_PATH.relative_to(REPO_ROOT)}")
    print(f"Updated schedule table in {INDEX_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
