"""Deterministic date-hint extractor — no new dependencies."""
from __future__ import annotations
import re
from datetime import date, timedelta
from typing import Optional


# ── Named-month patterns ──────────────────────────────────────────────────────

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_RELATION_BEFORE = re.compile(
    r"\b(before|earlier\s+than|until|prior\s+to)\b", re.I
)
_RELATION_AFTER = re.compile(
    r"\b(after|since|later\s+than)\b", re.I
)


def _parse_date(text: str) -> Optional[date]:
    """Try to parse a date string; return None on failure."""
    text = text.strip()
    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    # DD-MM-YYYY
    m = re.fullmatch(r"(\d{1,2})-(\d{2})-(\d{4})", text)
    if m:
        return date(int(m[3]), int(m[2]), int(m[1]))
    return None


def _named_month_date(day: int, month: int, year: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


# Pattern: "18th February 2025", "Feb 18, 2025", "18 Feb 2025"
_NAMED_DATE_PATTERNS = [
    # "18th February 2025" / "18 February 2025"
    re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
        + "|".join(_MONTH_NAMES.keys())
        + r")\s+(\d{4})\b",
        re.I,
    ),
    # "February 18, 2025" / "Feb 18 2025"
    re.compile(
        r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        re.I,
    ),
]


def _extract_named_dates(text: str) -> list[tuple[date, int, int]]:
    """Return [(date_obj, start_pos, end_pos), ...]."""
    results: list[tuple[date, int, int]] = []
    for pat in _NAMED_DATE_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            if groups[0].isdigit():               # pattern 1: day month year
                day, month_str, year = int(groups[0]), groups[1], int(groups[2])
            else:                                  # pattern 2: month day year
                month_str, day, year = groups[0], int(groups[1]), int(groups[2])
            month_num = _MONTH_NAMES.get(month_str.lower())
            if month_num:
                d = _named_month_date(day, month_num, year)
                if d:
                    results.append((d, m.start(), m.end()))
    return results


def extract_date_hints(query: str, today: Optional[date] = None) -> dict:
    """
    Extract date hints from a query string.

    Returns:
        {"dates": ["YYYY-MM-DD", ...], "relations": ["before"|"after"|"on", ...]}
    """
    if today is None:
        today = date.today()

    dates: list[str] = []
    relations: list[str] = []

    def _relation_for(pos: int) -> str:
        prefix = query[:pos]
        if _RELATION_BEFORE.search(prefix):
            return "before"
        if _RELATION_AFTER.search(prefix):
            return "after"
        return "on"

    # 1. ISO / numeric formats
    for m in re.finditer(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}-\d{2}-\d{4})\b", query
    ):
        d = _parse_date(m.group())
        if d:
            dates.append(d.isoformat())
            relations.append(_relation_for(m.start()))

    # 2. Named-month dates (skip spans already covered by numeric)
    covered = {d for d in dates}
    for d, start, _end in _extract_named_dates(query):
        iso = d.isoformat()
        if iso not in covered:
            dates.append(iso)
            relations.append(_relation_for(start))
            covered.add(iso)

    # 3. Relative phrases
    lower = query.lower()
    if "today" in lower:
        dates.append(today.isoformat())
        relations.append("on")
    if "yesterday" in lower:
        dates.append((today - timedelta(days=1)).isoformat())
        relations.append("on")
    if "last week" in lower:
        dates.append((today - timedelta(weeks=1)).isoformat())
        relations.append("after")

    return {"dates": dates, "relations": relations}
