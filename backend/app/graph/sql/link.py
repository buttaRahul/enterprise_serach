"""link_schema — score tables against question tokens, pick top-3."""
from __future__ import annotations
import re

SYNONYMS: dict[str, list[str]] = {
    "employee": ["staff", "person", "people", "colleague", "engineer", "member", "who"],
    "project": ["initiative", "work", "owner"],
    "department": ["team", "dept"],
    "manager": ["manages", "boss", "reports"],
    "hire_date": ["hired", "joined", "recent"],
}

_STOP_WORDS = frozenset({"the", "a", "an", "is", "are", "was", "were", "of", "in",
                          "on", "at", "to", "for", "and", "or", "with", "by", "from"})


def _tokenise(text: str) -> set[str]:
    words = re.split(r"[\W_]+", text.lower())
    words = [_deplural(w) for w in words if len(w) > 1 and w not in _STOP_WORDS]
    return set(words)


def _deplural(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _synonym_set(table_name: str, col_names: list[str]) -> set[str]:
    tokens: set[str] = set()
    for key, synonyms in SYNONYMS.items():
        if key in _tokenise(table_name) or any(key in _tokenise(c) for c in col_names):
            tokens.update(synonyms)
            tokens.add(key)
    return tokens


def score_table(question_tokens: set[str], table_name: str, col_names: list[str]) -> int:
    table_tokens = _tokenise(table_name) | {_deplural(w) for c in col_names for w in _tokenise(c)}
    syn_tokens = _synonym_set(table_name, col_names)
    all_table_tokens = table_tokens | syn_tokens

    score = len(question_tokens & all_table_tokens)
    if _deplural(table_name) in question_tokens or table_name in question_tokens:
        score += 2
    return score


def link_schema(
    question: str,
    database,
    excluded_tables: list[str] | None = None,
    schema_cache: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Return (linked_table_names, {table: ddl_string}).

    Also adds FK-referenced tables so joins are possible.
    """
    excluded = set(excluded_tables or [])
    q_tokens = _tokenise(question)

    from sqlalchemy import inspect as sa_inspect
    inspector = sa_inspect(database._engine)
    all_tables = [t for t in inspector.get_table_names() if t not in excluded]

    if not all_tables:
        return [], {}

    # Score each table
    scored: list[tuple[str, int]] = []
    for tbl in all_tables:
        cols = [c["name"] for c in inspector.get_columns(tbl)]
        s = score_table(q_tokens, tbl, cols)
        scored.append((tbl, s))

    scored.sort(key=lambda x: -x[1])
    top3 = [t for t, s in scored[:3] if s > 0]

    if not top3:
        top3 = [t for t, _ in scored[:3]] if len(scored) <= 5 else [t for t, _ in scored[:3]]

    # Add FK-referenced tables
    fk_tables: set[str] = set()
    for tbl in list(top3):
        for fk in inspector.get_foreign_keys(tbl):
            ref = fk.get("referred_table")
            if ref and ref not in excluded and ref not in top3:
                fk_tables.add(ref)
    linked = list(dict.fromkeys(top3 + list(fk_tables)))

    # Build schemas (use cache if available)
    cache = schema_cache or {}
    schemas: dict[str, str] = {}
    for tbl in linked:
        if tbl in cache:
            schemas[tbl] = cache[tbl]
        else:
            ddl_map = database.schema([tbl])
            schemas[tbl] = ddl_map.get(tbl, f"-- {tbl} (no schema)")
            cache[tbl] = schemas[tbl]

    return linked, schemas
