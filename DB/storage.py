"""SQLite mirror of the run-level results ledger for ad-hoc SQL / DBeaver access.

The canonical results store stays ``experiments/results.csv`` — one row per run,
written by ``results.logger.finalize_run``. This module projects that CSV into a
queryable SQLite table so runs can be explored in DBeaver or queried with SQL,
without standing up a database server. SQLite is a single file
(``experiments/results.db``) that DBeaver opens natively.

The ``results`` table mirrors ``results.logger.SUMMARY_FIELDS`` exactly, keyed by
``run_id``. Re-syncing is idempotent (upsert on ``run_id``), so the file can be
regenerated from the CSV at any time and is safe to delete.

Usage:
    python -m DB.storage                       # results.csv -> results.db
    python -m DB.storage --csv P --db Q         # custom paths
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from pathlib import Path
from typing import Mapping, Optional, Sequence

from results.logger import DEFAULT_RESULTS_CSV, SUMMARY_FIELDS

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("experiments/results.db")
RESULTS_TABLE = "results"

# SQLite column type per summary field. run_id is the natural primary key, so an
# upsert (INSERT OR REPLACE) keyed on it makes re-syncing idempotent.
_COLUMN_TYPES = {
    "run_id": "TEXT PRIMARY KEY",
    "split": "TEXT",
    "domain": "TEXT",
    "agent_model": "TEXT",
    "harness_version": "TEXT",
    "num_tasks": "INTEGER",
    "num_passed": "INTEGER",
    "pass_rate": "REAL",
    "total_cost_usd": "REAL",
    "cost_per_successful_task": "REAL",
    "mean_reward": "REAL",
    "generated_at": "TEXT",
}

# CSV stores everything as strings; these fields are parsed back to numbers.
_INTEGER_FIELDS = frozenset({"num_tasks", "num_passed"})
_REAL_FIELDS = frozenset(
    {"pass_rate", "total_cost_usd", "cost_per_successful_task", "mean_reward"}
)

# Fail fast if the ledger gains a column this schema does not cover.
_uncovered = [field for field in SUMMARY_FIELDS if field not in _COLUMN_TYPES]
if _uncovered:
    raise RuntimeError(
        f"DB schema is missing a column type for: {sorted(_uncovered)}. "
        "Update _COLUMN_TYPES in DB/storage.py to match results.logger.SUMMARY_FIELDS."
    )


def _create_table_sql() -> str:
    """Build the CREATE TABLE statement from the shared summary field order."""
    columns = ",\n    ".join(f"{field} {_COLUMN_TYPES[field]}" for field in SUMMARY_FIELDS)
    return f"CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (\n    {columns}\n)"


def _insert_sql() -> str:
    """Build the idempotent upsert statement for one run row."""
    columns = ", ".join(SUMMARY_FIELDS)
    placeholders = ", ".join("?" for _ in SUMMARY_FIELDS)
    return f"INSERT OR REPLACE INTO {RESULTS_TABLE} ({columns}) VALUES ({placeholders})"


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating parent dirs) a SQLite connection with row access by name."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the results table if it does not already exist."""
    with connect(db_path) as conn:
        conn.execute(_create_table_sql())


def upsert_run(summary: Mapping[str, object], *, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Insert or replace a single run summary row (keyed by run_id)."""
    values = _row_values(summary)
    with connect(db_path) as conn:
        conn.execute(_create_table_sql())
        conn.execute(_insert_sql(), values)


def sync_from_csv(
    csv_path: Path = DEFAULT_RESULTS_CSV, *, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Mirror every row of the results CSV into the SQLite table. Returns row count."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found at {csv_path}. Run an eval first so "
            "results.logger.finalize_run writes it, then sync."
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [_row_values(record) for record in csv.DictReader(handle)]
    with connect(db_path) as conn:
        conn.execute(_create_table_sql())
        conn.executemany(_insert_sql(), rows)
    logger.info("Synced %d run row(s) from %s -> %s", len(rows), csv_path, db_path)
    return len(rows)


def _row_values(record: Mapping[str, object]) -> list:
    """Project a summary mapping onto SUMMARY_FIELDS order, coercing CSV strings."""
    return [_coerce(field, record.get(field)) for field in SUMMARY_FIELDS]


def _coerce(field: str, value: object) -> Optional[object]:
    """Coerce a raw CSV/summary value to the field's SQLite type (or None)."""
    if value is None:
        return None
    # CSV path: every cell arrives as a string ("" / "none" mean SQL NULL).
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        if field in _INTEGER_FIELDS:
            return int(stripped)
        if field in _REAL_FIELDS:
            return float(stripped)
        return stripped
    # Summary-dict path: numeric fields already arrive as int/float.
    if field in _INTEGER_FIELDS and isinstance(value, (int, float)):
        return int(value)
    if field in _REAL_FIELDS and isinstance(value, (int, float)):
        return float(value)
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: sync the results CSV into the SQLite mirror."""
    parser = argparse.ArgumentParser(
        description="Mirror experiments/results.csv into a SQLite results table."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sync_from_csv(args.csv, db_path=args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
