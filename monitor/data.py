"""Data layer: read, validate, de-duplicate and store news articles in SQLite.

Why SQLite? It ships with Python and speaks real SQL, so the demo needs no setup.
In production this would be Postgres, Snowflake or Databricks; the ideas are the same:
validate on the way in, make loads idempotent, and only process what is new.
"""

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

REQUIRED_FIELDS = ("url", "source", "title", "body", "published_at")
MIN_BODY_CHARS = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS counterparties (
    name          TEXT PRIMARY KEY,
    aliases       TEXT NOT NULL,          -- pipe-separated, e.g. "Meridian Steel|MSC Corp"
    sector        TEXT,
    rating        TEXT,
    exposure_musd REAL
);

CREATE TABLE IF NOT EXISTS articles (
    article_id    TEXT PRIMARY KEY,       -- hash of the URL: same URL -> same row
    url           TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL UNIQUE,   -- hash of the normalised title: catches syndicated copies
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL,
    published_at  TEXT NOT NULL,          -- ISO date, so text comparison = date comparison
    ingested_at   TEXT NOT NULL,
    processed     INTEGER NOT NULL DEFAULT 0  -- 0 until the agent has fully finished with it
);

CREATE TABLE IF NOT EXISTS assessments (
    article_id    TEXT PRIMARY KEY REFERENCES articles(article_id),  -- one result per article
    counterparty  TEXT NOT NULL,
    severity      TEXT NOT NULL,
    risk_type     TEXT NOT NULL,
    rationale     TEXT NOT NULL,
    evidence      TEXT NOT NULL,          -- JSON list of sentences quoted from the article
    status        TEXT NOT NULL,          -- logged | alert_published | rejected_by_reviewer
    reviewer      TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected_records (
    raw           TEXT PRIMARY KEY,       -- the bad record as received
    reason        TEXT NOT NULL,
    seen_at       TEXT NOT NULL
);
"""


@dataclass
class IngestReport:
    inserted: list[str] = field(default_factory=list)   # new article_ids
    duplicates: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (url or "?", reason)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_hash(text: str) -> str:
    # sha1 is fine here: we need a stable ID, not security.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per line. A broken line becomes a record that validation will reject."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {"_parse_error": str(e), "_raw_line": line.strip()}


def validate_record(raw: dict) -> str | None:
    """Return the reason a record is unusable, or None if it is fine."""
    if "_parse_error" in raw:
        return f"invalid JSON: {raw['_parse_error']}"
    for name in REQUIRED_FIELDS:
        if not str(raw.get(name) or "").strip():
            return f"missing field: {name}"
    try:
        date.fromisoformat(raw["published_at"])
    except ValueError:
        return f"bad published_at: {raw['published_at']!r}"
    if len(raw["body"].strip()) < MIN_BODY_CHARS:
        return "body too short"
    return None


def find_counterparties(text: str, counterparties: list[dict]) -> list[str]:
    """Names of monitored counterparties mentioned in `text`, in order of first mention.

    Matches whole words only, so "Harbor Point" does not match "Harbor Pointe".
    """
    hits = []
    for cp in counterparties:
        first_position = None
        for alias in cp["aliases"]:
            match = re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE)
            if match and (first_position is None or match.start() < first_position):
                first_position = match.start()
        if first_position is not None:
            hits.append((first_position, cp["name"]))
    return [name for _, name in sorted(hits)]


class Store:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- reference data ----------

    def load_counterparties(self, csv_path: str | Path) -> int:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with self.conn:  # one transaction
            self.conn.executemany(
                """INSERT INTO counterparties (name, aliases, sector, rating, exposure_musd)
                   VALUES (:name, :aliases, :sector, :rating, :exposure_musd)
                   ON CONFLICT(name) DO UPDATE SET
                       aliases = excluded.aliases, sector = excluded.sector,
                       rating = excluded.rating, exposure_musd = excluded.exposure_musd""",
                rows,
            )
        return len(rows)

    def counterparties(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM counterparties ORDER BY name").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["aliases"] = [d["name"]] + [a for a in d["aliases"].split("|") if a]
            result.append(d)
        return result

    def profile(self, name: str) -> dict:
        row = self.conn.execute(
            "SELECT name, sector, rating, exposure_musd FROM counterparties WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else {}

    # ---------- ingestion ----------

    def ingest_records(self, records: Iterable[dict]) -> IngestReport:
        """Validate and insert records. Safe to run again on the same file (idempotent)."""
        report = IngestReport()
        now = _now()
        with self.conn:  # all-or-nothing: a crash mid-file leaves the table unchanged
            for raw in records:
                reason = validate_record(raw)
                if reason:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO rejected_records (raw, reason, seen_at) VALUES (?, ?, ?)",
                        (json.dumps(raw, sort_keys=True), reason, now),
                    )
                    report.rejected.append((raw.get("url", "?"), reason))
                    continue

                article_id = _short_hash(raw["url"])
                # INSERT OR IGNORE skips rows that break a UNIQUE rule: same URL or same title.
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO articles
                       (article_id, url, content_hash, source, title, body, published_at, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        article_id,
                        raw["url"],
                        _short_hash(normalise_title(raw["title"])),
                        raw["source"],
                        raw["title"].strip(),
                        raw["body"].strip(),
                        raw["published_at"],
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    report.inserted.append(article_id)
                else:
                    report.duplicates += 1
        return report

    # ---------- reads used by the agent ----------

    def get_article(self, article_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM articles WHERE article_id = ?", (article_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown article_id {article_id}")
        return dict(row)

    def article_id_for_url(self, url: str) -> str | None:
        row = self.conn.execute("SELECT article_id FROM articles WHERE url = ?", (url,)).fetchone()
        return row["article_id"] if row else None

    def all_articles(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM articles ORDER BY published_at, article_id")]

    def unprocessed_articles(self) -> list[dict]:
        """Incremental processing: only articles the agent has not finished yet, oldest first."""
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM articles WHERE processed = 0 ORDER BY published_at, article_id"
            )
        ]

    def mark_processed(self, article_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE articles SET processed = 1 WHERE article_id = ?", (article_id,))

    # ---------- long-term memory ----------

    def save_assessment(self, article_id: str, counterparty: str, assessment: dict,
                        status: str, reviewer: str | None = None) -> None:
        """Upsert: running the same article twice leaves exactly one row."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO assessments
                   (article_id, counterparty, severity, risk_type, rationale, evidence, status, reviewer, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(article_id) DO UPDATE SET
                       severity = excluded.severity, risk_type = excluded.risk_type,
                       rationale = excluded.rationale, evidence = excluded.evidence,
                       status = excluded.status, reviewer = excluded.reviewer,
                       created_at = excluded.created_at""",
                (
                    article_id,
                    counterparty,
                    assessment["severity"],
                    assessment["risk_type"],
                    assessment["rationale"],
                    json.dumps(assessment["evidence"]),
                    status,
                    reviewer,
                    _now(),
                ),
            )

    def prior_alerts(self, counterparty: str, before: str, within_days: int = 45,
                     exclude_article_id: str | None = None) -> list[dict]:
        """Published medium/high alerts for this counterparty in the `within_days` before `before`.

        Filtering on the article's publish date (not when we processed it) means the agent
        never "remembers" something that had not happened yet at the time of the article.
        """
        rows = self.conn.execute(
            """SELECT a.article_id, a.title, a.published_at, s.severity, s.risk_type
               FROM assessments AS s
               JOIN articles AS a ON a.article_id = s.article_id
               WHERE s.counterparty = ?
                 AND s.status = 'alert_published'
                 AND s.severity IN ('medium', 'high')
                 AND a.published_at < ?
                 AND a.published_at >= date(?, ?)
                 AND a.article_id != ?
               ORDER BY a.published_at DESC
               LIMIT 5""",
            (counterparty, before, before, f"-{within_days} days", exclude_article_id or ""),
        ).fetchall()
        return [dict(r) for r in rows]
