"""SQLite persistence layer for extraction results."""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from extraction_framework.scoring import ExtractionResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT    NOT NULL,
    run_number              INTEGER,
    pdf_name                TEXT    NOT NULL,
    extractor_name          TEXT,
    llm_provider            TEXT,
    llm_model               TEXT,
    success                 INTEGER NOT NULL,
    error                   TEXT,
    ttft_seconds            REAL,
    generation_seconds      REAL,
    total_inference_seconds REAL,
    input_tokens            INTEGER,
    output_tokens           INTEGER,
    total_tokens            INTEGER,
    raw_energy_kwh          REAL,
    energy_kwh_with_pue     REAL,
    co2_kg                  REAL,
    cpu_energy_kwh          REAL,
    gpu_energy_kwh          REAL,
    ram_energy_kwh          REAL,
    energy_source                TEXT,
    ecologits_prefill_correction REAL,
    regional_cloud_projections   TEXT,
    validation_score             REAL,
    ground_truth_score           REAL,
    extracted_data               TEXT,
    timestamp                    TEXT
)
"""


class ResultsDB:
    """Append-only SQLite store — one row per inference call."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema without breaking existing DBs."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(extraction_results)")}
        if "run_number" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN run_number INTEGER")
        if "regional_cloud_projections" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN regional_cloud_projections TEXT")
        if "validation_score" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN validation_score REAL")
        if "raw_energy_kwh" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN raw_energy_kwh REAL")
        if "energy_kwh_with_pue" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN energy_kwh_with_pue REAL")
        if "ecologits_prefill_correction" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN ecologits_prefill_correction REAL")
        if "ground_truth_score" not in existing:
            conn.execute("ALTER TABLE extraction_results ADD COLUMN ground_truth_score REAL")
        # Backfill legacy rows: energy_kwh (old column) → raw_energy_kwh if present
        if "energy_kwh" in existing and "raw_energy_kwh" in existing:
            conn.execute(
                "UPDATE extraction_results SET raw_energy_kwh = energy_kwh WHERE raw_energy_kwh IS NULL AND energy_kwh IS NOT NULL"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, result: ExtractionResult, session_id: str, run_number: Optional[int] = None) -> int:
        """Insert one result row and return its new id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO extraction_results (
                    session_id, run_number, pdf_name, extractor_name,
                    llm_provider, llm_model,
                    success, error,
                    ttft_seconds, generation_seconds,
                    total_inference_seconds,
                    input_tokens, output_tokens, total_tokens,
                    raw_energy_kwh, energy_kwh_with_pue, co2_kg,
                    cpu_energy_kwh, gpu_energy_kwh, ram_energy_kwh,
                    energy_source, ecologits_prefill_correction, regional_cloud_projections,
                    validation_score, ground_truth_score, extracted_data, timestamp
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    session_id,
                    run_number,
                    Path(result.pdf_file).name,
                    result.extractor_name,
                    result.llm_provider,
                    result.llm_model,
                    int(result.success),
                    result.error,
                    result.ttft_seconds,
                    result.generation_seconds,
                    result.total_inference_seconds,
                    result.input_tokens,
                    result.output_tokens,
                    result.total_tokens,
                    result.raw_energy_kwh,
                    result.energy_kwh_with_pue,
                    result.co2_kg,
                    result.cpu_energy_kwh,
                    result.gpu_energy_kwh,
                    result.ram_energy_kwh,
                    result.energy_source,
                    result.ecologits_prefill_correction,
                    json.dumps(result.regional_cloud_projections) if result.regional_cloud_projections else None,
                    result.validation_score,
                    result.ground_truth_score,
                    json.dumps(result.extracted_data) if result.extracted_data else None,
                    result.timestamp,
                ),
            )
            return cursor.lastrowid
