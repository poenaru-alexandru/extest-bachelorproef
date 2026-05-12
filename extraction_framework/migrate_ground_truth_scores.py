"""
Retroactief de ground-truth-nauwkeurigheidsscore berekenen voor alle bestaande rijen.

Wat wordt gemeten:
  Voor elke extractie wordt periodes[0] vergeleken met de bekende correcte waarden
  uit results/ground_truth_expected_fields_full.json.

Scoringslogica (25 punten per veld = 100 totaal):
  - supplier       : exacte overeenkomst (strip/lower; null == null = correct)
  - start_date     : overeenkomst na normalisering van datumformaten
                     (ondersteund: YYYY-MM-DD, DD/MM/YYYY, DD.MM.YYYY)
  - end_date       : idem
  - kwh_quantity   : numeriek gelijk binnen 0,5 % relatieve tolerantie of 0,01 kWh

Rijen zonder extraction_data (mislukte aanroepen of PDF niet in ground truth)
krijgen de waarde NULL — ze worden bewust uitgesloten van de verdere analyses.

Gebruik:
  cd extraction_framework
  python migrate_ground_truth_scores.py
  python migrate_ground_truth_scores.py --db results/results.db   # alternatief pad
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path


GT_PATH = Path(__file__).parent / "results" / "ground_truth_expected_fields_full.json"


def parse_date(value) -> str | None:
    """Normalize date string to ISO YYYY-MM-DD; returns None if not parseable."""
    if value is None:
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


def score_against_ground_truth(extracted_data: dict | None, expected: dict) -> float | None:
    """
    Compare extracted periodes[0] against expected_fields dict.
    Returns float 0-100, or None when extracted_data is absent.
    """
    if extracted_data is None:
        return None
    if not isinstance(extracted_data, dict):
        return 0.0

    periodes = extracted_data.get("periodes")
    if not isinstance(periodes, list) or len(periodes) == 0:
        return 0.0

    first = periodes[0]
    if not isinstance(first, dict):
        return 0.0

    score = 0.0

    # supplier (25 pts)
    exp_s = expected.get("supplier")
    got_s = first.get("supplier")
    if exp_s is None and got_s is None:
        score += 25.0
    elif exp_s is not None and got_s is not None:
        if str(exp_s).strip().lower() == str(got_s).strip().lower():
            score += 25.0

    # start_date (25 pts)
    if parse_date(first.get("start_date")) == parse_date(expected.get("start_date")):
        score += 25.0

    # end_date (25 pts)
    if parse_date(first.get("end_date")) == parse_date(expected.get("end_date")):
        score += 25.0

    # kwh_quantity (25 pts)
    exp_kwh = expected.get("kwh_quantity")
    got_kwh = first.get("kwh_quantity")
    if exp_kwh is None and got_kwh is None:
        score += 25.0
    elif exp_kwh is not None and got_kwh is not None:
        try:
            exp_f = float(exp_kwh)
            got_f = float(got_kwh)
            tol = max(0.01, 0.005 * abs(exp_f))
            if abs(got_f - exp_f) <= tol:
                score += 25.0
        except (TypeError, ValueError):
            pass

    return score


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        print(f"[FOUT] DB niet gevonden: {db_path}")
        return
    if not GT_PATH.exists():
        print(f"[FOUT] Ground-truth JSON niet gevonden: {GT_PATH}")
        return

    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    # Bouw lookup: pdf-bestandsnaam → expected_fields
    gt_lookup: dict[str, dict] = {
        doc["pdf"]: doc["expected_fields"]
        for doc in gt_data.get("documents", [])
    }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Kolom toevoegen als die nog niet bestaat
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(extraction_results)")}
    if "ground_truth_score" not in existing_cols:
        conn.execute("ALTER TABLE extraction_results ADD COLUMN ground_truth_score REAL")
        print("Kolom ground_truth_score toegevoegd aan DB.")

    rows = conn.execute(
        "SELECT id, pdf_name, extracted_data FROM extraction_results"
    ).fetchall()

    updated = skipped = 0
    for row in rows:
        pdf_name = row["pdf_name"]
        if pdf_name not in gt_lookup:
            # Geen ground truth beschikbaar voor dit document — laat NULL
            skipped += 1
            continue

        extracted = None
        if row["extracted_data"]:
            try:
                extracted = json.loads(row["extracted_data"])
            except (json.JSONDecodeError, TypeError):
                extracted = None

        gt_score = score_against_ground_truth(extracted, gt_lookup[pdf_name])

        conn.execute(
            "UPDATE extraction_results SET ground_truth_score = ? WHERE id = ?",
            (gt_score, row["id"]),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Klaar: {updated} rijen bijgewerkt, {skipped} rijen overgeslagen (geen ground truth).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bereken ground-truth-nauwkeurigheidsscores voor alle DB-rijen."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent / "results" / "results.db",
        help="Pad naar results.db (standaard: results/results.db)",
    )
    args = parser.parse_args()
    migrate(args.db)


if __name__ == "__main__":
    main()
