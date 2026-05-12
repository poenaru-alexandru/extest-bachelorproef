"""
Retroactief de EcoLogits prefill-correctie (α=0.25) toepassen op bestaande resultaten.

Reden: EcoLogits modelleerde enkel outputtokens. Caravaca et al. (arXiv:2511.05597)
tonen empirisch aan dat inputtokens ~4× minder energie kosten dan outputtokens.
Formule: E_cor = κ × E_raw, met κ = (N_out + 0.25 × N_in) / N_out

Wat dit script doet:
  - Voegt kolom ecologits_prefill_correction toe aan de DB (indien nog niet aanwezig)
  - Werkt alle cloud-rijen bij waar de correctie nog niet is toegepast
      → energy_kwh_with_pue  = raw_energy_kwh × κ
      → co2_kg               = oud co2_kg × κ
      → regional_cloud_projections = herberekend op basis van gecorrigeerde energie
      → ecologits_prefill_correction = κ
  - Lokale rijen (CodeCarbon) worden NIET aangepast: die meten reeds de volledige
    inferentieduur inclusief prefill.

Gebruik:
  cd extraction_framework
  python migrate_prefill_correction.py
  python migrate_prefill_correction.py --db results/results.db   # alternatief pad
"""
import argparse
import json
import sqlite3
from pathlib import Path

PREFILL_ENERGY_FACTOR = 0.25  # α uit Caravaca et al. (arXiv:2511.05597)

EMISSION_FACTORS = {
    "ITA": 0.28478,
    "BEL": 0.14982,
    "FRA": 0.04144,
    "DEU": 0.32965,
    "USA": 0.38440,
    "WOR": 0.45829,
}


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        print(f"[FOUT] DB niet gevonden: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Kolom toevoegen als die nog niet bestaat
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(extraction_results)")}
    if "ecologits_prefill_correction" not in existing_cols:
        conn.execute("ALTER TABLE extraction_results ADD COLUMN ecologits_prefill_correction REAL")
        print("Kolom ecologits_prefill_correction toegevoegd aan DB.")

    # Haal alle cloud-rijen op die nog niet gecorrigeerd zijn
    rows = conn.execute("""
        SELECT id, input_tokens, output_tokens, raw_energy_kwh, co2_kg
        FROM extraction_results
        WHERE llm_provider = 'huggingface'
          AND raw_energy_kwh IS NOT NULL
          AND output_tokens > 0
          AND ecologits_prefill_correction IS NULL
    """).fetchall()

    if not rows:
        print("Geen rijen gevonden om te migreren (al gecorrigeerd of geen cloud-data).")
        conn.close()
        return

    updated = 0
    for row in rows:
        n_in  = row["input_tokens"] or 0
        n_out = row["output_tokens"]
        raw   = row["raw_energy_kwh"]

        kappa            = (n_out + PREFILL_ENERGY_FACTOR * n_in) / n_out
        corrected_energy = raw * kappa
        corrected_co2    = row["co2_kg"] * kappa if row["co2_kg"] is not None else None
        regional         = {zone: corrected_energy * ef for zone, ef in EMISSION_FACTORS.items()}

        conn.execute("""
            UPDATE extraction_results
            SET energy_kwh_with_pue          = ?,
                co2_kg                       = ?,
                regional_cloud_projections   = ?,
                ecologits_prefill_correction = ?
            WHERE id = ?
        """, (corrected_energy, corrected_co2, json.dumps(regional), kappa, row["id"]))
        updated += 1

    conn.commit()
    conn.close()
    print(f"Klaar: {updated} cloud-rijen bijgewerkt in {db_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pas EcoLogits prefill-correctie toe op bestaande DB.")
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
