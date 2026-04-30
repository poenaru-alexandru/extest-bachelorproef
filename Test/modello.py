from typing import List, Optional
from pydantic import BaseModel, Field

class Periode(BaseModel):
    supplier: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    kwh_quantity: Optional[float] = None

class BachelorProefModel(BaseModel):
    """You are an expert at extracting structured data from Italian utility bills (bollette).
    Extract the following fields and return them as JSON.

    - supplier: string
    - start_date: string (Format: YYYY-MM-DD)
    - end_date: string (Format: YYYY-MM-DD)
    - kwh_quantity: float

    CRITICAL RULES — read carefully:
    1. Only extract the INVOICED billing period — the period this specific invoice charges for.
    2. IGNORE all historical tables such as "Andamento storico dei prelievi", yearly overviews,
       and any table showing multiple past months/years side by side. These are NOT the invoiced period.
    3. A typical invoice covers exactly ONE billing period (e.g. one month).
    4. A date range spanning a full year (e.g. January 1 to December 31) is NEVER a valid billing
       period — it is always a historical summary. Reject it and look for the actual invoice month.
    5. The kwh_quantity MUST correspond to the same billing period as start_date/end_date.
       Do not mix kWh values from one period with dates from another.

    WHERE TO FIND EACH FIELD:
    - supplier: the energy company name, usually at the top of the first page.
    - start_date / end_date: look in "importi riferiti al mese di", "periodo di fornitura",
      "fornitura dal ... al ...", or "dettagli riferiti alla fattura".
    - kwh_quantity: total active energy in kWh for the invoiced billing period only.
      If split by time bands (F1, F2, F3), SUM those kWh values from the "Misure" or
      "Dettaglio dei consumi" section. Do NOT use kWh values from historical/annual summary tables.
    If supplier, start_date, end_date or kwh_quantity is not present, set it to null.
    Always return exactly one entry in the periodes list."""

    periodes: List[Periode] = Field(..., min_length=1)