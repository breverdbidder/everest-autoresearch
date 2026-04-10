"""
score.py — Shapira Formula V1 (baseline)
=========================================
This is the ONLY file the autoresearch agent edits in future runs.

Returns a single float score for a parcel dict. Higher = better deal.
The score represents the maximum bid ceiling: if you bid above this number,
the deal math no longer works at 70% ARV with safety buffers.

Shapira V1: max_bid = (ARV * 0.70) - Repairs - $10K - MIN($25K, 15% * ARV)
"""


def shapira_score(parcel: dict) -> float:
    """
    Pure scoring function. No I/O. Must not crash on any row.

    Args:
        parcel: dict with keys from zw_parcels table.
                Required: val_market (used as ARV proxy).
                Optional: sqft_heated, year_built, flood_zone, etc.

    Returns:
        float — max bid ceiling. Negative = do not bid.
    """
    # ARV proxy: val_market from property appraiser
    try:
        arv = float(parcel.get("val_market") or 0)
        if arv != arv:  # NaN check
            arv = 0.0
    except (ValueError, TypeError):
        arv = 0.0
    if arv <= 0:
        return -1.0

    # Repair estimate: placeholder heuristic (no inspection data available)
    # Future versions can use year_built, sqft, condition signals
    repairs = _estimate_repairs(parcel)

    # Shapira V1 formula
    safety_buffer = 10_000
    profit_cap = min(25_000, 0.15 * arv)
    max_bid = (arv * 0.70) - repairs - safety_buffer - profit_cap

    return max_bid


def _estimate_repairs(parcel: dict) -> float:
    """
    Heuristic repair estimate. V1 uses a simple age-based model.
    The autoresearch agent is free to replace this entirely.
    """
    year_built = parcel.get("year_built")
    sqft_raw = parcel.get("sqft_heated") or 0
    try:
        sqft = float(sqft_raw)
        if sqft <= 0 or sqft != sqft:  # catches NaN
            sqft = 1500.0
    except (ValueError, TypeError):
        sqft = 1500.0

    if year_built is None or year_built == 0:
        return 25_000.0

    try:
        yb = int(float(year_built))
    except (ValueError, TypeError):
        return 25_000.0

    if yb <= 0:
        return 25_000.0

    age = 2026 - yb

    if age <= 10:
        per_sqft = 5.0
    elif age <= 30:
        per_sqft = 15.0
    elif age <= 50:
        per_sqft = 25.0
    else:
        per_sqft = 35.0

    return per_sqft * sqft
