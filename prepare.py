"""
prepare.py — Frozen evaluation harness for Shapira Formula tuning
==================================================================
READ-ONLY after initial SUMMIT. The autoresearch agent must NOT modify this file.

Connects to Supabase, loads evaluation data, and exposes:
    evaluate(score_fn) -> dict

Strategy:
    1. Attempt to load historical Brevard auctions with known outcomes
       (sold_amount > 0) from multi_county_auctions.
    2. If historical outcome data is sparse (<50 auctions with outcomes),
       fall back to BOOTSTRAP mode: use upcoming/recent auctions with
       market_value as ARV proxy, and Shapira V1 as pseudo-ground-truth.
       This is clearly documented as a bootstrap — NOT real ground truth.

Data source: multi_county_auctions table has auction parcels with market_value,
sqft, year_built, and property details. zw_parcels enrichment attempted via
parcel_id join when available.

Primary metric: mean_roi on BID-tier picks (top-10 by score). Higher is better.
"""

import os
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(override=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://mocerqjnksmhcjzxrewo.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY",
    os.environ.get("SUPABASE_KEY", ""),
)

# Evaluation constants
MIN_HISTORICAL_AUCTIONS = 50
BID_TIER_K = 10  # top-K picks for precision measurement
EVAL_TIMEOUT_SECONDS = 60
EXPERIMENT_BUDGET_SECONDS = 300  # 5-min wall clock per experiment

# Bootstrap mode flag — set during data load
_BOOTSTRAP_MODE = False
_eval_df: pd.DataFrame | None = None


def _get_client():
    if not SUPABASE_KEY:
        raise EnvironmentError(
            "Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY env var. "
            "See README.md for setup."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _load_historical(client) -> pd.DataFrame:
    """Load auctions with known outcomes (sold_amount > 0) from multi_county_auctions."""
    resp = (
        client.table("multi_county_auctions")
        .select(
            "id,case_number,parcel_id,property_address,county,"
            "auction_date,opening_bid,judgment_amount,sold_amount,"
            "market_value,po_market_value,po_avm_value,po_last_sale_price,"
            "beds,baths,sqft,year_built,lot_size,property_type,"
            "auction_status,winning_bidder"
        )
        .eq("county", "Brevard")
        .not_.is_("sold_amount", "null")
        .gt("sold_amount", "0")
        .limit(1000)
        .execute()
    )
    return pd.DataFrame(resp.data)


def _load_auction_parcels_for_bootstrap(client) -> pd.DataFrame:
    """
    Load Brevard auction parcels from multi_county_auctions for bootstrap eval.
    Uses market_value as ARV proxy. These are upcoming/recent auctions.
    """
    resp = (
        client.table("multi_county_auctions")
        .select(
            "id,case_number,parcel_id,property_address,county,"
            "auction_date,opening_bid,judgment_amount,"
            "market_value,po_market_value,po_avm_value,po_last_sale_price,"
            "beds,baths,sqft,year_built,lot_size,property_type,"
            "auction_status"
        )
        .eq("county", "Brevard")
        .not_.is_("market_value", "null")
        .gt("market_value", "0")
        .limit(500)
        .execute()
    )
    df = pd.DataFrame(resp.data)

    if df.empty:
        return df

    # Try to enrich with zw_parcels features via parcel_id
    parcel_ids = df["parcel_id"].dropna().unique().tolist()[:200]
    if parcel_ids:
        try:
            enrich_resp = (
                client.table("zw_parcels")
                .select(
                    "pin,val_market,val_land,val_building,"
                    "sqft_heated,year_built as zw_year_built,"
                    "beds as zw_beds,baths_full,"
                    "zoning_code,zoning_category,flood_zone,flood_bfe,"
                    "acres_deed,is_hoa_foreclosure"
                )
                .in_("pin", parcel_ids)
                .execute()
            )
            enrich_df = pd.DataFrame(enrich_resp.data)
            if not enrich_df.empty:
                df = df.merge(enrich_df, left_on="parcel_id", right_on="pin", how="left")
        except Exception:
            pass  # Enrichment is best-effort

    return df


def _normalize_parcel(row: dict) -> dict:
    """
    Normalize a multi_county_auctions row to the format expected by score.py.
    Maps auction table columns to the zw_parcels-like keys that shapira_score expects.
    """
    return {
        # ARV proxy — prefer zw_parcels val_market, fall back to auction market_value
        "val_market": row.get("val_market") or row.get("market_value") or 0,
        "val_land": row.get("val_land") or 0,
        "val_building": row.get("val_building") or 0,
        # Physical
        "sqft_heated": row.get("sqft_heated") or row.get("sqft") or 0,
        "year_built": row.get("zw_year_built") or row.get("year_built") or 0,
        "beds": row.get("zw_beds") or row.get("beds") or 0,
        "baths_full": row.get("baths_full") or row.get("baths") or 0,
        "acres_deed": row.get("acres_deed") or row.get("lot_size") or 0,
        # Zoning/risk
        "zoning_code": row.get("zoning_code") or "",
        "zoning_category": row.get("zoning_category") or "",
        "flood_zone": row.get("flood_zone") or "",
        "flood_bfe": row.get("flood_bfe") or 0,
        "is_hoa_foreclosure": row.get("is_hoa_foreclosure") or False,
        # Auction
        "judgment_amt": row.get("judgment_amount") or 0,
        "opening_bid": row.get("opening_bid") or 0,
        "property_type": row.get("property_type") or "",
        # Identity (for debugging)
        "case_number": row.get("case_number") or "",
        "property_address": row.get("property_address") or "",
    }


def _build_bootstrap_eval(df: pd.DataFrame) -> pd.DataFrame:
    """
    BOOTSTRAP MODE: No historical outcomes available.
    Synthesize pseudo-ground-truth using Shapira V1 as the reference scorer.
    The 'pseudo_roi' simulates what ROI would be IF the parcel sold at
    market_value and the investor bid at the V1 max_bid.

    THIS IS A BOOTSTRAP — NOT REAL GROUND TRUTH.
    """
    from score import shapira_score

    rows = []
    for _, row in df.iterrows():
        parcel = _normalize_parcel(row.to_dict())
        v1_max_bid = shapira_score(parcel)
        arv = float(parcel.get("val_market") or 0)

        if v1_max_bid <= 0 or arv <= 0:
            continue

        # Pseudo outcome: assume investor buys at max_bid, sells at ARV
        pseudo_roi = (arv - v1_max_bid) / v1_max_bid

        parcel["_v1_max_bid"] = v1_max_bid
        parcel["_pseudo_roi"] = pseudo_roi
        parcel["_arv"] = arv
        rows.append(parcel)

    return pd.DataFrame(rows)


def _load_eval_data() -> pd.DataFrame:
    """Load evaluation dataset. Sets _BOOTSTRAP_MODE flag."""
    global _BOOTSTRAP_MODE, _eval_df

    if _eval_df is not None:
        return _eval_df

    client = _get_client()

    # Try historical first
    hist_df = _load_historical(client)

    if len(hist_df) >= MIN_HISTORICAL_AUCTIONS:
        _BOOTSTRAP_MODE = False
        # Normalize historical rows
        normalized = []
        for _, row in hist_df.iterrows():
            p = _normalize_parcel(row.to_dict())
            p["_sold_amount"] = float(row.get("sold_amount") or 0)
            p["_arv"] = float(p["val_market"])
            normalized.append(p)
        _eval_df = pd.DataFrame(normalized)
        print(f"[prepare] HISTORICAL mode: {len(_eval_df)} auctions with outcomes")
        return _eval_df
    else:
        _BOOTSTRAP_MODE = True
        hist_count = len(hist_df)
        warnings.warn(
            f"Only {hist_count} historical auctions with outcomes found "
            f"(need {MIN_HISTORICAL_AUCTIONS}). Using BOOTSTRAP mode with "
            "Shapira V1 pseudo-ground-truth. Results are synthetic.",
            stacklevel=2,
        )
        auction_df = _load_auction_parcels_for_bootstrap(client)
        if auction_df.empty:
            print(f"[prepare] BOOTSTRAP mode: 0 auction parcels found — EMPTY eval set")
            _eval_df = pd.DataFrame()
            return _eval_df

        bootstrap_df = _build_bootstrap_eval(auction_df)
        _eval_df = bootstrap_df
        print(f"[prepare] BOOTSTRAP mode: {len(bootstrap_df)} parcels (pseudo-ground-truth)")
        return bootstrap_df


def evaluate(score_fn: Callable[[dict], float]) -> dict:
    """
    Frozen evaluation function. Returns metrics dict.

    Args:
        score_fn: callable(parcel_dict) -> float (higher = better deal)

    Returns:
        dict with keys:
            mean_roi: average ROI on BID-tier picks
            precision_at_k: fraction of top-K picks that are profitable
            max_drawdown: worst single-deal loss in top-K
            sharpe_like: mean_roi / std_roi (risk-adjusted return)
            num_bids: total parcels where score > 0
            num_crashes: parcels where score_fn raised an exception
            mode: "HISTORICAL" or "BOOTSTRAP"
            eval_set_size: number of parcels evaluated
    """
    start = time.time()
    df = _load_eval_data()

    if df.empty:
        return {
            "mean_roi": 0.0,
            "precision_at_k": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "num_bids": 0,
            "num_crashes": 0,
            "mode": "EMPTY",
            "eval_set_size": 0,
        }

    scores = []
    crashes = 0

    for _, row in df.iterrows():
        if time.time() - start > EVAL_TIMEOUT_SECONDS:
            warnings.warn("Evaluation timed out at 60s", stacklevel=2)
            break

        parcel = row.to_dict()
        try:
            s = score_fn(parcel)
            scores.append(s)
        except Exception:
            scores.append(-999.0)
            crashes += 1

    df = df.iloc[: len(scores)].copy()
    df["_score"] = scores

    # Filter to BID-tier (positive score = willing to bid)
    bid_mask = df["_score"] > 0
    num_bids = int(bid_mask.sum())

    # Sort by score descending, take top K
    top_k = df[bid_mask].nlargest(BID_TIER_K, "_score")

    if len(top_k) == 0:
        return {
            "mean_roi": 0.0,
            "precision_at_k": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "num_bids": num_bids,
            "num_crashes": crashes,
            "mode": "BOOTSTRAP" if _BOOTSTRAP_MODE else "HISTORICAL",
            "eval_set_size": len(df),
        }

    # Compute ROI for each top-K pick
    rois = []
    for _, r in top_k.iterrows():
        arv = float(r.get("_arv") or r.get("val_market") or 0)
        bid = r["_score"]
        if bid > 0 and arv > 0:
            roi = (arv - bid) / bid
        else:
            roi = 0.0
        rois.append(roi)

    rois = np.array(rois)
    mean_roi = float(np.mean(rois)) if len(rois) > 0 else 0.0
    std_roi = float(np.std(rois)) if len(rois) > 0 else 1.0
    max_drawdown = float(np.min(rois)) if len(rois) > 0 else 0.0
    precision = float(np.mean(rois > 0)) if len(rois) > 0 else 0.0
    sharpe = mean_roi / std_roi if std_roi > 0 else 0.0

    elapsed = time.time() - start

    return {
        "mean_roi": round(mean_roi, 6),
        "precision_at_k": round(precision, 4),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe_like": round(sharpe, 4),
        "num_bids": num_bids,
        "num_crashes": crashes,
        "mode": "BOOTSTRAP" if _BOOTSTRAP_MODE else "HISTORICAL",
        "eval_set_size": len(df),
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    from score import shapira_score

    results = evaluate(shapira_score)
    print("\n=== Shapira V1 Baseline Evaluation ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
