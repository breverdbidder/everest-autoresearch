# program.md — Shapira Formula Autoresearch Skill

> Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT).
> Edited by humans only. The autoresearch agent must NOT modify this file.

---

## Setup

You are an autonomous research agent tuning the **Shapira Formula** — a real estate
foreclosure auction bid-ceiling calculator used by Everest Capital USA (BidDeed.AI).

Your workspace:
- `prepare.py` — **FROZEN.** Evaluation harness. Do not edit.
- `score.py` — **YOUR FILE.** Contains `shapira_score(parcel: dict) -> float`. Edit freely.
- `program.md` — **THIS FILE.** Your instructions. Do not edit.
- `results.tsv` — Append-only experiment log.

Your goal: **maximize `mean_roi`** on the held-out evaluation set by improving
`shapira_score()` in `score.py`.

## How the formula works

The Shapira Formula computes a **max bid ceiling** for a foreclosure auction parcel.
If an investor bids above this number, the deal math breaks. Higher ceiling = more
aggressive bidding = higher ROI if the parcel is correctly valued.

Baseline (V1):
```
max_bid = (ARV * 0.70) - Repairs - $10K - MIN($25K, 15% * ARV)
```

Where:
- ARV = After-Repair Value (proxied by `val_market` from the property appraiser)
- Repairs = heuristic estimate based on age and square footage
- $10K = fixed safety buffer
- MIN($25K, 15% * ARV) = profit cap (ensures minimum profit margin)

## Available parcel features

Each parcel dict contains columns from the `zw_parcels` Supabase table (77 columns).
Key features you can use:

| Feature | Type | Notes |
|---------|------|-------|
| val_market | int | Market value (ARV proxy) |
| val_land | int | Land value |
| val_building | int | Building/improvement value |
| sqft_heated | int | Heated square footage |
| year_built | int | Year constructed |
| beds | int | Bedrooms |
| baths_full | int | Full bathrooms |
| acres_deed | float | Lot size in acres |
| zoning_code | str | Zoning designation |
| zoning_category | str | Broad zoning category |
| flood_zone | str | FEMA flood zone |
| flood_bfe | float | Base flood elevation |
| judgment_amt | int | Court judgment amount |
| opening_bid | int | Auction opening bid |
| is_hoa_foreclosure | bool | HOA vs mortgage foreclosure |
| sale_price | int | Last recorded sale price |
| sale_date | date | Last sale date |

## Experimentation Rules

1. **LOOP FOREVER.** Run experiments continuously until the budget is exhausted.
   **NEVER STOP** unless you hit the 50-experiment cap or 10 consecutive failures.

2. **One change at a time.** Each experiment modifies exactly ONE aspect of
   `shapira_score()`. Document what you changed and why in the commit message.

3. **Rollback discipline.** After each edit:
   - Run: `python -c "from prepare import evaluate; from score import shapira_score; print(evaluate(shapira_score))"`
   - If `mean_roi` improved → `git add score.py && git commit -m "experiment N: <description>"`
   - If `mean_roi` did NOT improve → `git checkout score.py` (revert to last good version)
   - **Never keep a change that doesn't improve the primary metric.**

4. **Simplicity criterion.** If you can delete code and `mean_roi` stays the same
   or improves, DELETE IT. Simpler formulas are better. Deletions that preserve
   metric = always keep.

5. **No external I/O.** `shapira_score()` must remain a pure function. No network
   calls, no file reads, no imports beyond standard library + numpy.

6. **No crashes.** The function must return a float for every row in the eval set.
   If `num_crashes > 0`, the experiment is an automatic failure — revert.

## Branching Convention

- Work on branch: `autoresearch/shapira-YYYYMMDD`
- Each successful experiment = one commit on that branch
- Failed experiments = reverted, not committed

## Logging

Append to `results.tsv` after EVERY experiment (success or failure):

```
commit\tmean_roi\tprecision_at_k\tstatus\tdescription
```

- `commit`: git short hash (or "REVERTED" if rolled back)
- `mean_roi`: from evaluate() output
- `precision_at_k`: from evaluate() output
- `status`: "improved" | "neutral" | "regressed" | "crashed"
- `description`: one-line description of what was tried

## Cost Discipline

- **Max 50 experiments per run.** Stop after 50, checkpoint results.
- **Checkpoint to Supabase** `autoresearch_runs` table after each experiment:
  `(run_id, branch, commit, mean_roi, precision_at_k, status, description, created_at)`
- **Escalation:** If 10 consecutive experiments fail to improve `mean_roi`,
  stop and report to Ariel. The formula may be at a local optimum requiring
  a fundamentally different approach.

## Output Format

At the end of each run, print a summary:

```
=== Autoresearch Run Summary ===
Branch: autoresearch/shapira-20260410
Experiments: 23/50
Improvements: 7
Best mean_roi: 0.4523 (vs baseline 0.3891)
Best commit: abc1234
Status: IMPROVED | PLATEAU | FAILED
```

## What NOT to do

- Do NOT edit `prepare.py` or `program.md`
- Do NOT add LangGraph, LangChain, or any orchestration framework
- Do NOT add GPU/CUDA dependencies
- Do NOT add more than 3 new imports to score.py
- Do NOT make the function longer than 100 lines
- Do NOT use machine learning models (no sklearn, no torch) — this is formula tuning
- Do NOT fabricate evaluation data or modify the eval set
