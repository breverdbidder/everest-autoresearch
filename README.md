# everest-autoresearch

Autonomous tuning of the **Shapira Formula V2** — a foreclosure auction max-bid calculator — using the keep-or-revert experimentation pattern.

Built by [Everest Capital USA](https://biddeed.ai) (BidDeed.AI + ZoneWise.AI).

## Attribution

This project implements the **autoresearch pattern** created by [Andrej Karpathy](https://github.com/karpathy):

- **Upstream repo:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT License)
- **Pattern proof:** [nanochat commit 6ed7d1d](https://github.com/karpathy/nanochat/commit/6ed7d1d82cee16c2e26f45d559ad3338447a6c1b)

The pattern: 3 files, ~1250 LOC, a minimalist keep-or-revert tuning loop where an AI agent edits a scoring function, evaluates it against a frozen harness, and either commits (if improved) or reverts (if not). Repeat forever.

## What is the Shapira Formula?

A bid-ceiling calculator for Florida foreclosure auctions, developed over 10+ years of investing in Brevard County:

```
max_bid = (ARV × 0.70) − Repairs − $10K − MIN($25K, 15% × ARV)
```

This repo tunes the formula's parameters and functional form against historical auction outcomes to find V2.

## Files

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Frozen evaluation harness | Nobody (after initial setup) |
| `score.py` | Shapira scoring function | Autoresearch agent only |
| `program.md` | Agent instructions/skill | Humans only |
| `results.tsv` | Experiment log (append-only) | Agent appends |

## Quickstart

```bash
# Clone
git clone https://github.com/breverdbidder/everest-autoresearch.git
cd everest-autoresearch

# Install deps
pip install -e .

# Set env vars
export SUPABASE_URL="https://mocerqjnksmhcjzxrewo.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-key-here"

# Run baseline evaluation
python -c "from prepare import evaluate; from score import shapira_score; print(evaluate(shapira_score))"
```

## Data Source

Evaluation data comes from Supabase tables:
- `zw_parcels` — 342K Brevard County parcels with 77 columns of features
- `multi_county_auctions` — 245K auction records with outcome data

If historical outcome data is sparse, the harness falls back to **bootstrap mode** using Shapira V1 as pseudo-ground-truth. This is clearly flagged in output.

## Running the Autoresearch Loop

> **Do NOT start the loop without Ariel reviewing `program.md` first.**

The loop is triggered by pointing a Claude Code agent at `program.md` as its skill file. The agent will:
1. Read `program.md` for instructions
2. Edit `score.py` with one change at a time
3. Evaluate via `prepare.py`
4. Commit if improved, revert if not
5. Log to `results.tsv` and Supabase `autoresearch_runs`
6. Repeat (max 50 experiments per run)

## Constraints

- **CPU-only.** No torch, no CUDA. Runs on Hetzner `everest-dispatch`.
- **No ML models.** This is formula tuning, not model training.
- **MIT License.** Full attribution to @karpathy for the pattern.
- **Cost discipline.** Max 50 experiments per run. Checkpoint after each.

## License

MIT — see [LICENSE](LICENSE).
