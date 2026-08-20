# SKU Replenishment Recommender

> Per-customer, per-channel weekly SKU recommendation for retail / FMCG.
> Walk-forward validated, AutoGluon + LightGBM ensemble, **per-segment modeling**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this repo is

A **reusable methodology and code template** for the task:

> *"Given each customer's last N weeks of purchase history, predict a ranked
> list of SKUs to recommend for the coming week, evaluated by **F1** (or
> precision/recall at K). Per-channel segmentation is required when channels
> behave differently."*

The repo contains:

- **`src/core/lgb_v4.py`** — LightGBM with **co-occurrence** + **route
  collaboration** + **SKU momentum** features (the highest-impact feature set)
- **`src/core/ag_v*.py`** — AutoGluon multi-model ensembles (GBM + CAT + XGB + RF + XT)
- **`src/experiments/ag_v*.py`** — versioned experiments showing what worked
  and what failed (with root causes)
- **`docs/LESSONS_LEARNED.md`** — full engineering methodology
- **`docs/FEATURE_LESSONS.md`** — feature engineering lessons from Instacart /
  Store Sales / Favorita / M5 / H&M
- **`docs/EDA_GUIDE.md`** — EDA discipline for retail recommendation problems

## What this repo is NOT

- **Not a ready-to-run pipeline.** It requires *your* orders / customers /
  products CSVs (not shipped for confidentiality).
- **Not specific to any one customer, channel taxonomy, or product mix.** All
  business-specific values are abstracted to `channel_a/b/c/d` and generic
  field names — you must adapt to your own taxonomy.

---

## Quick start (your own data)

### 1. Prepare data (3 CSVs)

| File | Required columns |
|---|---|
| `orders.csv` | `OUTLET`, `ARTICLE`, `ORDER_DATE`, `SETTLE_CASE`, `CHANNEL`, `PS_ROUTE` |
| `customers.csv` | `OUTLET`, `CHANNEL`, `PS_ROUTE` |
| `products.csv` | `ARTICLE`, ... (any product attributes) |

### 2. Configure channel mapping

Edit `CHANNEL_MAP` in `src/core/lgb_v4.py` to match your taxonomy:

```python
CHANNEL_MAP = {
    'channel_a': ['sub_code_1', 'sub_code_2'],
    'channel_b': ['sub_code_8', 'sub_code_9', 'sub_code_10'],
    'channel_c': ['sub_code_13', 'sub_code_14'],
    'channel_d': ['sub_code_3', 'sub_code_4'],
}
```

### 3. Run

```bash
cd src/core
python lgb_v4.py
```

Output: per-channel walk-forward F1 scores and per-week logs.

---

## Methodology (TL;DR)

The full engineering discipline is in `docs/LESSONS_LEARNED.md`. The core ideas:

1. **Walk-forward validation** (not K-fold) — see "Evaluation discipline"
2. **Per-channel independent models** — channels behave differently, share no N
3. **Co-occurrence + route + momentum features** — the proven feature trio
   (see `src/core/lgb_v4.py`)
4. **AutoGluon first** — default ensemble beats hand-tuned single models for
   tabular retail data
5. **Direct F1 optimization** — Instacart top solutions do this; AUC is a
   proxy, F1 is the target
6. **Per-customer N formula** — `N = clamp(typical_n + base, lo, hi)`, tuned
   per channel (see `src/experiments/ag_v28.py` for Optuna-tuned N)
7. **Saturation discipline** — declare "I'm saturated" after 3-4 versions with
   no F1 gain; switch paradigm rather than tune harder

---

## Repo structure

```
sku-replenishment-recommender/
├── README.md                       # this file
├── LICENSE                         # MIT
├── .gitignore                      # strict excludes
├── src/
│   ├── core/
│   │   ├── lgb_v4.py               # LightGBM with co-occurrence + route + momentum
│   │   ├── ag_v9.py                # AutoGluon baseline
│   │   ├── ag_v13.py               # + full feature EDA
│   │   └── ag_v20.py               # + route × SKU interaction
│   ├── experiments/
│   │   ├── ag_v12.py, ag_v19.py    # versioned experiments
│   │   ├── ag_v23.py               # Optuna N tuning
│   │   ├── ag_v26.py, ag_v27.py    # 13-week training
│   │   ├── ag_v28.py               # per-channel Optuna N
│   │   └── ag_v29_rolling.py       # walk-forward validation across 4 weeks
│   ├── fix_submission_coverage.py  # fill missing outlets with channel hot-SKUs
│   └── generate_final_submission.py # generate per-week submission
├── docs/
│   ├── LESSONS_LEARNED.md          # engineering methodology (start here)
│   ├── FEATURE_LESSONS.md          # feature engineering lessons
│   └── EDA_GUIDE.md                # EDA discipline for retail recommendation
└── examples/
    └── README.md                   # quickstart on your own data
```

---

## When to use this repo

✅ **Good fit:**
- Per-customer, per-week purchase recommendation
- F1 / precision-recall-at-K metric
- 500-5000 customers per channel, 50-500 SKUs
- Channels with significantly different behavior (split recommended)
- You have ≥10 weeks of history per customer

❌ **Not a fit:**
- Single-channel, single-product forecasting (use M5-style methods)
- Cold-start dominant (>30% new customers per week)
- Real-time / streaming requirements
- Image / text / multimodal features (this repo is tabular-only)

---

## Key results (anonymized)

On a representative retail/FMCG benchmark (anonymized):

| Metric | Baseline | This method | Gain |
|---|---|---|---|
| **F1 (avg over channels)** | ~30% | ~76% | **+46pp (+150%)** |
| Walk-forward stability (4 weeks) | — | ±0.2pp | very stable |

The exact dataset, channel mix, and SKUs are intentionally not documented
here. What generalizes across retail / FMCG settings is the **methodology**,
not the specific numbers.

---

## Adapting to your project

The methodology is the asset, not the code. To apply:

1. **Read `docs/LESSONS_LEARNED.md`** end-to-end (most important)
2. **Apply the 8-point leakage checklist** to your data before any training
3. **Run the EDA discipline** in `docs/EDA_GUIDE.md`
4. **Start with AutoGluon** (`src/core/ag_v9.py`) as your baseline
5. **Add the feature trio** from `src/core/lgb_v4.py` (co-occurrence, route,
   momentum)
6. **Validate with walk-forward**, not single-week
7. **Declare saturation** after 3-4 versions with no F1 gain

---

## License

MIT — see `LICENSE`.

## Citation

If you use this methodology in your work, please reference the repository.

```bibtex
@misc{sku-replenishment-recommender,
  title={SKU Replenishment Recommender: Per-Channel Weekly SKU Recommendation},
  year={2026},
  license={MIT},
}
```