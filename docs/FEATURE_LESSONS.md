# Retail SKU Recommendation — Feature Engineering Lessons

> Cross-competition insights distilled from Instacart, Store Sales, Favorita,
> M5 Forecasting, and H&M Silver Medal solutions.
> Applicable to any **per-customer SKU recommendation** problem with F1/AUC metric.

---

## 1. Lessons from each competition

### Instacart (top 2%, F1 metric — closest analog to retail recommendation)

1. **User-product interaction features** (the core):
   - `reorder_ratio` = purchase count / total order count
   - Purchase position (which item in cart)
   - Purchase interval mean / std (behavioral consistency)
   - **`UP_order_strike`** (single strongest feature): `1 / 2^(reverse_order_idx)` — exponential recency weighting

2. **product2vec embeddings**: treat orders as sentences, products as words
3. **Hierarchical interactions**: user × category, user × aisle purchase ratio
4. **Two-stage modeling**: predict probabilities in stage 1, use them as features in stage 2
5. **Candidate restriction**: only recommend historically purchased products (exclude cold items)

### Store Sales (time-series retail)

1. **Multi-window rolling** (7/14/28-day mean + std)
2. **EWM** (exponentially weighted moving average)
3. **YoY lag** (364-day, captures yearly seasonality)
4. **Promotion lag** (requires promotion data)
5. **External covariates** (oil price / holidays / transactions)
6. **Payday features** (only meaningful at daily granularity)

### Favorita (1st place solution)

1. **Days since last appearance** (recency gap)
2. **Mean value delta between adjacent windows** (trend change)
3. **Multi-level aggregation** (family-level, store-level)

### M5 Forecasting

1. **Rolling multi-statistics** (mean / std / min / max / skew)
2. **Price elasticity** (price change → sales change)
3. **Event features** (promotion / event day flags)

---

## 2. Feature priority framework (Instacart-validated P0)

These are the **must-have** features for any retail SKU recommendation task:

- [ ] **`UP_order_strike`**: `1 / 2^(reverse_period_idx)` — recency exponential decay
- [ ] **`reorder_ratio`**: purchase count / customer total purchase periods
- [ ] **Purchase interval mean + std**: behavioral consistency
- [ ] **Window mean delta**: `roll_mean_3 - roll_mean_8` (short-term vs long-term trend change)

### P1 (Store Sales / Favorita validated)

- [ ] **Category × customer interaction**: customer purchase ratio per category
- [ ] **Product embeddings** (product2vec or simplified co-occurrence matrix)
- [ ] **Multi-statistics rolling**: min / max / skew (not just mean / std)

### P2 (uncertain, validate before adding)

- [ ] **Two-stage modeling**: stage-1 probability → stage-2 feature
- [ ] **Hierarchical aggregation**: channel-level / category-level customer averages
- [ ] **Customer RFM** (Recency / Frequency / Monetary quintiles)

---

## 3. M5 Forecasting lessons (additions)

1. **Tweedie objective** — ideal for sparse sales with many zeros (intermittent demand)
2. **Long-horizon lags (>28 days)** — use only distant lags, leave recent to rolling
3. **Price features** — min / max / mean / std / normalized price
4. **Per-store splitting** — train one model per store when patterns diverge
5. **Recursive forecasting** — fill lag with predictions for week 2+, or use `rolling >= 7 day window` to reduce error propagation

---

## 4. H&M Silver Medal lessons (recap+rank architectures)

1. **Recall + rank two-stage**: 2 strategies, ~50 candidates per user, then ranker
2. **Ranker ensembles**: LGB ranker + LGB classifier + DNN, fused
3. **Word2Vec embeddings**: item / user vectors (CBOW + Skip-gram)
4. **DSSM / YouTube DNN**: deep retrieval for user/item vectors
5. **Time decay**: only train on recent ~4 weeks for fast-moving catalogs

---

## 5. Methodology takeaways (highest impact)

1. **`UP_order_strike` is the single strongest feature** — not complex models, just simple exponential decay
2. **Candidate restriction matters** — only recommend historical purchases (Instacart fully excludes new products)
3. **Two-stage modeling helps** — but watch for leakage
4. **Multi-level aggregation may overfit** on day-specific models
5. **AG default config usually wins** — don't manually pile features; validate before adding

---

## 6. Optimization directions to test (P0)

1. **Direct F1 optimization** (Instacart top 2/4/6 consensus) — change `eval_metric`
2. **Tweedie objective** (M5) — for sparse targets
3. **LGB ranker** (H&M) — purpose-built for ranking, may beat classifier
4. **Threshold optimization** (Instacart) — tune threshold post-prediction to maximize F1