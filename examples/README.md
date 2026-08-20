# Examples / Quickstart

> ⚠️ **This directory is a placeholder.** Real usage examples require
> the actual project data (orders / customers / products), which is **NOT**
> shipped with this repo for client confidentiality reasons.

## How to use this repo on your own data

### 1. Prepare your data
You need 3 CSV files (column names are configurable in `src/core/lgb_v4.py`):

| File | Required columns |
|---|---|
| `orders.csv` | `OUTLET`, `ARTICLE`, `ORDER_DATE`, `SETTLE_CASE`, `CHANNEL`, `PS_ROUTE` |
| `customers.csv` | `OUTLET`, `CHANNEL`, `PS_ROUTE` |
| `products.csv` | `ARTICLE`, ... (any product attributes) |

### 2. Configure channel mapping
Edit `CHANNEL_MAP` in `src/core/lgb_v4.py` to match your own channel taxonomy:

```python
CHANNEL_MAP = {
    'channel_a': ['A70', 'A72'],
    'channel_b': ['F21', 'F20', 'F22'],
    'channel_c': ['G12', 'G10'],
    'channel_d': ['A60', 'A53'],
}
```

### 3. Run
```bash
cd src/core
python lgb_v4.py
```

The script outputs per-channel walk-forward F1 scores and per-week logs.

## Key features you can swap

| Generic variable | Original (de-sensitized) | Meaning |
|---|---|---|
| `OUTLET` | customer/store id | Customer identifier |
| `ARTICLE` | sku id | SKU identifier |
| `CHANNEL` | channel code | Distribution channel |
| `PS_ROUTE` | route id | Sales route / cluster |
| `SETTLE_CASE` | settled cases | Quantity metric |

See `docs/LESSONS_LEARNED.md` for the full methodology and gotchas.