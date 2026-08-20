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