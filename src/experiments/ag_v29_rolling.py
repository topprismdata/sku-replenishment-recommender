"""
AG v29 — 滚动验证 (4 个目标周)
==============================
确认 v28 参数在多个周都稳定, 不是 W23 过拟合

每周用前 13 周训练:
- W20: train W7-W19
- W21: train W8-W20
- W22: train W9-W21
- W23: train W10-W22 (已知 F1=73.9%)

用 v28 的最优 N 参数:
  channel_a: base=7.1, mult=0.73
  channel_b: base=3.4, mult=1.01
  channel_c: base=1.1, mult=1.17
  channel_d: base=3.8, mult=0.80
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
# 定位项目根目录 (含数据文件的目录, 有 lgb_v4.py 或 orders_2026.csv)
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(5):  # 向上找最多 5 层
    if os.path.exists(os.path.join(_ROOT, 'orders_2026.csv')) or os.path.exists(os.path.join(_ROOT, 'src', 'core', 'lgb_v4.py')):
        break
    _ROOT = os.path.dirname(_ROOT)
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'core'))
warnings.filterwarnings('ignore')

from lgb_v4 import load_data, build_cooccur_matrix, CHANNEL_N_CAP, CHANNEL_MAP
from ag_v13 import load_all_static, load_weekly_with_revenue
from ag_v20 import build_features_v20
from autogluon.tabular import TabularPredictor

# v28 最优 N 参数
BEST_N = {
    'channel_a': (7.1, 0.73),
    'channel_b': (3.4, 1.01),
    'channel_c': (1.1, 1.17),
    'channel_d': (3.8, 0.80),
}

TARGET_WEEKS = [20, 21, 22, 23]


def evaluate_n(candidates, valid_skus, channel, base, mult):
    tr, ta, th = 0, 0, 0
    cap = CHANNEL_N_CAP[channel]
    for c in set(candidates['OUTLET']) & set(valid_skus):
        cd = candidates[candidates['OUTLET'] == c]
        typical = cd['cust_typical_n'].iloc[0]
        N = int(min(max(base + mult * typical, 5), cap))
        rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
        act = valid_skus[c]
        tr += len(rec); ta += len(act); th += len(rec & act)
    p = th / max(tr, 1); r = th / max(ta, 1)
    return 2 * p * r / max(p + r, 1e-8)


def main():
    print("=" * 60)
    print("AG v29 滚动验证 (4 目标周 W20-W23)")
    print("=" * 60)

    weekly = load_weekly_with_revenue()
    prod_feat, cust_feat = load_all_static()
    prod_raw = pd.read_csv("products.csv", low_memory=False)
    prod_raw['size_ml'] = pd.to_numeric(prod_raw['PACKAGESIZE_NAME_CN'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    prod_raw['sub_unit_num'] = pd.to_numeric(prod_raw['SUB_UNIT'], errors='coerce').fillna(1)
    prod_raw['total_ml'] = prod_raw['size_ml'] * prod_raw['sub_unit_num']
    weekly = weekly.merge(
        prod_raw[['ARTICLE_NO', 'size_ml', 'sub_unit_num', 'total_ml', 'CALORIE_INDICATOR_NAME_CN']].rename(
            columns={'ARTICLE_NO': 'ARTICLE', 'CALORIE_INDICATOR_NAME_CN': 'calorie_cat'}),
        on='ARTICLE', how='left')

    all_results = {}

    for tw in TARGET_WEEKS:
        tws = list(range(tw - 13, tw))  # 前 13 周
        vs = weekly[weekly['week'] == tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

        print(f"\n{'=' * 50}")
        print(f"目标周 W{tw} (训练 W{tws[0]}-W{tws[-1]})")
        print(f"{'=' * 50}")

        ch_f1s = []
        for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
            cooccur = build_cooccur_matrix(weekly, ch, tws)
            candidates, feat, cat_feat = build_features_v20(weekly, ch, tws, tw, cooccur, prod_feat, cust_feat)

            # 容量特征
            prod_stats = weekly[weekly['sub_channel'] == ch].groupby('ARTICLE').agg(
                size_ml=('size_ml', 'first'), sub_unit_num=('sub_unit_num', 'first'), total_ml=('total_ml', 'first')
            ).reset_index()
            candidates = candidates.merge(prod_stats, on='ARTICLE', how='left')
            candidates[['size_ml', 'sub_unit_num', 'total_ml']] = candidates[['size_ml', 'sub_unit_num', 'total_ml']].fillna(0)
            candidates = candidates.merge(weekly.groupby('ARTICLE')['calorie_cat'].first().reset_index(), on='ARTICLE', how='left')
            candidates['calorie_cat'] = candidates['calorie_cat'].fillna('-').astype('category')
            feat = feat + ['size_ml', 'sub_unit_num', 'total_ml']
            cat_feat = cat_feat + ['calorie_cat']

            if 'prob' in candidates.columns:
                candidates = candidates.drop(columns=['prob'])

            train_df = candidates[feat + cat_feat + ['label']].copy()
            t0 = time.time()
            model = TabularPredictor(
                label='label', problem_type='binary', eval_metric='roc_auc',
                path=f'ag_v29_{ch}_w{tw}', verbosity=0,
            ).fit(
                train_data=train_df,
                hyperparameters={'GBM': [{'num_leaves': 63, 'min_child_samples': 100},
                                          {'num_leaves': 31, 'min_child_samples': 50}],
                             'CAT': {}, 'XGB': {},
                             'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}], 'XT': {}},
                time_limit=240, excluded_model_types=['NN_TORCH', 'FASTAI'],
            )
            all_feat = feat + cat_feat
            proba = model.predict_proba(candidates[all_feat])
            candidates['prob'] = proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba[:, 1]

            base, mult = BEST_N[ch]
            f1 = evaluate_n(candidates, vs, ch, base, mult)
            ch_f1s.append(f1)
            all_results.setdefault(ch, []).append(f1)
            print(f"  {ch}: F1={100 * f1:.1f}% ({time.time() - t0:.0f}s)")

        avg = np.mean(ch_f1s)
        all_results.setdefault('avg', []).append(avg)
        print(f"  >>> W{tw} 平均: {100 * avg:.1f}%")

    # 滚动验证汇总
    print(f"\n{'=' * 60}")
    print("滚动验证汇总 (4 周)")
    print(f"{'=' * 60}")
    print(f"{'channel':<8}", end='')
    for tw in TARGET_WEEKS:
        print(f" {'W' + str(tw):>7}", end='')
    print(f" {'mean':>7} {'std':>6}")
    for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d', 'avg']:
        vals = all_results[ch]
        label = '平均' if ch == 'avg' else ch
        print(f"{label:<8}", end='')
        for v in vals:
            print(f" {100 * v:>6.1f}%", end='')
        print(f" {100 * np.mean(vals):>6.1f}% {100 * np.std(vals):>5.1f}pp")


if __name__ == "__main__":
    main()
