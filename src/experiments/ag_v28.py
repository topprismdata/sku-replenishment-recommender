"""
AG v28 — 13周训练 + 分渠道 Optuna 调 N
======================================
基于 v26 (13周训练, F1=72.9%) 进一步优化:
1. 13周训练 (W10-W22) - 已验证最大单步提升 (+4pp)
2. 分渠道 Optuna 调 N 公式 (base + mult)
   - 各渠道客户行为差异大, 不应共享 N
3. AG 超参用 v26 已验证参数 (num_leaves 63+31, min_child 100+50)

对比: v26 固定 N (typical+5) vs v28 Optuna N (per channel)
"""
# ============================================================
# Week configuration - adjust to your data
# ============================================================
TRAIN_START_WEEK = 1   # first training week (inclusive)
TARGET_WEEK = 1        # first target / prediction week (inclusive)
# Extend with: TARGET_WEEK_2 = TARGET_WEEK + 1, etc. for walk-forward
# ============================================================

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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_n_params(candidates, valid_skus, channel, base, mult):
    """用给定的 N 公式评估 F1"""
    tr, ta, th = 0, 0, 0
    for c in set(candidates['OUTLET']) & set(valid_skus):
        cd = candidates[candidates['OUTLET'] == c]
        typical = cd['cust_typical_n'].iloc[0]
        cap = CHANNEL_N_CAP[channel]
        N = int(min(max(base + mult * typical, 5), cap))
        rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
        act = valid_skus[c]
        tr += len(rec); ta += len(act); th += len(rec & act)
    p = th / max(tr, 1); r = th / max(ta, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return f1


def main():
    print("=" * 60)
    print("AG v28 (13周训练 + 分渠道 Optuna 调 N)")
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

    tw, tws = TARGET_WEEK, list(range(TRAIN_START_WEEK, TARGET_WEEK))  # 13周训练
    vs = weekly[weekly['week'] == tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    results_v26 = []  # 固定 N = typical+5
    results_v28 = []  # Optuna N
    best_params = {}

    for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
        print(f"\n{'=' * 50}")
        print(f"渠道: {ch}")
        print(f"{'=' * 50}")

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

        # 删除旧 prob 列
        if 'prob' in candidates.columns:
            candidates = candidates.drop(columns=['prob'])

        train_df = candidates[feat + cat_feat + ['label']].copy()

        # 训练 (v26 参数)
        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v28_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={'GBM': [{'num_leaves': 63, 'min_child_samples': 100},
                                      {'num_leaves': 31, 'min_child_samples': 50}],
                             'CAT': {}, 'XGB': {},
                             'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}], 'XT': {}},
            time_limit=300, excluded_model_types=['NN_TORCH', 'FASTAI'],
        )
        print(f"  训练完成: {time.time() - t0:.0f}s, 最佳模型: {model.model_best}")

        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba[:, 1]

        # v26 基线: N = typical + 5
        f1_v26 = evaluate_n_params(candidates, vs, ch, 5, 1.0)
        results_v26.append(f1_v26)
        print(f"  v26 N (typical+5): F1={100 * f1_v26:.1f}%")

        # v28 Optuna 调 N
        def objective_n(trial):
            base = trial.suggest_float('base', 0, 15)
            mult = trial.suggest_float('mult', 0.3, 2.0)
            return evaluate_n_params(candidates, vs, ch, base, mult)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_n, n_trials=50)
        bp = study.best_params
        best_params[ch] = bp
        f1_v28 = study.best_value
        results_v28.append(f1_v28)
        print(f"  v28 Optuna N: base={bp['base']:.1f}, mult={bp['mult']:.2f} → F1={100 * f1_v28:.1f}%")
        print(f"  提升: {100 * (f1_v28 - f1_v26):+.1f}pp")

    # 汇总
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    print(f"{'channel':<8} {'v26(typical+5)':>15} {'v28(Optuna N)':>15} {'提升':>8}")
    for i, ch in enumerate(['channel_a', 'channel_b', 'channel_c', 'channel_d']):
        diff = results_v28[i] - results_v26[i]
        print(f"{ch:<8} {100 * results_v26[i]:>14.1f}% {100 * results_v28[i]:>14.1f}% {100 * diff:>+7.1f}pp")
    print(f"{'平均':<8} {100 * np.mean(results_v26):>14.1f}% {100 * np.mean(results_v28):>14.1f}% {100 * (np.mean(results_v28) - np.mean(results_v26)):>+7.1f}pp")

    print(f"\n最优 N 参数 (可用于提交):")
    for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
        bp = best_params[ch]
        print(f"  {ch}: base={bp['base']:.1f}, mult={bp['mult']:.2f}")


if __name__ == "__main__":
    main()
