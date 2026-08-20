"""
AG v26 — 13周训练周期 (业务经验: 低频客户需要长周期)
=====================================================
业务经验: 低频客户每周买<5 SKU, 4周内只出现2-3次, 模型学不到
13周 = 3个月, 覆盖大部分低频客户的完整周期

对比: 4周训练 vs 13周训练, 看低频客户表现
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


def main():
    print("="*60)
    print("AG v26 (13周训练周期: 低频客户需要长周期)")
    print("="*60)

    weekly = load_weekly_with_revenue()
    prod_feat, cust_feat = load_all_static()
    prod_raw = pd.read_csv("products.csv", low_memory=False)
    prod_raw['size_ml'] = pd.to_numeric(prod_raw['PACKAGESIZE_NAME_CN'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    prod_raw['sub_unit_num'] = pd.to_numeric(prod_raw['SUB_UNIT'], errors='coerce').fillna(1)
    prod_raw['total_ml'] = prod_raw['size_ml'] * prod_raw['sub_unit_num']
    weekly = weekly.merge(prod_raw[['ARTICLE_NO','size_ml','sub_unit_num','total_ml','CALORIE_INDICATOR_NAME_CN']].rename(columns={'ARTICLE_NO':'ARTICLE','CALORIE_INDICATOR_NAME_CN':'calorie_cat'}), on='ARTICLE', how='left')

    tw, tws_short = 23, list(range(TRAIN_START_WEEK, TARGET_WEEK))  # 4周训练
    tws_long = list(range(TRAIN_START_WEEK, TARGET_WEEK))           # 13周训练 (W10-W22)

    vs = weekly[weekly['week']==tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    for tws_name, tws in [('4周', tws_short), ('13周', tws_long)]:
        print(f"\n{'='*50}")
        print(f"训练周期: {tws_name} (W{tws[0]}-W{tws[-1]})")
        print(f"{'='*50}")

        all_f1 = []
        for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
            cooccur = build_cooccur_matrix(weekly, ch, tws)
            candidates, feat, cat_feat = build_features_v20(weekly, ch, tws, tw, cooccur, prod_feat, cust_feat)
            # 加容量特征
            prod_stats = weekly[weekly['sub_channel']==ch].groupby('ARTICLE').agg(
                size_ml=('size_ml','first'), sub_unit_num=('sub_unit_num','first'), total_ml=('total_ml','first')
            ).reset_index()
            candidates = candidates.merge(prod_stats, on='ARTICLE', how='left')
            candidates[['size_ml','sub_unit_num','total_ml']] = candidates[['size_ml','sub_unit_num','total_ml']].fillna(0)
            candidates = candidates.merge(weekly.groupby('ARTICLE')['calorie_cat'].first().reset_index(), on='ARTICLE', how='left')
            candidates['calorie_cat'] = candidates['calorie_cat'].fillna('-').astype('category')
            feat = feat + ['size_ml','sub_unit_num','total_ml']
            cat_feat = cat_feat + ['calorie_cat']

            train_df = candidates[feat + cat_feat + ['label']].copy()
            t0 = time.time()
            model = TabularPredictor(
                label='label', problem_type='binary', eval_metric='roc_auc',
                path=f'ag_v26_{ch}_{tws_name}_w{tw}', verbosity=0,
            ).fit(
                train_data=train_df,
                hyperparameters={'GBM':[{'num_leaves':63,'min_child_samples':100},{'num_leaves':31,'min_child_samples':50}],
                                 'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
                time_limit=300, excluded_model_types=['NN_TORCH','FASTAI'],
            )
            all_feat = feat + cat_feat
            proba = model.predict_proba(candidates[all_feat])
            candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]

            # 固定 N (v20 方式)
            cap = CHANNEL_N_CAP[ch]
            tr, ta, th = 0, 0, 0
            for c in set(candidates['OUTLET']) & set(vs):
                cd = candidates[candidates['OUTLET']==c]
                N = int(min(max(cd['cust_typical_n'].iloc[0]+5, 10), cap))
                rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
                act = vs[c]
                tr += len(rec); ta += len(act); th += len(rec & act)
            p = th/max(tr,1); r = th/max(ta,1); f1 = 2*p*r/max(p+r,1e-8)
            all_f1.append(f1)
            print(f"  {ch}: F1={100*f1:.0f}% (P={100*p:.0f}% R={100*r:.0f}%)")

        avg = np.mean(all_f1)
        print(f"  >>> {tws_name}平均 F1: {100*avg:.1f}%")


if __name__ == "__main__":
    main()
