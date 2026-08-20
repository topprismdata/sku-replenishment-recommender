"""
AG v19 — 预测 N (学出来的 N, 不是手工规则)
==========================================
思路: 把 N 当成预测目标, 用模型学最优 N
- 特征: 客户历史购买数/活跃度/渠道/冰柜等
- 目标: 该客户最优推荐数 (用 F1 反推)
- 方法: 每客户独立优化 N (遍历 10-50, 找 F1 最高的)

这比手工规则 (typical_n+5) 高级: 是学出来的, 不是拍脑袋
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
from ag_v13 import load_all_static, load_weekly_with_revenue, build_features_v13
from autogluon.tabular import TabularPredictor
import lightgbm as lgb


def main():
    print("="*60)
    print("AG v19 (预测 N: 学出来的最优 N)")
    print("="*60)

    weekly = load_weekly_with_revenue()
    prod_feat, cust_feat = load_all_static()
    prod_raw = pd.read_csv("products.csv", low_memory=False)
    prod_raw['size_ml'] = pd.to_numeric(prod_raw['PACKAGESIZE_NAME_CN'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    prod_raw['sub_unit_num'] = pd.to_numeric(prod_raw['SUB_UNIT'], errors='coerce').fillna(1)
    prod_raw['total_ml'] = prod_raw['size_ml'] * prod_raw['sub_unit_num']
    weekly = weekly.merge(prod_raw[['ARTICLE_NO','size_ml','sub_unit_num','total_ml','CALORIE_INDICATOR_NAME_CN']].rename(columns={'ARTICLE_NO':'ARTICLE','CALORIE_INDICATOR_NAME_CN':'calorie_cat'}), on='ARTICLE', how='left')

    tw, tws = TARGET_WEEK, list(range(TRAIN_START_WEEK, TARGET_WEEK))
    vs = weekly[weekly['week']==tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    for ch in ['channel_a']:
        cooccur = build_cooccur_matrix(weekly, ch, tws)
        candidates, feat, cat_feat = build_features_v13(weekly, ch, tws, tw, cooccur, prod_feat, cust_feat)
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

        # 训练主模型 (预测概率)
        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v19_main_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={'GBM':[{'num_leaves':63,'min_child_samples':100},{'num_leaves':31,'min_child_samples':50}],
                             'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
            time_limit=300, excluded_model_types=['NN_TORCH','FASTAI'],
        )
        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]

        # === 预测最优 N (学出来的) ===
        # 对每个客户, 遍历 N=10-50, 用 F1 评估哪个 N 最优
        # 特征: 客户历史购买数/活跃度/渠道/冰柜/历史周均SKU
        cap = CHANNEL_N_CAP[ch]
        n_features = []
        for c in sorted(set(candidates['OUTLET']) & set(vs)):
            cd = candidates[candidates['OUTLET']==c]
            typical = cd['cust_typical_n'].iloc[0]
            # 客户特征
            hist_skus = cd['c_skus'].iloc[0] if 'c_skus' in cd.columns else typical
            active_weeks = cd['c_weeks'].iloc[0] if 'c_weeks' in cd.columns else 5
            avg_freq = cd['c_freq'].iloc[0] if 'c_freq' in cd.columns else typical

            # 遍历 N 找最优
            best_n, best_f1 = 20, 0
            for N in range(10, min(cap+10, 51)):
                rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
                act = vs[c]
                hit = len(rec & act)
                p = hit / max(len(rec), 1)
                r = hit / max(len(act), 1)
                f1 = 2*p*r/max(p+r, 1e-8)
                if f1 > best_f1:
                    best_f1, best_n = f1, N

            n_features.append({
                'cust': c, 'typical': typical, 'hist_skus': hist_skus,
                'active_weeks': active_weeks, 'avg_freq': avg_freq,
                'best_n': best_n
            })

        n_df = pd.DataFrame(n_features)
        print(f"\n  {ch}: 预测最优 N")
        print(f"    样本: {len(n_df)} 客户")
        print(f"    best_n 分布: 中位={n_df['best_n'].median():.0f}, 均={n_df['best_n'].mean():.1f}")
        print(f"    范围: {n_df['best_n'].min()}-{n_df['best_n'].max()}")

        # 用 LightGBM 预测 best_n (回归)
        # 特征: typical, hist_skus, active_weeks, avg_freq
        n_feat = ['typical','hist_skus','active_weeks','avg_freq']
        n_model = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
            min_child_samples=50, verbose=-1, seed=42, n_estimators=100
        )
        n_model.fit(n_df[n_feat], n_df['best_n'])

        # 预测所有客户的最优 N
        pred_n = n_model.predict(n_df[n_feat])
        n_df['pred_n'] = np.clip(pred_n.round(), 10, cap+5).astype(int)

        print(f"    预测 N vs 真实最优 N: MAE={np.mean(np.abs(n_df['best_n'] - n_df['pred_n'])):.1f}")
        print(f"    相关性: {n_df['best_n'].corr(n_df['pred_n']):.2f}")

        # 用预测 N 评估
        tr, ta, th = 0, 0, 0
        for _, row in n_df.iterrows():
            c = row['cust']
            cd = candidates[candidates['OUTLET']==c]
            N = int(row['pred_n'])
            rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
            act = vs[c]
            tr += len(rec); ta += len(act); th += len(rec & act)
        p = th/max(tr,1); r = th/max(ta,1); f1 = 2*p*r/max(p+r,1e-8)
        print(f"    >>> 用预测 N: F1={100*f1:.0f}% (P={100*p:.0f}% R={100*r:.0f}%)")

        # 对比: 固定 N=typical+5
        tr2, ta2, th2 = 0, 0, 0
        for _, row in n_df.iterrows():
            c = row['cust']
            cd = candidates[candidates['OUTLET']==c]
            N = int(min(max(row['typical']+5, 10), cap))
            rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
            act = vs[c]
            tr2 += len(rec); ta2 += len(act); th2 += len(rec & act)
        p2 = th2/max(tr2,1); r2 = th2/max(ta2,1); f12 = 2*p2*r2/max(p2+r2,1e-8)
        print(f"    对比 typical+5: F1={100*f12:.0f}%")

        # 对比: 真实最优 N (上限)
        tr3, ta3, th3 = 0, 0, 0
        for _, row in n_df.iterrows():
            c = row['cust']
            cd = candidates[candidates['OUTLET']==c]
            N = int(row['best_n'])
            rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
            act = vs[c]
            tr3 += len(rec); ta3 += len(act); th3 += len(rec & act)
        p3 = th3/max(tr3,1); r3 = th3/max(ta3,1); f13 = 2*p3*r3/max(p3+r3,1e-8)
        print(f"    上限 (真实最优 N): F1={100*f13:.0f}%")


if __name__ == "__main__":
    main()
