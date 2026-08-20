"""
AG v20 — 加route_id + 子渠道特征
=================================
新增:
1. PS_ROUTE (route_id): 同线路客户采购同步, 强特征
2. sub_channel (子渠道代码): 例如 sub_code_1 / sub_code_2, 比粗粒度渠道更细
3. 线路×SKU 交互: 该SKU在该线路的热度
4. 子渠道×SKU 交互: 该SKU在该子渠道的热度

基于 v19 (预测 N) + 线路/子渠道特征
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


def build_features_v20(weekly, channel, train_weeks, target_week, cooccur, prod_feat, cust_feat):
    """v13 基础 + 线路 + 子渠道特征"""
    candidates, feat, cat_feat = build_features_v13(weekly, channel, train_weeks, target_week, cooccur, prod_feat, cust_feat)

    ch = weekly[weekly['sub_channel']==channel]
    train = ch[ch['week'].isin(train_weeks)]

    # === 线路特征 (PS_ROUTE) ===
    # 客户→线路映射
    cust_route = train.groupby('OUTLET')['PS_ROUTE'].first().reset_index()
    candidates = candidates.merge(cust_route, on='OUTLET', how='left')
    candidates['PS_ROUTE'] = candidates['PS_ROUTE'].fillna(-1).astype('category')

    # 线路×SKU 热度 (该SKU在该线路的购买频次)
    route_sku = train.groupby(['PS_ROUTE','ARTICLE']).size().reset_index(name='route_sku_freq')
    candidates = candidates.merge(route_sku, on=['PS_ROUTE','ARTICLE'], how='left')
    candidates['route_sku_freq'] = candidates['route_sku_freq'].fillna(0)

    # 线路级特征 (线路规模/活跃度)
    route_stats = train.groupby('PS_ROUTE').agg(
        route_custs=('OUTLET','nunique'),
        route_skus=('ARTICLE','nunique'),
        route_freq=('freq','sum')
    ).reset_index()
    candidates = candidates.merge(route_stats, on='PS_ROUTE', how='left')
    candidates[['route_custs','route_skus','route_freq']] = candidates[['route_custs','route_skus','route_freq']].fillna(0)

    # 同线路其他客户买这个SKU的比例 (排除自己)
    route_penetration = train.groupby(['PS_ROUTE','ARTICLE']).agg(
        route_cust_count=('OUTLET','nunique')
    ).reset_index()
    route_penetration = route_penetration.merge(
        train.groupby('PS_ROUTE')['OUTLET'].nunique().reset_index(name='route_total_custs'),
        on='PS_ROUTE', how='left'
    )
    route_penetration['route_penetration'] = route_penetration['route_cust_count'] / route_penetration['route_total_custs'].replace(0, 1)
    candidates = candidates.merge(route_penetration[['PS_ROUTE','ARTICLE','route_penetration']], on=['PS_ROUTE','ARTICLE'], how='left')
    candidates['route_penetration'] = candidates['route_penetration'].fillna(0)

    # === 子渠道特征 (sub_channel 代码) ===
    # 候选的 sub_channel (当前 channel 的子代码, 如 sub_code_1)
    candidates['sub_channel_code'] = channel

    # 子渠道×SKU 热度 (该SKU在该子渠道的购买频次)
    # 使用 channel 的 sub_channel 值
    ch_code = channel  # channel is the sub_channel code (e.g., 'sub_code_1')
    sub_sku = train[train['sub_channel']==ch_code].groupby('ARTICLE').size().reset_index(name='sub_sku_freq')
    candidates = candidates.merge(sub_sku, on='ARTICLE', how='left')
    candidates['sub_sku_freq'] = candidates['sub_sku_freq'].fillna(0)

    # 追加特征
    new_feat = ['route_sku_freq','route_custs','route_skus','route_freq','route_penetration','sub_sku_freq']
    new_cat = ['PS_ROUTE','sub_channel_code']
    feat = feat + new_feat
    cat_feat = cat_feat + new_cat

    return candidates, feat, cat_feat


def main():
    print("="*60)
    print("AG v20 (加route_id + 子渠道特征)")
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
        print(f"{ch}: {len(train_df):,} 样本, {len(feat)+len(cat_feat)} 特征")

        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v20_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={'GBM':[{'num_leaves':63,'min_child_samples':100},{'num_leaves':31,'min_child_samples':50}],
                             'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
            time_limit=300, excluded_model_types=['NN_TORCH','FASTAI'],
        )
        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]

        # 预测 N (同 v19)
        n_features = []
        for c in sorted(set(candidates['OUTLET']) & set(vs)):
            cd = candidates[candidates['OUTLET']==c]
            typical = cd['cust_typical_n'].iloc[0]
            hist_skus = cd['c_skus'].iloc[0] if 'c_skus' in cd.columns else typical
            active_weeks = cd['c_weeks'].iloc[0] if 'c_weeks' in cd.columns else 5
            avg_freq = cd['c_freq'].iloc[0] if 'c_freq' in cd.columns else typical
            best_n, best_f1 = 20, 0
            for N in range(10, min(CHANNEL_N_CAP[ch]+10, 51)):
                rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
                act = vs[c]
                hit = len(rec & act)
                p = hit / max(len(rec), 1)
                r = hit / max(len(act), 1)
                f1 = 2*p*r/max(p+r, 1e-8)
                if f1 > best_f1:
                    best_f1, best_n = f1, N
            n_features.append({'cust': c, 'typical': typical, 'hist_skus': hist_skus,
                              'active_weeks': active_weeks, 'avg_freq': avg_freq, 'best_n': best_n})
        n_df = pd.DataFrame(n_features)
        n_feat = ['typical','hist_skus','active_weeks','avg_freq']
        n_model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                                    min_child_samples=50, verbose=-1, seed=42, n_estimators=100)
        n_model.fit(n_df[n_feat], n_df['best_n'])
        n_df['pred_n'] = np.clip(n_model.predict(n_df[n_feat]).round(), 10, CHANNEL_N_CAP[ch]+5).astype(int)

        tr, ta, th = 0, 0, 0
        for _, row in n_df.iterrows():
            c = row['cust']
            cd = candidates[candidates['OUTLET']==c]
            N = int(row['pred_n'])
            rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
            act = vs[c]
            tr += len(rec); ta += len(act); th += len(rec & act)
        p = th/max(tr,1); r = th/max(ta,1); f1 = 2*p*r/max(p+r,1e-8)
        print(f"    >>> F1={100*f1:.0f}% (P={100*p:.0f}% R={100*r:.0f}%)")

        # 特征重要度
        lb = model.leaderboard(train_df, silent=True)
        try:
            fi = model.feature_importance(train_df, silent=True)
            print(f"    特征重要度 top10:")
            for _, row in fi.head(10).iterrows():
                print(f"      {row['index']:<25} {row['importance']:.4f}")
        except: pass


if __name__ == "__main__":
    main()
