"""
AG v13 — 全特征 EDA 启发 (冰柜+等级+卡路里+金额+包装规格)
============================================================
新增 P0 特征:
1. WITH_COOLER (冰柜门数) — 决定能买冷饮
2. ABCD_CODE (客户等级) — 购买力
3. CALORIE_INDICATOR (卡路里) — 无糖/普通偏好
4. SETTLE_NET_REVENUE (净收入) — 金额特征
5. PACKAGESIZE (包装规格) — 330ml vs 680ml
6. OPEN_DATE → 客户年龄
7. n_provinces (产品渗透度)
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
from ag_v9 import build_features_v9
from autogluon.tabular import TabularPredictor


def load_all_static():
    """加载客户表+产品表的完整特征"""
    prod = pd.read_csv("products.csv")
    prod_features = prod[['ARTICLE_NO','BRAND_NAME_CN','BEVCAT_NAME_CN',
        'CALORIE_INDICATOR_NAME_CN','PACKAGESIZE_NAME_CN','PACKAGETYPE_NAME_CN',
        'SUB_UNIT','WEIGHT','LAUNCH_BEFORE2010','NO_SALES_PRODUCT']].copy()
    prod_features.columns = ['ARTICLE','brand','category','calorie','pkg_size','pkg_type',
                             'sub_unit','weight','is_old','no_sales']
    # 上市省份数
    launch_cols = [c for c in prod.columns if c.startswith('LAUNCH_DATE_')]
    prod_features['n_provinces'] = prod[launch_cols].notna().sum(axis=1).values

    cust = pd.read_csv("customers.csv")
    cust_features = cust[['OUTLET_NO','ABCD_CODE','WITH_COOLER','OPEN_DATE',
                          'REGION_CODE','CHANNELGROUP_CODE']].copy()
    cust_features.columns = ['OUTLET','abcd','cooler','open_date','region','ch_group']
    # 客户年龄 (年)
    cust_features['open_date'] = pd.to_numeric(cust_features['open_date'], errors='coerce')
    cust_features['cust_age'] = 2026 - (cust_features['open_date'] // 10000)
    cust_features['cust_age'] = cust_features['cust_age'].fillna(20)

    return prod_features, cust_features


def load_weekly_with_revenue():
    """加载带净收入的周数据"""
    orders = pd.read_csv("orders_2026.csv",
        usecols=['OUTLET','ARTICLE','ORDER_DATE','SETTLE_CASE','SETTLE_NET_REVENUE',
                 'CHANNEL','PS_ROUTE','FREE_CASE'], low_memory=False)
    orders['ORDER_DATE'] = pd.to_datetime(orders['ORDER_DATE'], errors='coerce')
    orders = orders[orders['SETTLE_CASE']>0].copy()
    orders['week'] = orders['ORDER_DATE'].dt.isocalendar().week.astype(int)
    orders = orders[orders['ORDER_DATE'].dt.year==2026]
    CODE_TO_CH = {c:ch for ch,cs in CHANNEL_MAP.items() for c in cs}
    orders['sub_channel'] = orders['CHANNEL'].map(CODE_TO_CH)
    sub = pd.read_csv("pre_may_recommendations/pre_may_recommendations.csv", usecols=['customer_id'])
    orders = orders[orders['sub_channel'].notna() & orders['OUTLET'].isin(set(sub['customer_id']))]

    weekly = orders.groupby(['OUTLET','ARTICLE','week','sub_channel','PS_ROUTE']).agg(
        freq=('SETTLE_CASE','count'),
        revenue=('SETTLE_NET_REVENUE','sum'),
        revenue_mean=('SETTLE_NET_REVENUE','mean'),
    ).reset_index()
    # has_free: 有没有赠品
    free_agg = orders.groupby(['OUTLET','ARTICLE','week','sub_channel','PS_ROUTE'])['FREE_CASE'].apply(
        lambda x: int((x != 0).any())
    ).reset_index(name='has_free')
    weekly = weekly.merge(free_agg, on=['OUTLET','ARTICLE','week','sub_channel','PS_ROUTE'], how='left')
    weekly['has_free'] = weekly['has_free'].fillna(0)
    return weekly


def build_features_v13(weekly, channel, train_weeks, target_week, cooccur, prod_feat, cust_feat):
    """v9 基础 + 全 EDA 特征"""
    candidates, feat, cat_feat = build_features_v9(weekly, channel, train_weeks, target_week, cooccur, prod_feat[['ARTICLE','category']].copy())

    ch = weekly[weekly['sub_channel']==channel]
    train = ch[ch['week'].isin(train_weeks)]

    # === 客户特征 (客户表) ===
    candidates = candidates.merge(cust_feat[['OUTLET','cooler','abcd','cust_age','region']],
                                  on='OUTLET', how='left')
    candidates['cooler'] = candidates['cooler'].fillna(0)
    candidates['abcd'] = candidates['abcd'].fillna(99).astype('category')
    candidates['cust_age'] = candidates['cust_age'].fillna(20)
    candidates['region'] = candidates['region'].fillna('unknown').astype('category')

    # === 产品特征 (产品表) ===
    candidates = candidates.merge(
        prod_feat[['ARTICLE','calorie','pkg_size','sub_unit','n_provinces','is_old']],
        on='ARTICLE', how='left')
    candidates['calorie'] = candidates['calorie'].fillna('-').astype('category')
    candidates['pkg_size'] = candidates['pkg_size'].fillna('unknown').astype('category')
    candidates['sub_unit'] = candidates['sub_unit'].fillna(0)
    candidates['n_provinces'] = candidates['n_provinces'].fillna(0)
    candidates['is_old'] = candidates['is_old'].fillna(0)

    # === 金额特征 (订单表) ===
    if 'revenue' in weekly.columns:
        rev_stats = train.groupby(['OUTLET','ARTICLE']).agg(
            rev_mean=('revenue','mean'),
            rev_total=('revenue','sum'),
        ).reset_index()
        candidates = candidates.merge(rev_stats, on=['OUTLET','ARTICLE'], how='left')
        candidates[['rev_mean','rev_total']] = candidates[['rev_mean','rev_total']].fillna(0)
    else:
        candidates['rev_mean'] = 0
        candidates['rev_total'] = 0

    # === 促销特征 ===
    if 'has_free' in weekly.columns:
        free_stats = train.groupby(['OUTLET','ARTICLE'])['has_free'].max().reset_index(name='ever_promoted')
        candidates = candidates.merge(free_stats, on=['OUTLET','ARTICLE'], how='left')
        candidates['ever_promoted'] = candidates['ever_promoted'].fillna(0)
    else:
        candidates['ever_promoted'] = 0

    # 汇总特征列表
    new_feat = ['cooler','cust_age','sub_unit','n_provinces','is_old',
                'rev_mean','rev_total','ever_promoted']
    new_cat = ['abcd','region','calorie','pkg_size']
    feat = feat + new_feat
    cat_feat = cat_feat + new_cat

    return candidates, feat, cat_feat


def main():
    print("="*60)
    print("AG v13 (全特征EDA: 冰柜+等级+卡路里+金额+包装)")
    print("="*60)

    weekly = load_weekly_with_revenue()
    prod_feat, cust_feat = load_all_static()
    print(f"周聚合: {len(weekly):,}")
    print(f"产品特征: {len(prod_feat):,}, 客户特征: {len(cust_feat):,}")

    tw, tws = TARGET_WEEK, list(range(TRAIN_START_WEEK, TARGET_WEEK))
    vs = weekly[weekly['week']==tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    all_f1 = []
    for ch in CHANNEL_MAP:
        cooccur = build_cooccur_matrix(weekly, ch, tws)
        candidates, feat, cat_feat = build_features_v13(weekly, ch, tws, tw, cooccur, prod_feat, cust_feat)
        train_df = candidates[feat + cat_feat + ['label']].copy()
        print(f"\n  {ch}: {len(train_df):,} 样本, {len(feat)+len(cat_feat)} 特征")

        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v13_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={'GBM':[{'num_leaves':63,'min_child_samples':100},{'num_leaves':31,'min_child_samples':50}],
                             'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
            time_limit=300, excluded_model_types=['NN_TORCH','FASTAI'],
        )
        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]

        cap = CHANNEL_N_CAP[ch]
        tr,ta,th = 0,0,0
        for c in set(candidates['OUTLET']) & set(vs):
            cd = candidates[candidates['OUTLET']==c]
            N = int(min(max(cd['cust_typical_n'].iloc[0]+5,10), cap))
            rec = set(cd.nlargest(N,'prob')['ARTICLE']); act = vs[c]
            tr+=len(rec); ta+=len(act); th+=len(rec&act)
        p=th/max(tr,1); r=th/max(ta,1); f1=2*p*r/max(p+r,1e-8)
        all_f1.append(f1)
        lb = model.leaderboard(train_df, silent=True)
        print(f"    >>> F1={100*f1:.0f}% (P={100*p:.0f}% R={100*r:.0f}%) best={lb.iloc[0]['model']} ({time.time()-t0:.0f}s)")

        if ch == 'channel_a':
            try:
                fi = model.feature_importance(train_df, silent=True)
                print(f"    特征重要度 top20:")
                for _, row in fi.head(20).iterrows():
                    print(f"      {row['index']:<25} {row['importance']:.4f}")
            except: pass

    avg = np.mean(all_f1)
    print(f"\n{'='*60}")
    print(f"W23 平均 F1: {100*avg:.1f}% (v_N=TBD%, v_N=TBD%, v_N=TBD%)")


if __name__ == "__main__":
    main()
