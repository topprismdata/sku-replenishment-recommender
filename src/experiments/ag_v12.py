"""
AG v12 — 加线路(业务员)特征
============================
关键领域知识: 线路 = 业务员, 同线路客户采购有同步性
EDA 手册: "线路是最强预测维度之一"

新增特征:
1. route_sku_freq: 该SKU在同线路其他客户中的购买频次
2. route_sku_rank: 该SKU在同线路的购买排名
3. route_total_skus: 线路级SKU多样性
4. route_top_match: 候选SKU是否在该线路top-10热销里
5. route_cust_count: 同线路客户数 (规模)

基于 v9 (最强 67.7%) + 线路特征, 不加 v11 的噪声特征
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
from ag_v8 import load_static_features
from ag_v9 import build_features_v9
from autogluon.tabular import TabularPredictor


def load_weekly_with_route():
    """加载带route_id的周数据"""
    orders = pd.read_csv("orders_2026.csv",
        usecols=['OUTLET','ARTICLE','ORDER_DATE','SETTLE_CASE','CHANNEL','PS_ROUTE'], low_memory=False)
    orders['ORDER_DATE'] = pd.to_datetime(orders['ORDER_DATE'], errors='coerce')
    orders = orders[orders['SETTLE_CASE']>0].copy()
    orders['week'] = orders['ORDER_DATE'].dt.isocalendar().week.astype(int)
    orders = orders[orders['ORDER_DATE'].dt.year==2026]
    CODE_TO_CH = {c:ch for ch,cs in CHANNEL_MAP.items() for c in cs}
    orders['sub_channel'] = orders['CHANNEL'].map(CODE_TO_CH)
    sub = pd.read_csv("pre_may_recommendations/pre_may_recommendations.csv", usecols=['customer_id'])
    orders = orders[orders['sub_channel'].notna() & orders['OUTLET'].isin(set(sub['customer_id']))]
    weekly = orders.groupby(['OUTLET','ARTICLE','week','sub_channel','PS_ROUTE']).size().reset_index(name='freq')
    return weekly


def build_features_v12(weekly, channel, train_weeks, target_week, cooccur, prod):
    """v9 基础 + 线路特征"""
    # 先用 v9 的特征构建
    candidates, feat, cat_feat = build_features_v9(weekly, channel, train_weeks, target_week, cooccur, prod)

    ch = weekly[weekly['sub_channel']==channel]
    train = ch[ch['week'].isin(train_weeks)]

    # 客户→线路映射
    cust_route = train.groupby('OUTLET')['PS_ROUTE'].first().reset_index()
    candidates = candidates.merge(cust_route, on='OUTLET', how='left')

    # === 线路级 SKU 特征 ===
    # 1. route_sku_freq: 该SKU在同线路的购买频次 (排除当前客户)
    route_sku = train.groupby(['PS_ROUTE','ARTICLE']).agg(
        route_freq=('freq','sum'),
        route_custs=('OUTLET','nunique'),
        route_weeks=('week','nunique'),
    ).reset_index()
    candidates = candidates.merge(route_sku, on=['PS_ROUTE','ARTICLE'], how='left')
    candidates[['route_freq','route_custs','route_weeks']] = candidates[['route_freq','route_custs','route_weeks']].fillna(0)

    # 2. route_sku_rank: 该SKU在同线路的排名 (按频次)
    route_sku['route_rank'] = route_sku.groupby('PS_ROUTE')['route_freq'].rank(ascending=False, method='min')
    candidates = candidates.merge(route_sku[['PS_ROUTE','ARTICLE','route_rank']], on=['PS_ROUTE','ARTICLE'], how='left')
    candidates['route_rank'] = candidates['route_rank'].fillna(999)

    # 3. 线路级特征
    route_stats = train.groupby('PS_ROUTE').agg(
        route_total_custs=('OUTLET','nunique'),
        route_total_skus=('ARTICLE','nunique'),
        route_total_freq=('freq','sum'),
    ).reset_index()
    candidates = candidates.merge(route_stats, on='PS_ROUTE', how='left')
    candidates[['route_total_custs','route_total_skus','route_total_freq']] = candidates[['route_total_custs','route_total_skus','route_total_freq']].fillna(0)

    # 4. route_top_match: 候选SKU是否在该线路top-10热销里
    route_top10 = train.groupby('PS_ROUTE')['ARTICLE'].apply(
        lambda x: set(x.value_counts().head(10).index)
    ).to_dict()
    candidates['route_top_match'] = candidates.apply(
        lambda r: int(r['ARTICLE'] in route_top10.get(r['PS_ROUTE'], set())), axis=1
    )

    # 5. 同线路其他客户买这个SKU的比例 (排除自己)
    candidates['route_penetration'] = candidates['route_custs'] / candidates['route_total_custs'].replace(0, 1)

    # 清理 PS_ROUTE (不作为特征, 只用于分组)
    candidates = candidates.drop(columns=['PS_ROUTE'])

    # 追加新特征到 feat 列表
    new_feat = ['route_freq','route_custs','route_weeks','route_rank',
                'route_total_custs','route_total_skus','route_total_freq',
                'route_top_match','route_penetration']
    feat = feat + new_feat

    return candidates, feat, cat_feat


def main():
    print("="*60)
    print("AG v12 (v9 + 线路/业务员特征)")
    print("="*60)
    weekly = load_weekly_with_route()
    prod, _ = load_static_features()
    print(f"周聚合: {len(weekly):,} (含线路)")

    tw, tws = 23, list(range(19,23))
    vs = weekly[weekly['week']==tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    all_f1 = []
    for ch in CHANNEL_MAP:
        cooccur = build_cooccur_matrix(weekly, ch, tws)
        candidates, feat, cat_feat = build_features_v12(weekly, ch, tws, tw, cooccur, prod)
        train_df = candidates[feat + cat_feat + ['label']].copy()
        print(f"\n  {ch}: {len(train_df):,} 样本, {len(feat)+len(cat_feat)} 特征")

        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v12_{ch}_w{tw}', verbosity=0,
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

        # Print feature importance on first channel
        if ch == 'channel_a':
            try:
                fi = model.feature_importance(train_df, silent=True)
                print(f"    特征重要度 top15:")
                for _, row in fi.head(15).iterrows():
                    print(f"      {row['index']:<25} {row['importance']:.4f}")
            except: pass

    avg = np.mean(all_f1)
    print(f"\n{'='*60}")
    print(f"W23 平均 F1: {100*avg:.1f}% (v9=67.7%, v6=64.1%)")


if __name__ == "__main__":
    main()
