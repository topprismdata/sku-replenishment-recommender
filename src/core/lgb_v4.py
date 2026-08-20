"""
Per-Channel LGB v4 — 加共现 + route_collab特征
=========================================
v3->v4 优化:
1. cooccurrence_features: 该客户历史买的SKU里, 跟候选SKU的cooccurrence_strength
2. route_collab: 同线路其他客户最近买的SKU (需要 PS_ROUTE)
3. SKU 动量: 该SKU在渠道内最近几周的流行度趋势
"""
# ============================================================
# Week configuration - adjust to your data
# ============================================================
TRAIN_START_WEEK = 1   # first training week (inclusive)

# N formula tuning - adjust to your data
# These are business-tuned values that must be calibrated for your own
# per-channel N formula. Defaults shown are placeholders.
TBD_N_OFFSET = 5          # added to typical_n (e.g., N = typical_n + TBD_N_OFFSET)
TBD_N_MIN = 10            # minimum N (recommended per row)
TBD_N_BONUS = 10          # bonus N above CHANNEL_N_CAP for search range
TBD_N_DEFAULT = 25        # default N if CHANNEL_N_CAP entry is missing
TBD_CAP = 25              # default per-channel N upper bound
TARGET_WEEK = 1        # first target / prediction week (inclusive)
# Extend with: TARGET_WEEK_2 = TARGET_WEEK + 1, etc. for walk-forward
# ============================================================

import pandas as pd, numpy as np, lightgbm as lgb, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

CHANNEL_MAP = {'channel_a':['sub_code_1','sub_code_2'],'channel_b':['sub_code_8','sub_code_9','sub_code_10','sub_code_11','sub_code_12'],'channel_c':['sub_code_13','sub_code_14','sub_code_15','sub_code_16'],'channel_d':['sub_code_3','sub_code_4','sub_code_5','sub_code_6','sub_code_7']}
CODE_TO_CH = {c:ch for ch,cs in CHANNEL_MAP.items() for c in cs}
CHANNEL_N_CAP = {'channel_a': TBD_CAP, 'channel_b': TBD_CAP, 'channel_c': TBD_CAP, 'channel_d': TBD_CAP}


def load_data():
    orders = pd.read_csv("./orders_2026.csv",
        usecols=['OUTLET','ARTICLE','ORDER_DATE','SETTLE_CASE','CHANNEL','PS_ROUTE'], low_memory=False)
    orders['ORDER_DATE'] = pd.to_datetime(orders['ORDER_DATE'], errors='coerce')
    orders = orders[orders['SETTLE_CASE']>0].copy()
    orders['week'] = orders['ORDER_DATE'].dt.isocalendar().week.astype(int)
    orders['year'] = orders['ORDER_DATE'].dt.year
    orders = orders[orders['year']==2026]
    orders['sub_channel'] = orders['CHANNEL'].map(CODE_TO_CH)
    sub = pd.read_csv("./pre_may_recommendations/pre_may_recommendations.csv", usecols=['customer_id'])
    sub_custs = set(sub['customer_id'].unique())
    orders = orders[orders['sub_channel'].notna() & orders['OUTLET'].isin(sub_custs)]
    weekly = orders.groupby(['OUTLET','ARTICLE','week','sub_channel','PS_ROUTE']).size().reset_index(name='freq')
    return weekly


def build_cooccur_matrix(weekly, channel, train_weeks):
    """构建渠道级 SKU cooccurrence_matrix (客户×周 级别)"""
    ch = weekly[(weekly['sub_channel']==channel) & (weekly['week'].isin(train_weeks))]
    # 按 客户×周 聚合 SKU 集合
    baskets = ch.groupby(['OUTLET','week'])['ARTICLE'].apply(set).reset_index()
    # 共现计数
    cooccur = defaultdict(lambda: defaultdict(int))
    for _, row in baskets.iterrows():
        skus = sorted(row['ARTICLE'])
        for i in range(len(skus)):
            for j in range(i+1, len(skus)):
                cooccur[skus[i]][skus[j]] += 1
                cooccur[skus[j]][skus[i]] += 1
    return cooccur


def build_features_v4(weekly, channel, train_weeks, target_week, cooccur):
    ch = weekly[weekly['sub_channel']==channel]
    train = ch[ch['week'].isin(train_weeks)]

    # 候选
    cust_sku = train[['OUTLET','ARTICLE']].drop_duplicates()
    top_skus = train['ARTICLE'].value_counts().head(50).index.tolist()
    all_custs = train['OUTLET'].unique()
    cust_top = pd.DataFrame([(c,s) for c in all_custs for s in top_skus], columns=['OUTLET','ARTICLE'])
    candidates = pd.concat([cust_sku, cust_top]).drop_duplicates()

    # v3 基础特征
    pair = train.groupby(['OUTLET','ARTICLE']).agg(
        pair_freq=('freq','sum'), pair_weeks=('week','nunique'), pair_last=('week','max')
    ).reset_index()
    candidates = candidates.merge(pair, on=['OUTLET','ARTICLE'], how='left')
    candidates[['pair_freq','pair_weeks']] = candidates[['pair_freq','pair_weeks']].fillna(0)
    candidates['pair_last'] = candidates['pair_last'].fillna(0)
    candidates['pair_gap'] = target_week - candidates['pair_last']

    # lag 特征
    for lag_idx, lag_label in [(1,'lag1'),(2,'lag2')]:
        if len(train_weeks) >= lag_idx:
            w = train_weeks[-lag_idx]
            lag = train[train['week']==w].groupby(['OUTLET','ARTICLE']).size().reset_index(name=f'{lag_label}_freq')
            candidates = candidates.merge(lag, on=['OUTLET','ARTICLE'], how='left')
        else:
            candidates[f'{lag_label}_freq'] = 0
        candidates[f'{lag_label}_freq'] = candidates[f'{lag_label}_freq'].fillna(0)

    # 递减加权
    weights = {w: i+1 for i, w in enumerate(train_weeks)}
    tw = train.copy(); tw['weight'] = tw['week'].map(weights)
    tw['wf'] = tw['freq'] * tw['weight']
    decay = tw.groupby(['OUTLET','ARTICLE'])['wf'].sum().reset_index(name='weighted_freq')
    candidates = candidates.merge(decay, on=['OUTLET','ARTICLE'], how='left')
    candidates['weighted_freq'] = candidates['weighted_freq'].fillna(0)

    # 客户级
    cust = train.groupby('OUTLET').agg(
        c_weeks=('week','nunique'), c_skus=('ARTICLE','nunique'), c_freq=('freq','sum')
    ).reset_index()
    candidates = candidates.merge(cust, on='OUTLET', how='left')
    cust_n = train.groupby(['OUTLET','week'])['ARTICLE'].nunique().groupby('OUTLET').median().reset_index(name='cust_typical_n')
    candidates = candidates.merge(cust_n, on='OUTLET', how='left')

    # SKU 渠道级
    sku = train.groupby('ARTICLE').agg(
        s_custs=('OUTLET','nunique'), s_freq=('freq','sum'), s_weeks=('week','nunique')
    ).reset_index()
    candidates = candidates.merge(sku, on='ARTICLE', how='left')
    candidates[['s_custs','s_freq','s_weeks']] = candidates[['s_custs','s_freq','s_weeks']].fillna(0)

    # v4 新增: cooccurrence_features
    # 对每个候选SKU, 算客户历史买的SKU跟它的cooccurrence_strength
    cust_hist_map = train.groupby('OUTLET')['ARTICLE'].apply(set).to_dict()
    cooccur_scores = []
    for _, row in candidates.iterrows():
        cust = row['OUTLET']; sku = row['ARTICLE']
        hist_skus = cust_hist_map.get(cust, set())
        if sku in hist_skus:
            hist_skus = hist_skus - {sku}  # 排除自己
        score = sum(cooccur.get(sku, {}).get(s, 0) for s in hist_skus)
        cooccur_scores.append(score)
    candidates['cooccur_score'] = cooccur_scores

    # v4 新增: route_collab (同 PS_ROUTE 其他客户买的)
    route_data = train.groupby(['PS_ROUTE','ARTICLE']).agg(
        route_custs=('OUTLET','nunique'), route_freq=('freq','sum')
    ).reset_index()
    cust_route = train.groupby('OUTLET')['PS_ROUTE'].first().reset_index()
    candidates = candidates.merge(cust_route, on='OUTLET', how='left')
    candidates = candidates.merge(route_data, on=['PS_ROUTE','ARTICLE'], how='left')
    candidates[['route_custs','route_freq']] = candidates[['route_custs','route_freq']].fillna(0)
    candidates = candidates.drop(columns=['PS_ROUTE'])

    # v4 新增: SKU 最近一周渠道热度 (动量)
    if len(train_weeks) > 0:
        last_w = train_weeks[-1]
        last_week = train[train['week']==last_w].groupby('ARTICLE').agg(
            last_w_custs=('OUTLET','nunique'), last_w_freq=('freq','sum')
        ).reset_index()
        candidates = candidates.merge(last_week, on='ARTICLE', how='left')
        candidates[['last_w_custs','last_w_freq']] = candidates[['last_w_custs','last_w_freq']].fillna(0)
    else:
        candidates['last_w_custs'] = 0; candidates['last_w_freq'] = 0

    # 标签
    valid = ch[ch['week']==target_week]
    vp = valid[['OUTLET','ARTICLE']].drop_duplicates(); vp['label']=1
    candidates = candidates.merge(vp, on=['OUTLET','ARTICLE'], how='left')
    candidates['label'] = candidates['label'].fillna(0).astype(int)

    feat = ['pair_freq','pair_weeks','pair_gap','lag1_freq','lag2_freq','weighted_freq',
            'c_weeks','c_skus','c_freq','cust_typical_n',
            's_custs','s_freq','s_weeks',
            'cooccur_score','route_custs','route_freq','last_w_custs','last_w_freq']
    return candidates, feat


def train_predict_v4(weekly, channel, train_weeks, target_week):
    cooccur = build_cooccur_matrix(weekly, channel, train_weeks)
    candidates, feat = build_features_v4(weekly, channel, train_weeks, target_week, cooccur)
    X, y = candidates[feat], candidates['label']
    params = dict(num_leaves=63 if channel in ['channel_a','channel_b'] else 31,
                  min_child_samples=100 if channel in ['channel_a','channel_b'] else 50)
    model = lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.05,
        feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=42,
        n_estimators=200,**params)
    model.fit(X, y)
    candidates['prob'] = model.predict_proba(X)[:,1]
    return candidates[['OUTLET','ARTICLE','prob','cust_typical_n']], feat, model


def evaluate_dynamic_n(preds, valid_skus, channel):
    cap = CHANNEL_N_CAP.get(channel, TBD_N_DEFAULT)
    tr, ta, th = 0, 0, 0
    for c in set(preds['OUTLET']) & set(valid_skus):
        cd = preds[preds['OUTLET']==c]
        typical = cd['cust_typical_n'].iloc[0] if len(cd)>0 else TBD_N_DEFAULT
        N = int(min(max(typical + TBD_N_OFFSET, TBD_N_MIN), cap))
        rec = set(cd.nlargest(N,'prob')['ARTICLE'])
        act = valid_skus[c]
        tr += len(rec); ta += len(act); th += len(rec & act)
    p = th/max(tr,1); r = th/max(ta,1); f1 = 2*p*r/max(p+r,1e-8)
    return p, r, f1


def main():
    print("="*60)
    print("Per-Channel LGB v4 (共现 + route_collab + SKU动量)")
    print("="*60)
    weekly = load_data()
    print(f"周聚合: {len(weekly):,} 行")

    all_f1 = []
    first_iter = True
    for tw in range(TRAIN_START_WEEK, TARGET_WEEK):
        tws = list(range(tw-4, tw))
        va = weekly[weekly['week']==tw]
        vs = va.groupby('OUTLET')['ARTICLE'].apply(set).to_dict()
        ch_results = {}
        for ch in CHANNEL_MAP:
            preds, feat, model = train_predict_v4(weekly, ch, tws, tw)
            p, r, f1 = evaluate_dynamic_n(preds, vs, ch)
            ch_results[ch] = (p, r, f1)
            # 第一次打印特征重要度
            if first_iter and ch == 'channel_a':
                imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)
                print(f"\n特征重要度 ({ch}):")
                for k, v in imp.items():
                    print(f"  {k:<20} {v}")
        first_iter = False
        avg_f1 = np.mean([v[2] for v in ch_results.values()])
        all_f1.append(avg_f1)
        chs = " ".join(f"{c[:2]}={100*v[2]:.0f}%" for c,v in ch_results.items())
        print(f"W{tw}: F1={100*avg_f1:.1f}% | {chs}")

    print(f"\nv4 滚动平均 F1: {100*np.mean(all_f1):.1f}% (v_N=TBD%, v_N=TBD%, 基线=41.2%)")


if __name__ == "__main__":
    main()
