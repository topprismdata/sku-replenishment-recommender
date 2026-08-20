"""
AG v9 — Instacart 启发的特征工程
================================
核心新增 (Instacart top 2% 验证过):
1. UP_order_strike: 1/2^(逆向周序号) — 最强特征, 近期购买指数衰减
2. reorder_ratio: (客户,SKU) 购买次数 / 客户总下单周数
3. 购买间隔均值+标准差: 行为一致性
4. 窗口均值差: roll_mean_3 - roll_mean_8 (趋势加速/减速)
5. 品类×客户交互: 客户在各品类的购买比例
6. rolling min/max (不只 mean/std)

保持 v8 的有效特征, 不动 AG 配置 (v7 教训: 过优化反降)
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
# 定位项目根目录 (数据文件所在)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..')) if ('core' in _HERE or 'experiments' in _HERE) else _HERE
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'core'))
warnings.filterwarnings('ignore')

from lgb_v4 import load_data, build_cooccur_matrix, CHANNEL_N_CAP, CHANNEL_MAP
from autogluon.tabular import TabularPredictor


def load_static_features():
    """加载产品表 + 客户表的 static features (原 ag_v8)"""
    prod = pd.read_csv("products.csv", usecols=['ARTICLE_NO','BRAND_NAME_CN','BEVCAT_NAME_CN','PACKAGETYPE_NAME_CN'])
    prod.columns = ['ARTICLE','brand','category','package_type']
    cust = pd.read_csv("customers.csv", usecols=['OUTLET_NO','ABCD_CODE','WITH_COOLER','CHANNEL_CODE'])
    cust.columns = ['OUTLET','abcd','cooler','channel_code']
    return prod, cust


def build_features_v9(weekly, channel, train_weeks, target_week, cooccur, prod):
    ch = weekly[weekly['sub_channel']==channel]
    train = ch[ch['week'].isin(train_weeks)]

    # 候选
    cust_sku = train[['OUTLET','ARTICLE']].drop_duplicates()
    top_skus = train['ARTICLE'].value_counts().head(50).index.tolist()
    all_custs = train['OUTLET'].unique()
    cust_top = pd.DataFrame([(c,s) for c in all_custs for s in top_skus], columns=['OUTLET','ARTICLE'])
    candidates = pd.concat([cust_sku, cust_top]).drop_duplicates()

    # === v8 基础特征 (保留) ===
    pair = train.groupby(['OUTLET','ARTICLE']).agg(
        pair_freq=('freq','sum'), pair_weeks=('week','nunique'), pair_last=('week','max')
    ).reset_index()
    candidates = candidates.merge(pair, on=['OUTLET','ARTICLE'], how='left')
    candidates[['pair_freq','pair_weeks']] = candidates[['pair_freq','pair_weeks']].fillna(0)
    candidates['pair_last'] = candidates['pair_last'].fillna(0)
    candidates['pair_gap'] = target_week - candidates['pair_last']

    for lag_idx, lag_label in [(1,'lag1'),(2,'lag2')]:
        if len(train_weeks) >= lag_idx:
            w = train_weeks[-lag_idx]
            lag = train[train['week']==w].groupby(['OUTLET','ARTICLE']).size().reset_index(name=f'{lag_label}_freq')
            candidates = candidates.merge(lag, on=['OUTLET','ARTICLE'], how='left')
        else:
            candidates[f'{lag_label}_freq'] = 0
        candidates[f'{lag_label}_freq'] = candidates[f'{lag_label}_freq'].fillna(0)

    weights = {w: i+1 for i, w in enumerate(train_weeks)}
    tw_df = train.copy(); tw_df['weight'] = tw_df['week'].map(weights)
    tw_df['wf'] = tw_df['freq'] * tw_df['weight']
    decay = tw_df.groupby(['OUTLET','ARTICLE'])['wf'].sum().reset_index(name='weighted_freq')
    candidates = candidates.merge(decay, on=['OUTLET','ARTICLE'], how='left')
    candidates['weighted_freq'] = candidates['weighted_freq'].fillna(0)

    cust = train.groupby('OUTLET').agg(
        c_weeks=('week','nunique'), c_skus=('ARTICLE','nunique'), c_freq=('freq','sum')
    ).reset_index()
    candidates = candidates.merge(cust, on='OUTLET', how='left')
    cust_n = train.groupby(['OUTLET','week'])['ARTICLE'].nunique().groupby('OUTLET').median().reset_index(name='cust_typical_n')
    candidates = candidates.merge(cust_n, on='OUTLET', how='left')

    sku = train.groupby('ARTICLE').agg(
        s_custs=('OUTLET','nunique'), s_freq=('freq','sum'), s_weeks=('week','nunique')
    ).reset_index()
    candidates = candidates.merge(sku, on='ARTICLE', how='left')
    candidates[['s_custs','s_freq','s_weeks']] = candidates[['s_custs','s_freq','s_weeks']].fillna(0)

    # 共现
    cust_hist_map = train.groupby('OUTLET')['ARTICLE'].apply(set).to_dict()
    cooccur_scores = []
    for _, row in candidates.iterrows():
        c = row['OUTLET']; s = row['ARTICLE']
        hist = cust_hist_map.get(c, set()) - {s}
        cooccur_scores.append(sum(cooccur.get(s, {}).get(x, 0) for x in hist))
    candidates['cooccur_score'] = cooccur_scores

    # rolling (v8)
    pair_weekly = train.groupby(['OUTLET','ARTICLE','week'])['freq'].first().reset_index()
    for window in [3, 5, 8]:
        rolled = pair_weekly.sort_values(['OUTLET','ARTICLE','week']).groupby(['OUTLET','ARTICLE']).rolling(
            window, on='week', min_periods=1)['freq'].agg(['mean','std']).reset_index()
        rolled.columns = ['OUTLET','ARTICLE','week',f'roll_mean_{window}',f'roll_std_{window}']
        last_roll = rolled.sort_values('week').groupby(['OUTLET','ARTICLE']).last().reset_index()
        candidates = candidates.merge(
            last_roll[['OUTLET','ARTICLE',f'roll_mean_{window}',f'roll_std_{window}']],
            on=['OUTLET','ARTICLE'], how='left')
        candidates[f'roll_mean_{window}'] = candidates[f'roll_mean_{window}'].fillna(0)
        candidates[f'roll_std_{window}'] = candidates[f'roll_std_{window}'].fillna(0)

    ewm = pair_weekly.sort_values(['OUTLET','ARTICLE','week']).groupby(['OUTLET','ARTICLE']).apply(
        lambda g: g['freq'].ewm(span=4).mean().iloc[-1], include_groups=False
    ).reset_index(name='ewm_4')
    candidates = candidates.merge(ewm, on=['OUTLET','ARTICLE'], how='left')
    candidates['ewm_4'] = candidates['ewm_4'].fillna(0)

    # === v9 新增: Instacart 启发 ===

    # 1. UP_order_strike: 1/2^(逆向周序号) — Instacart 最强特征
    # 对每个 (客户,SKU), 计算最近购买的逆向权重和
    sorted_weeks = sorted(train_weeks, reverse=True)  # 最近在前
    week_reverse = {w: i for i, w in enumerate(sorted_weeks)}  # 最近=0, 次近=1...
    train_strike = train.copy()
    train_strike['reverse_idx'] = train_strike['week'].map(week_reverse)
    train_strike['strike'] = 1.0 / (2.0 ** train_strike['reverse_idx'])
    strike = train_strike.groupby(['OUTLET','ARTICLE'])['strike'].sum().reset_index(name='up_order_strike')
    candidates = candidates.merge(strike, on=['OUTLET','ARTICLE'], how='left')
    candidates['up_order_strike'] = candidates['up_order_strike'].fillna(0)

    # 2. reorder_ratio: 购买次数 / 客户总下单周数
    candidates['reorder_ratio'] = candidates['pair_weeks'] / candidates['c_weeks'].replace(0, 1)

    # 3. 购买间隔均值+标准差
    pair_gaps_list = train.groupby(['OUTLET','ARTICLE']).apply(
        lambda g: np.diff(sorted(g['week'].unique())), include_groups=False
    ).reset_index(name='gaps')
    pair_gaps_list['gap_mean'] = pair_gaps_list['gaps'].apply(lambda x: np.mean(x) if len(x)>0 else 99)
    pair_gaps_list['gap_std'] = pair_gaps_list['gaps'].apply(lambda x: np.std(x) if len(x)>1 else 0)
    candidates = candidates.merge(pair_gaps_list[['OUTLET','ARTICLE','gap_mean','gap_std']], on=['OUTLET','ARTICLE'], how='left')
    candidates['gap_mean'] = candidates['gap_mean'].fillna(99)
    candidates['gap_std'] = candidates['gap_std'].fillna(0)

    # 4. 窗口均值差 (趋势变化)
    candidates['trend_diff_3_8'] = candidates['roll_mean_3'] - candidates['roll_mean_8']

    # 5. rolling min/max (Instacart 多统计量)
    for window in [5]:
        rolled_minmax = pair_weekly.sort_values(['OUTLET','ARTICLE','week']).groupby(['OUTLET','ARTICLE']).rolling(
            window, on='week', min_periods=1)['freq'].agg(['min','max']).reset_index()
        rolled_minmax.columns = ['OUTLET','ARTICLE','week',f'roll_min_{window}',f'roll_max_{window}']
        last_mm = rolled_minmax.sort_values('week').groupby(['OUTLET','ARTICLE']).last().reset_index()
        candidates = candidates.merge(
            last_mm[['OUTLET','ARTICLE',f'roll_min_{window}',f'roll_max_{window}']],
            on=['OUTLET','ARTICLE'], how='left')
        candidates[f'roll_min_{window}'] = candidates[f'roll_min_{window}'].fillna(0)
        candidates[f'roll_max_{window}'] = candidates[f'roll_max_{window}'].fillna(0)

    # 6. 品类×客户交互
    candidates = candidates.merge(prod[['ARTICLE','category']], on='ARTICLE', how='left')
    cust_cat = train.merge(prod[['ARTICLE','category']], on='ARTICLE', how='left')
    cust_cat_count = cust_cat.groupby(['OUTLET','category']).size().reset_index(name='cust_cat_count')
    cust_total = cust_cat.groupby('OUTLET').size().reset_index(name='cust_total_orders')
    cust_cat_count = cust_cat_count.merge(cust_total, on='OUTLET', how='left')
    cust_cat_count['cat_purchase_ratio'] = cust_cat_count['cust_cat_count'] / cust_cat_count['cust_total_orders']
    candidates = candidates.merge(cust_cat_count[['OUTLET','category','cat_purchase_ratio']], on=['OUTLET','category'], how='left')
    candidates['cat_purchase_ratio'] = candidates['cat_purchase_ratio'].fillna(0)
    candidates['category'] = candidates['category'].fillna('unknown').astype('category')

    # 标签
    valid = ch[ch['week']==target_week]
    vp = valid[['OUTLET','ARTICLE']].drop_duplicates(); vp['label']=1
    candidates = candidates.merge(vp, on=['OUTLET','ARTICLE'], how='left')
    candidates['label'] = candidates['label'].fillna(0).astype(int)

    feat = [
        # v8 基础
        'pair_freq','pair_weeks','pair_gap','lag1_freq','lag2_freq','weighted_freq',
        'c_weeks','c_skus','c_freq','cust_typical_n',
        's_custs','s_freq','s_weeks','cooccur_score',
        'roll_mean_3','roll_std_3','roll_mean_5','roll_std_5','roll_mean_8','roll_std_8','ewm_4',
        # v9 Instacart 新增
        'up_order_strike','reorder_ratio','gap_mean','gap_std',
        'trend_diff_3_8','roll_min_5','roll_max_5','cat_purchase_ratio',
    ]
    cat_feat = ['category']
    return candidates, feat, cat_feat


def main():
    print("="*60)
    print("AG v9 (Instacart 启发: UP_order_strike + reorder_ratio + 间隔)")
    print("="*60)
    weekly = load_data()
    prod, _ = load_static_features()
    print(f"周聚合: {len(weekly):,}")

    tw, tws = 23, list(range(19,23))
    vs = weekly[weekly['week']==tw].groupby('OUTLET')['ARTICLE'].apply(set).to_dict()

    all_f1 = []
    for ch in CHANNEL_MAP:
        cooccur = build_cooccur_matrix(weekly, ch, tws)
        candidates, feat, cat_feat = build_features_v9(weekly, ch, tws, tw, cooccur, prod)
        train_df = candidates[feat + cat_feat + ['label']].copy()
        print(f"\n  {ch}: {len(train_df):,} 样本, {len(feat)+len(cat_feat)} 特征")

        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v9_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={
                'GBM': [{'num_leaves':63,'min_child_samples':100},{'num_leaves':31,'min_child_samples':50}],
                'CAT': {}, 'XGB': {}, 'RF': [{'criterion':'gini'},{'criterion':'entropy'}], 'XT': {},
            },
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
                fi = model.feature_importance(train_df, silent=True).head(20)
                print(f"    特征重要度 top15:")
                for _, row in fi.head(15).iterrows():
                    print(f"      {row['index']:<25} {row['importance']:.4f}")
            except: pass

    avg = np.mean(all_f1)
    print(f"\n{'='*60}")
    print(f"W23 平均 F1: {100*avg:.1f}% (v8=65.8%, v6=64.1%)")


if __name__ == "__main__":
    main()
