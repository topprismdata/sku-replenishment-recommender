"""
生成最终submission_file (修正版)
==========================
两阶段策略:
1. 训练: train_weeks=W16-W28, label 来自 W29 (有真实标签)
2. 预测: 用 W17-W29 的特征构建候选, target_week=W30 (label 忽略), 用训练好的模型预测

这样模型用了最近 13 周数据训练, 预测下一周 (W30)
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
    'channel_a': (TBD_BASE, TBD_MULT),
    'channel_b': (TBD_BASE, TBD_MULT),
    'channel_c': (TBD_BASE, TBD_MULT),
    'channel_d': (TBD_BASE, TBD_MULT),
    'channel_e': (TBD_BASE, TBD_MULT),  # reuse channel_a params
}


def build_and_train(weekly, ch, train_weeks, label_week, prod_feat, cust_feat):
    """训练阶段: build features from train_weeks, label from label_week"""
    cooccur = build_cooccur_matrix(weekly, ch, train_weeks)
    candidates, feat, cat_feat = build_features_v20(weekly, ch, train_weeks, label_week, cooccur, prod_feat, cust_feat)

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

    train_df = candidates[feat + cat_feat + ['label']].copy()
    model = TabularPredictor(
        label='label', problem_type='binary', eval_metric='roc_auc',
        path=f'ag_final_{ch}', verbosity=0,
    ).fit(
        train_data=train_df,
        hyperparameters={'GBM': [{'num_leaves': 63, 'min_child_samples': 100},
                                  {'num_leaves': 31, 'min_child_samples': 50}],
                         'CAT': {}, 'XGB': {},
                         'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}], 'XT': {}},
        time_limit=300, excluded_model_types=['NN_TORCH', 'FASTAI'],
    )
    return model, feat, cat_feat


def build_predict_candidates(weekly, ch, feat_weeks, target_week, prod_feat, cust_feat):
    """预测阶段: build features from feat_weeks, target_week for gap/lag, label ignored"""
    cooccur = build_cooccur_matrix(weekly, ch, feat_weeks)
    candidates, feat, cat_feat = build_features_v20(weekly, ch, feat_weeks, target_week, cooccur, prod_feat, cust_feat)

    # 容量特征 (同训练)
    prod_stats = weekly[weekly['sub_channel'] == ch].groupby('ARTICLE').agg(
        size_ml=('size_ml', 'first'), sub_unit_num=('sub_unit_num', 'first'), total_ml=('total_ml', 'first')
    ).reset_index()
    candidates = candidates.merge(prod_stats, on='ARTICLE', how='left')
    candidates[['size_ml', 'sub_unit_num', 'total_ml']] = candidates[['size_ml', 'sub_unit_num', 'total_ml']].fillna(0)
    candidates = candidates.merge(weekly.groupby('ARTICLE')['calorie_cat'].first().reset_index(), on='ARTICLE', how='left')
    candidates['calorie_cat'] = candidates['calorie_cat'].fillna('-').astype('category')
    feat = feat + ['size_ml', 'sub_unit_num', 'total_ml']
    cat_feat = cat_feat + ['calorie_cat']

    return candidates, feat, cat_feat


def main():
    print("=" * 60)
    print("生成最终submission_file")
    print("=" * 60)

    sub = pd.read_csv("./pre_may_recommendations/pre_may_recommendations.csv")
    sub_custs = set(sub['customer_id'].unique())
    print(f"提交客户: {len(sub_custs)}")

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

    max_week = int(weekly['week'].max())
    # W30 不完整 (12729 行), 用 W29 为最新完整周
    # 训练: W16-W28 → label W29
    # 预测: features W17-W29 → target W30
    label_week = 29
    train_weeks = list(range(label_week - 13, label_week))  # W16-W28
    feat_weeks = list(range(label_week - 12, label_week + 1))  # W17-W29
    pred_week = label_week + 1  # W30

    print(f"训练: features W{train_weeks[0]}-W{train_weeks[-1]}, label W{label_week}")
    print(f"预测: features W{feat_weeks[0]}-W{feat_weeks[-1]}, target W{pred_week}")

    all_recs = []
    trained_models = {}

    for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
        print(f"\n--- {ch} ---")
        t0 = time.time()
        model, feat, cat_feat = build_and_train(weekly, ch, train_weeks, label_week, prod_feat, cust_feat)
        print(f"  训练: {time.time() - t0:.0f}s, best={model.model_best}")
        trained_models[ch] = model

        # 预测候选
        candidates, _, _ = build_predict_candidates(weekly, ch, feat_weeks, pred_week, prod_feat, cust_feat)
        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba[:, 1]

        # 按 N 公式推荐
        base, mult = BEST_N[ch]
        cap = CHANNEL_N_CAP.get(ch, 25)
        ch_custs = sub[sub['channel'] == ch]['customer_id'].unique()
        n_rec = 0
        for c in ch_custs:
            cd = candidates[candidates['OUTLET'] == c]
            if len(cd) == 0:
                continue
            typical = cd['cust_typical_n'].iloc[0]
            N = int(min(max(base + mult * typical, 5), cap))
            top = cd.nlargest(N, 'prob')[['OUTLET', 'ARTICLE', 'prob']].copy()
            top['channel'] = ch
            all_recs.append(top)
            n_rec += len(top)
        print(f"  推荐: {len(ch_custs)} 客户, {n_rec} 条")

    # channel_e (convenience): use channel_a model
    print(f"\n--- channel_e (using channel_a model) ---")
    cv_custs = sub[sub['channel'] == 'channel_e']['customer_id'].unique()
    candidates_cv, _, _ = build_predict_candidates(weekly, 'channel_a', feat_weeks, pred_week, prod_feat, cust_feat)
    candidates_cv = candidates_cv[candidates_cv['OUTLET'].isin(cv_custs)]
    if len(candidates_cv) > 0:
        model_cv = trained_models['channel_a']
        all_feat_cv = feat + cat_feat  # channel_a feature list
        proba_cv = model_cv.predict_proba(candidates_cv[all_feat_cv])
        candidates_cv['prob'] = proba_cv.iloc[:, 1].values if hasattr(proba_cv, 'iloc') else proba_cv[:, 1]
        base, mult = BEST_N['channel_e']
        cap = CHANNEL_N_CAP.get('channel_a', 25)
        n_rec = 0
        for c in cv_custs:
            cd = candidates_cv[candidates_cv['OUTLET'] == c]
            if len(cd) == 0:
                continue
            typical = cd['cust_typical_n'].iloc[0]
            N = int(min(max(base + mult * typical, 5), cap))
            top = cd.nlargest(N, 'prob')[['OUTLET', 'ARTICLE', 'prob']].copy()
            top['channel'] = 'channel_e'
            all_recs.append(top)
            n_rec += len(top)
        print(f"  推荐: {len(cv_custs)} 客户, {n_rec} 条")
    else:
        print(f"  channel_e customers have no orders, skipping")

    # 合并
    print(f"\n{'=' * 60}")
    print("合并推荐结果")
    print(f"{'=' * 60}")
    final = pd.concat(all_recs, ignore_index=True)
    print(f"总推荐: {len(final)} 条, {final['OUTLET'].nunique()} 客户")

    # 生成提交格式
    cust_info = sub[['customer_id', 'customer_name', 'channel', 'route_id']].drop_duplicates('customer_id').set_index('customer_id')
    sku_info = prod_raw[['ARTICLE_NO', 'PRODUCT_NAME_CN']].drop_duplicates('ARTICLE_NO').set_index('ARTICLE_NO')

    final = final.rename(columns={'OUTLET': 'customer_id', 'ARTICLE': 'sku_id'})
    final = final.merge(cust_info, on='customer_id', how='left')
    final = final.merge(sku_info.rename(columns={'PRODUCT_NAME_CN': 'sku_name'}), left_on='sku_id', right_index=True, how='left')
    final['recommended_cases'] = 1.0
    final['algo_group'] = 'A'
    final['模型'] = 'AG_v28_13w_optunaN'

    out_cols = ['customer_id', 'customer_name', 'channel', 'route_id', 'sku_id', 'sku_name', 'recommended_cases', 'algo_group', '模型']
    final_out = final[out_cols].copy()
    final_out.to_csv("submission_final.csv", index=False, encoding='utf-8-sig')
    print(f"\n已保存: submission_final.csv ({len(final_out)} 行)")

    print(f"\n渠道分布:")
    for ch_name, grp in final_out.groupby('channel'):
        n_cust = grp['customer_id'].nunique()
        print(f"  {ch_name}: {len(grp)} 行, {n_cust} 客户, 平均 {len(grp) / n_cust:.1f} SKU/客户")


if __name__ == "__main__":
    main()
