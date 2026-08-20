"""
AG v27 — 13周训练 + Optuna 调优 (N + AG 超参)
==============================================
在 13 周训练基础上, 用 Optuna 自动调优:
1. N 公式参数 (base + mult)
2. AG 超参 (num_leaves, learning_rate, min_child_samples)
3. 目标: 最大化 F1

13周训练 (W10-W22) 是业务经验, Optuna 找最优参数
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
import optuna


def evaluate_n_params(candidates, valid_skus, channel, base, mult):
    """用给定的 N 公式评估 F1"""
    tr, ta, th = 0, 0, 0
    for c in set(candidates['OUTLET']) & set(valid_skus):
        cd = candidates[candidates['OUTLET']==c]
        typical = cd['cust_typical_n'].iloc[0]
        N = int(min(max(base + mult * typical, 10), CHANNEL_N_CAP[channel] * 1.5))
        rec = set(cd.nlargest(N, 'prob')['ARTICLE'])
        act = valid_skus[c]
        tr += len(rec); ta += len(act); th += len(rec & act)
    p = th/max(tr,1); r = th/max(ta,1); f1 = 2*p*r/max(p+r,1e-8)
    return f1


def main():
    print("="*60)
    print("AG v27 (13周训练 + Optuna 调优 N + AG 超参)")
    print("="*60)

    weekly = load_weekly_with_revenue()
    prod_feat, cust_feat = load_all_static()
    prod_raw = pd.read_csv("products.csv", low_memory=False)
    prod_raw['size_ml'] = pd.to_numeric(prod_raw['PACKAGESIZE_NAME_CN'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    prod_raw['sub_unit_num'] = pd.to_numeric(prod_raw['SUB_UNIT'], errors='coerce').fillna(1)
    prod_raw['total_ml'] = prod_raw['size_ml'] * prod_raw['sub_unit_num']
    weekly = weekly.merge(prod_raw[['ARTICLE_NO','size_ml','sub_unit_num','total_ml','CALORIE_INDICATOR_NAME_CN']].rename(columns={'ARTICLE_NO':'ARTICLE','CALORIE_INDICATOR_NAME_CN':'calorie_cat'}), on='ARTICLE', how='left')

    tw, tws = 23, list(range(10,23))  # 13周训练
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

        # Optuna 调 AG 超参
        print(f"\n  {ch}: Optuna 调 AG 超参")
        def objective_ag(trial):
            num_leaves = trial.suggest_int('num_leaves', 31, 127)
            lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            min_child = trial.suggest_int('min_child_samples', 20, 200)
            f1 = 0
            try:
                model = TabularPredictor(
                    label='label', problem_type='binary', eval_metric='roc_auc',
                    path=f'ag_v27_trial_{ch}_w{tw}', verbosity=0,
                ).fit(
                    train_data=train_df,
                    hyperparameters={'GBM':[{'num_leaves':num_leaves,'min_child_samples':min_child,'learning_rate':lr}],
                                     'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
                    time_limit=180, excluded_model_types=['NN_TORCH','FASTAI'],
                )
                all_feat = feat + cat_feat
                proba = model.predict_proba(candidates[all_feat])
                candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]
                f1 = evaluate_n_params(candidates, vs, ch, 5, 1.0)
            except Exception as e:
                f1 = 0
            return f1

        study_ag = optuna.create_study(direction='maximize')
        study_ag.optimize(objective_ag, n_trials=10, timeout=300)
        best_ag = study_ag.best_params
        best_f1_ag = study_ag.best_value
        print(f"    最优 AG 超参: leaves={best_ag['num_leaves']}, lr={best_ag['learning_rate']:.3f}, child={best_ag['min_child_samples']}")
        print(f"    最优 F1 (AG): {100*best_f1_ag:.0f}%")

        # 用最优 AG 超参训练最终模型
        t0 = time.time()
        model = TabularPredictor(
            label='label', problem_type='binary', eval_metric='roc_auc',
            path=f'ag_v27_final_{ch}_w{tw}', verbosity=0,
        ).fit(
            train_data=train_df,
            hyperparameters={'GBM':[{'num_leaves':best_ag['num_leaves'],'min_child_samples':best_ag['min_child_samples'],'learning_rate':best_ag['learning_rate']}],
                             'CAT':{}, 'XGB':{}, 'RF':[{'criterion':'gini'},{'criterion':'entropy'}], 'XT':{}},
            time_limit=300, excluded_model_types=['NN_TORCH','FASTAI'],
        )
        all_feat = feat + cat_feat
        proba = model.predict_proba(candidates[all_feat])
        candidates['prob'] = proba.iloc[:,1].values if hasattr(proba,'iloc') else proba[:,1]

        # Optuna 调 N
        print(f"\n  {ch}: Optuna 调 N")
        def objective_n(trial):
            base = trial.suggest_float('base', 0, 10)
            mult = trial.suggest_float('mult', 0.5, 2.0)
            return evaluate_n_params(candidates, vs, ch, base, mult)
        study_n = optuna.create_study(direction='maximize')
        study_n.optimize(objective_n, n_trials=20, timeout=60)
        best_n = study_n.best_params
        best_f1_n = study_n.best_value
        print(f"    最优 N: base={best_n['base']:.1f}, mult={best_n['mult']:.2f}")
        print(f"    最优 F1 (N): {100*best_f1_n:.0f}%")

        # 对比
        f1_typical = evaluate_n_params(candidates, vs, ch, 5, 1.0)
        print(f"    对比 typical+5: F1={100*f1_typical:.0f}%")


if __name__ == "__main__":
    main()
