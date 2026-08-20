"""
补全submission_file — 为fill_missing_outlets
=============================
策略:
1. 在订单数据里但训练窗口外的客户: 用该渠道 top 热门 SKU
2. 完全不在订单数据的客户 (冷启动): 用该渠道 top 热门 SKU
3. channel_e (convenience): use channel_a top hot SKUs
"""
import sys, os, warnings
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

from lgb_v4 import load_data, CHANNEL_MAP, CODE_TO_CH

# 各渠道推荐数 (用 v28 N 公式的中位数估计)
CHANNEL_N_FALLBACK = {
    'channel_a': 18,
    'channel_b': 24,
    'channel_c': 15,
    'channel_d': 13,
    'channel_e': 18,
}


def main():
    print("=" * 60)
    print("补全submission_file (fill_missing_outlets)")
    print("=" * 60)

    sub = pd.read_csv("./pre_may_recommendations/pre_may_recommendations.csv")
    final = pd.read_csv("submission_final.csv")
    weekly = load_data()

    covered = set(final['customer_id'].unique())
    all_custs = set(sub['customer_id'].unique())
    missing = all_custs - covered
    print(f"已覆盖: {len(covered)}, 缺失: {len(missing)}")

    # 各渠道 top 热门 SKU (按购买客户数)
    prod_raw = pd.read_csv("products.csv", low_memory=False)
    sku_info = prod_raw[['ARTICLE_NO', 'PRODUCT_NAME_CN']].drop_duplicates('ARTICLE_NO').set_index('ARTICLE_NO')
    cust_info = sub[['customer_id', 'customer_name', 'channel', 'route_id']].drop_duplicates('customer_id').set_index('customer_id')

    top_skus_by_ch = {}
    for ch in ['channel_a', 'channel_b', 'channel_c', 'channel_d']:
        ch_data = weekly[weekly['sub_channel'] == ch]
        top = ch_data.groupby('ARTICLE')['OUTLET'].nunique().sort_values(ascending=False).head(30)
        top_skus_by_ch[ch] = top.index.tolist()
        print(f"  {ch} top5 SKU: {top.head(5).index.tolist()}")

    # 为每个缺失客户推荐热门SKU
    fallback_recs = []
    miss_df = sub[sub['customer_id'].isin(missing)].drop_duplicates('customer_id')

    for _, row in miss_df.iterrows():
        cust = row['customer_id']
        ch = row['channel']
        # 决定用哪个渠道的热门SKU
        sku_source = ch if ch in top_skus_by_ch else 'channel_a'
        n = CHANNEL_N_FALLBACK.get(ch, 15)
        skus = top_skus_by_ch.get(sku_source, top_skus_by_ch['channel_a'])[:n]

        for sku in skus:
            fallback_recs.append({
                'customer_id': cust,
                'sku_id': sku,
                'channel': ch,
            })

    fb = pd.DataFrame(fallback_recs)
    print(f"\nfallback_recommendations: {len(fb)} 条, {fb['customer_id'].nunique()} 客户")

    # 格式化fallback_data
    fb = fb.merge(cust_info, on='customer_id', how='left')
    fb = fb.merge(sku_info.rename(columns={'PRODUCT_NAME_CN': 'sku_name'}), left_on='sku_id', right_index=True, how='left')
    fb['recommended_cases'] = 1.0
    fb['algo_group'] = 'A'
    fb['模型'] = 'popular_sku_fallback'
    fb = fb.drop(columns=['channel'])
    fb = fb[['customer_id', 'customer_name', 'channel', 'route_id', 'sku_id', 'sku_name', 'recommended_cases', 'algo_group', '模型']]

    # 合并
    final_full = pd.concat([final, fb], ignore_index=True)
    final_full.to_csv("submission_final_complete.csv", index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    print(f"complete_submission_file: submission_final_complete.csv")
    print(f"{'=' * 60}")
    print(f"总行数: {len(final_full)}")
    print(f"总客户: {final_full['customer_id'].nunique()} / {len(all_custs)} ({100 * final_full['customer_id'].nunique() / len(all_custs):.0f}%)")
    print(f"\n渠道分布:")
    for ch_name, grp in final_full.groupby('channel'):
        n_cust = grp['customer_id'].nunique()
        print(f"  {ch_name}: {len(grp)} 行, {n_cust} 客户, 平均 {len(grp) / n_cust:.1f} SKU/客户")
    print(f"\n模型分布:")
    print(final_full['模型'].value_counts())


if __name__ == "__main__":
    main()
