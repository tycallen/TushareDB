#!/usr/bin/env python3
"""
技术指标特征统计分析

对10只代表性股票计算所有技术指标特征，并统计:
- 各特征的均值、标准差、分位数
- 缺失率和异常值比例
- 特征相关性分析
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入技术指标计算模块
from technical_indicators import calculate_all_features, get_feature_list


def get_representative_stocks(conn) -> List[str]:
    """
    选取10只代表性股票

    选取标准:
    - 数据量充足 (>5000条)
    - 覆盖不同市场 (上海/深圳)
    - 覆盖不同板块 (主板/中小板/创业板)
    """
    # 按数据量选取
    stocks = [
        '000001.SZ',  # 平安银行 - 深圳主板 金融
        '600519.SH',  # 贵州茅台 - 上海主板 消费
        '000858.SZ',  # 五粮液 - 深圳主板 消费
        '601318.SH',  # 中国平安 - 上海主板 金融
        '000333.SZ',  # 美的集团 - 深圳主板 家电
        '600036.SH',  # 招商银行 - 上海主板 金融
        '002415.SZ',  # 海康威视 - 中小板 科技
        '300750.SZ',  # 宁德时代 - 创业板 新能源
        '601012.SH',  # 隆基绿能 - 上海主板 新能源
        '000725.SZ',  # 京东方A - 深圳主板 科技
    ]

    # 验证股票是否存在
    valid_stocks = []
    for code in stocks:
        count = conn.execute(f"SELECT COUNT(*) FROM daily WHERE ts_code = '{code}'").fetchone()[0]
        if count > 100:
            valid_stocks.append(code)
            print(f"  {code}: {count} 条数据")

    # 如果不够10只，补充数据量最大的股票
    if len(valid_stocks) < 10:
        additional = conn.execute(f"""
            SELECT ts_code, COUNT(*) as cnt
            FROM daily
            WHERE ts_code NOT IN ({','.join([f"'{s}'" for s in valid_stocks])})
            GROUP BY ts_code
            ORDER BY cnt DESC
            LIMIT {10 - len(valid_stocks)}
        """).fetchdf()
        for _, row in additional.iterrows():
            valid_stocks.append(row['ts_code'])
            print(f"  {row['ts_code']}: {row['cnt']} 条数据 (补充)")

    return valid_stocks[:10]


def load_stock_data(conn, ts_code: str) -> pd.DataFrame:
    """
    加载单只股票数据 (带复权因子)
    """
    df = conn.execute(f"""
        SELECT d.*, a.adj_factor
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = '{ts_code}'
        ORDER BY d.trade_date
    """).fetchdf()
    return df


def calculate_feature_statistics(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征的统计指标
    """
    # 选择数值列 (排除 ts_code, trade_date)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

    stats = []
    for col in numeric_cols:
        data = features_df[col]

        # 基础统计
        stat = {
            'feature': col,
            'count': data.count(),
            'missing': data.isna().sum(),
            'missing_rate': data.isna().mean() * 100,
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q25': data.quantile(0.25),
            'median': data.median(),
            'q75': data.quantile(0.75),
            'max': data.max(),
        }

        # 异常值检测 (使用 IQR 方法)
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        stat['outlier_count'] = outliers
        stat['outlier_rate'] = outliers / data.count() * 100 if data.count() > 0 else 0

        # 偏度和峰度
        stat['skewness'] = data.skew()
        stat['kurtosis'] = data.kurtosis()

        stats.append(stat)

    return pd.DataFrame(stats)


def analyze_missing_patterns(all_features: pd.DataFrame) -> pd.DataFrame:
    """
    分析缺失值模式
    """
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns.tolist()

    patterns = []
    for col in numeric_cols:
        # 按股票统计缺失
        by_stock = all_features.groupby('ts_code')[col].apply(lambda x: x.isna().mean() * 100)

        patterns.append({
            'feature': col,
            'overall_missing_rate': all_features[col].isna().mean() * 100,
            'min_missing_by_stock': by_stock.min(),
            'max_missing_by_stock': by_stock.max(),
            'avg_missing_by_stock': by_stock.mean(),
        })

    return pd.DataFrame(patterns)


def main():
    print("=" * 80)
    print("技术指标特征统计分析")
    print("=" * 80)

    # 连接数据库
    conn = duckdb.connect('/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db', read_only=True)

    # 1. 选取代表性股票
    print("\n1. 选取10只代表性股票")
    print("-" * 40)
    stocks = get_representative_stocks(conn)
    print(f"\n选取股票: {stocks}")

    # 2. 计算所有股票的特征
    print("\n2. 计算技术指标特征")
    print("-" * 40)
    all_features = []

    for code in stocks:
        print(f"  处理 {code}...")
        df = load_stock_data(conn, code)
        if len(df) > 0:
            features = calculate_all_features(df)
            all_features.append(features)

    all_features = pd.concat(all_features, ignore_index=True)
    print(f"\n总计: {len(all_features)} 条记录, {len(all_features.columns)} 个特征")

    # 3. 统计分析
    print("\n3. 特征统计分析")
    print("-" * 40)
    stats = calculate_feature_statistics(all_features)

    # 4. 分类统计
    feature_groups = get_feature_list()
    print("\n按类别统计:")

    for group_name, features in feature_groups.items():
        group_stats = stats[stats['feature'].isin(features)]
        if len(group_stats) > 0:
            print(f"\n【{group_name}】")
            print(f"  特征数: {len(group_stats)}")
            print(f"  平均缺失率: {group_stats['missing_rate'].mean():.2f}%")
            print(f"  平均异常值率: {group_stats['outlier_rate'].mean():.2f}%")

    # 5. 输出详细统计
    print("\n4. 详细统计表")
    print("-" * 40)

    # 格式化输出
    stats_display = stats[['feature', 'count', 'missing_rate', 'mean', 'std',
                           'min', 'median', 'max', 'outlier_rate', 'skewness']].copy()
    stats_display = stats_display.round(4)

    print("\n关键特征统计:")
    key_features = ['close', 'ma_20', 'macd', 'rsi_14', 'atr', 'obv', 'bb_percent_b']
    for f in key_features:
        if f in stats['feature'].values:
            row = stats[stats['feature'] == f].iloc[0]
            print(f"\n  {f}:")
            print(f"    均值: {row['mean']:.4f}, 标准差: {row['std']:.4f}")
            print(f"    范围: [{row['min']:.4f}, {row['max']:.4f}]")
            print(f"    分位数: Q25={row['q25']:.4f}, 中位数={row['median']:.4f}, Q75={row['q75']:.4f}")
            print(f"    缺失率: {row['missing_rate']:.2f}%, 异常值率: {row['outlier_rate']:.2f}%")

    # 6. 缺失值分析
    print("\n5. 缺失值模式分析")
    print("-" * 40)
    missing_patterns = analyze_missing_patterns(all_features)

    # 找出缺失率最高的特征
    high_missing = missing_patterns[missing_patterns['overall_missing_rate'] > 5].sort_values(
        'overall_missing_rate', ascending=False
    )
    if len(high_missing) > 0:
        print("\n缺失率 > 5% 的特征:")
        for _, row in high_missing.iterrows():
            print(f"  {row['feature']}: {row['overall_missing_rate']:.2f}%")
    else:
        print("\n所有特征缺失率均 < 5%")

    # 7. 异常值分析
    print("\n6. 异常值分析")
    print("-" * 40)
    high_outlier = stats[stats['outlier_rate'] > 10].sort_values('outlier_rate', ascending=False)
    if len(high_outlier) > 0:
        print("\n异常值率 > 10% 的特征:")
        for _, row in high_outlier.iterrows():
            print(f"  {row['feature']}: {row['outlier_rate']:.2f}%")
    else:
        print("\n所有特征异常值率均 < 10%")

    # 8. 保存结果
    output_dir = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check'

    # 保存统计结果
    stats.to_csv(f'{output_dir}/feature_statistics.csv', index=False)
    missing_patterns.to_csv(f'{output_dir}/missing_patterns.csv', index=False)

    # 保存样例数据
    sample_features = all_features[all_features['ts_code'] == stocks[0]].tail(100)
    sample_features.to_csv(f'{output_dir}/sample_features.csv', index=False)

    print(f"\n7. 结果已保存到:")
    print(f"  - {output_dir}/feature_statistics.csv")
    print(f"  - {output_dir}/missing_patterns.csv")
    print(f"  - {output_dir}/sample_features.csv")

    # 9. 生成摘要报告
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"""
特征总数: {len(stats)} 个
数据总量: {len(all_features)} 条记录

缺失率统计:
  - 平均缺失率: {stats['missing_rate'].mean():.2f}%
  - 最大缺失率: {stats['missing_rate'].max():.2f}% ({stats.loc[stats['missing_rate'].idxmax(), 'feature']})
  - 无缺失特征: {(stats['missing_rate'] == 0).sum()} 个

异常值统计:
  - 平均异常值率: {stats['outlier_rate'].mean():.2f}%
  - 最大异常值率: {stats['outlier_rate'].max():.2f}% ({stats.loc[stats['outlier_rate'].idxmax(), 'feature']})

数据分布:
  - 高偏度特征 (|skew| > 2): {(stats['skewness'].abs() > 2).sum()} 个
  - 高峰度特征 (kurt > 10): {(stats['kurtosis'] > 10).sum()} 个
""")

    conn.close()


if __name__ == '__main__':
    main()
