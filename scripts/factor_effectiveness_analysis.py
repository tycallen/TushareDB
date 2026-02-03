#!/usr/bin/env python3
"""
因子有效性分析脚本
分析 stk_factor_pro 表中的因子有效性
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/factor_effectiveness_analysis.md'

# 定义因子分类和含义
FACTOR_DEFINITIONS = {
    # 基础行情
    'pct_chg': ('收益率', '日涨跌幅'),
    'turnover_rate': ('换手率', '换手率'),
    'turnover_rate_f': ('自由流通换手率', '自由流通股换手率'),
    'volume_ratio': ('量比', '量比'),

    # 估值因子
    'pe': ('市盈率', '静态市盈率'),
    'pe_ttm': ('市盈率TTM', '滚动市盈率'),
    'pb': ('市净率', '市净率'),
    'ps': ('市销率', '静态市销率'),
    'ps_ttm': ('市销率TTM', '滚动市销率'),
    'dv_ratio': ('股息率', '股息率'),
    'dv_ttm': ('股息率TTM', '滚动股息率'),

    # 市值因子
    'total_mv': ('总市值', '总市值'),
    'circ_mv': ('流通市值', '流通市值'),

    # 技术指标因子 (使用后复权 _hfq 版本，避免价格跳跃)
    'bias1_hfq': ('BIAS1', '乖离率(6日)'),
    'bias2_hfq': ('BIAS2', '乖离率(12日)'),
    'bias3_hfq': ('BIAS3', '乖离率(24日)'),
    'cci_hfq': ('CCI', '顺势指标'),
    'cr_hfq': ('CR', '带状能量线'),
    'dmi_adx_hfq': ('DMI_ADX', '趋向指标ADX'),
    'dmi_pdi_hfq': ('DMI_PDI', '趋向指标PDI'),
    'dmi_mdi_hfq': ('DMI_MDI', '趋向指标MDI'),
    'kdj_hfq': ('KDJ_J', 'KDJ指标J值'),
    'kdj_k_hfq': ('KDJ_K', 'KDJ指标K值'),
    'kdj_d_hfq': ('KDJ_D', 'KDJ指标D值'),
    'macd_hfq': ('MACD', 'MACD柱状值'),
    'macd_dif_hfq': ('MACD_DIF', 'MACD快线'),
    'macd_dea_hfq': ('MACD_DEA', 'MACD慢线'),
    'mfi_hfq': ('MFI', '资金流量指标'),
    'mtm_hfq': ('MTM', '动量指标'),
    'roc_hfq': ('ROC', '变动速率'),
    'rsi_hfq_6': ('RSI6', '6日相对强弱指标'),
    'rsi_hfq_12': ('RSI12', '12日相对强弱指标'),
    'rsi_hfq_24': ('RSI24', '24日相对强弱指标'),
    'wr_hfq': ('WR', '威廉指标'),
    'psy_hfq': ('PSY', '心理线'),
    'vr_hfq': ('VR', '成交量变异率'),
    'brar_ar_hfq': ('AR', '人气指标'),
    'brar_br_hfq': ('BR', '意愿指标'),
    'atr_hfq': ('ATR', '真实波幅'),
    'emv_hfq': ('EMV', '简易波动指标'),
    'trix_hfq': ('TRIX', '三重指数平滑移动平均'),
    'obv_hfq': ('OBV', '能量潮'),
    'mass_hfq': ('MASS', '梅斯线'),

    # 特殊因子
    'updays': ('连涨天数', '连续上涨天数'),
    'downdays': ('连跌天数', '连续下跌天数'),
    'topdays': ('距顶天数', '距离阶段高点天数'),
    'lowdays': ('距底天数', '距离阶段低点天数'),
}

# 选择用于分析的核心因子
CORE_FACTORS = [
    # 估值类
    'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm',
    # 市值类
    'total_mv', 'circ_mv',
    # 流动性类
    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
    # 动量类
    'bias1_hfq', 'bias2_hfq', 'bias3_hfq', 'mtm_hfq', 'roc_hfq',
    # 超买超卖类
    'rsi_hfq_6', 'rsi_hfq_12', 'cci_hfq', 'wr_hfq', 'kdj_hfq',
    # 趋势类
    'macd_hfq', 'dmi_adx_hfq', 'trix_hfq',
    # 成交量类
    'vr_hfq', 'obv_hfq', 'mfi_hfq',
    # 情绪类
    'psy_hfq', 'brar_ar_hfq', 'brar_br_hfq',
    # 波动率类
    'atr_hfq',
    # 时间类
    'updays', 'downdays',
]


def load_data(conn, sample_size=None, start_date='20200101'):
    """加载数据"""
    print(f"正在加载数据 (start_date >= {start_date})...")

    factor_cols = ', '.join(CORE_FACTORS)

    if sample_size:
        query = f"""
        SELECT ts_code, trade_date, pct_chg, {factor_cols}
        FROM stk_factor_pro
        WHERE trade_date >= '{start_date}'
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
    else:
        query = f"""
        SELECT ts_code, trade_date, pct_chg, {factor_cols}
        FROM stk_factor_pro
        WHERE trade_date >= '{start_date}'
        """

    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df)} 条记录")
    return df


def calculate_future_returns(conn, periods=[1, 5, 10, 20]):
    """计算未来N日收益率"""
    print("正在计算未来收益率...")

    factor_cols = ', '.join(CORE_FACTORS)

    query = f"""
    WITH base_data AS (
        SELECT
            ts_code,
            trade_date,
            close_hfq,
            pct_chg,
            {factor_cols}
        FROM stk_factor_pro
        WHERE trade_date >= '20200101'
    ),
    with_future AS (
        SELECT
            *,
            LEAD(close_hfq, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as close_1d,
            LEAD(close_hfq, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) as close_5d,
            LEAD(close_hfq, 10) OVER (PARTITION BY ts_code ORDER BY trade_date) as close_10d,
            LEAD(close_hfq, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) as close_20d
        FROM base_data
    )
    SELECT
        ts_code, trade_date, pct_chg,
        {factor_cols},
        (close_1d / close_hfq - 1) * 100 as ret_1d,
        (close_5d / close_hfq - 1) * 100 as ret_5d,
        (close_10d / close_hfq - 1) * 100 as ret_10d,
        (close_20d / close_hfq - 1) * 100 as ret_20d
    FROM with_future
    WHERE close_hfq IS NOT NULL AND close_hfq > 0
    """

    df = conn.execute(query).fetchdf()
    print(f"计算完成，共 {len(df)} 条记录")
    return df


def descriptive_statistics(df, factors):
    """计算因子描述性统计"""
    print("正在计算描述性统计...")

    stats_list = []
    for factor in factors:
        if factor not in df.columns:
            continue

        data = df[factor].dropna()
        if len(data) == 0:
            continue

        # 基本统计
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        median_val = data.median()

        # 分位数
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)

        # 偏度和峰度
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)

        # 缺失率
        missing_rate = df[factor].isna().sum() / len(df) * 100

        # 异常值比例 (超过3倍标准差)
        outlier_rate = ((data > mean_val + 3*std_val) | (data < mean_val - 3*std_val)).sum() / len(data) * 100

        factor_name = FACTOR_DEFINITIONS.get(factor, (factor, ''))[0]

        stats_list.append({
            '因子': factor,
            '因子名称': factor_name,
            '样本量': len(data),
            '缺失率%': round(missing_rate, 2),
            '均值': round(mean_val, 4),
            '标准差': round(std_val, 4),
            '最小值': round(min_val, 4),
            '25%分位': round(q1, 4),
            '中位数': round(median_val, 4),
            '75%分位': round(q3, 4),
            '最大值': round(max_val, 4),
            '偏度': round(skew_val, 4),
            '峰度': round(kurt_val, 4),
            '异常值%': round(outlier_rate, 2),
        })

    return pd.DataFrame(stats_list)


def calculate_correlation_matrix(df, factors):
    """计算因子相关系数矩阵"""
    print("正在计算相关系数矩阵...")

    valid_factors = [f for f in factors if f in df.columns]
    corr_matrix = df[valid_factors].corr()

    return corr_matrix


def cluster_factors(corr_matrix, threshold=0.7):
    """对高相关因子进行聚类"""
    print("正在进行因子聚类分析...")

    # 将相关系数转换为距离矩阵
    dist_matrix = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist_matrix, 0)

    # 确保距离矩阵是对称的
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # 层次聚类
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # 根据阈值划分聚类
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')

    # 整理聚类结果
    cluster_result = {}
    for i, factor in enumerate(corr_matrix.columns):
        cluster_id = clusters[i]
        if cluster_id not in cluster_result:
            cluster_result[cluster_id] = []
        cluster_result[cluster_id].append(factor)

    return cluster_result


def find_high_correlations(corr_matrix, threshold=0.7):
    """找出高度相关的因子对"""
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    '因子1': corr_matrix.columns[i],
                    '因子2': corr_matrix.columns[j],
                    '相关系数': round(corr_val, 4)
                })

    return pd.DataFrame(high_corr_pairs).sort_values('相关系数', key=abs, ascending=False)


def calculate_ic(df, factors, return_col='ret_5d'):
    """计算因子IC (Information Coefficient)"""
    print(f"正在计算IC (使用 {return_col})...")

    ic_results = []

    for factor in factors:
        if factor not in df.columns or return_col not in df.columns:
            continue

        # 按日期计算IC
        ic_by_date = df.groupby('trade_date').apply(
            lambda x: x[factor].corr(x[return_col], method='spearman')
            if len(x.dropna(subset=[factor, return_col])) > 30 else np.nan
        ).dropna()

        if len(ic_by_date) < 10:
            continue

        ic_mean = ic_by_date.mean()
        ic_std = ic_by_date.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_rate = (ic_by_date > 0).sum() / len(ic_by_date) * 100

        # IC的t检验
        t_stat, p_value = stats.ttest_1samp(ic_by_date, 0)

        factor_name = FACTOR_DEFINITIONS.get(factor, (factor, ''))[0]

        ic_results.append({
            '因子': factor,
            '因子名称': factor_name,
            'IC均值': round(ic_mean, 4),
            'IC标准差': round(ic_std, 4),
            'IR': round(ir, 4),
            'IC>0占比%': round(ic_positive_rate, 2),
            't统计量': round(t_stat, 4),
            'p值': round(p_value, 4),
            '显著性': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else '')),
        })

    return pd.DataFrame(ic_results).sort_values('IR', key=abs, ascending=False)


def calculate_ic_decay(df, factors, return_periods=['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d']):
    """计算IC衰减"""
    print("正在计算IC衰减...")

    decay_results = []

    for factor in factors:
        if factor not in df.columns:
            continue

        ic_values = {}
        for ret_col in return_periods:
            if ret_col not in df.columns:
                continue

            ic_by_date = df.groupby('trade_date').apply(
                lambda x: x[factor].corr(x[ret_col], method='spearman')
                if len(x.dropna(subset=[factor, ret_col])) > 30 else np.nan
            ).dropna()

            if len(ic_by_date) > 0:
                ic_values[ret_col] = round(ic_by_date.mean(), 4)

        if ic_values:
            factor_name = FACTOR_DEFINITIONS.get(factor, (factor, ''))[0]
            decay_results.append({
                '因子': factor,
                '因子名称': factor_name,
                **ic_values
            })

    return pd.DataFrame(decay_results)


def group_backtest(df, factor, n_groups=5, return_col='ret_5d'):
    """分组回测"""
    results = []

    for date, group in df.groupby('trade_date'):
        valid_data = group.dropna(subset=[factor, return_col])
        if len(valid_data) < n_groups * 10:
            continue

        # 按因子值分组
        valid_data = valid_data.copy()
        valid_data['group'] = pd.qcut(valid_data[factor], n_groups, labels=False, duplicates='drop')

        # 计算每组收益
        group_returns = valid_data.groupby('group')[return_col].mean()

        results.append({
            'date': date,
            **{f'G{i+1}': group_returns.get(i, np.nan) for i in range(n_groups)}
        })

    return pd.DataFrame(results)


def calculate_monotonicity(group_returns_df, n_groups=5):
    """计算因子单调性"""
    # 计算每组的平均收益
    group_means = group_returns_df[[f'G{i+1}' for i in range(n_groups)]].mean()

    # 计算单调性得分 (Spearman相关)
    ranks = np.arange(1, n_groups + 1)
    if group_means.isna().any():
        return np.nan, group_means.to_dict()

    monotonicity = stats.spearmanr(ranks, group_means.values)[0]

    return monotonicity, group_means.to_dict()


def batch_group_backtest(df, factors, n_groups=5, return_col='ret_5d'):
    """批量分组回测"""
    print(f"正在进行分组回测 (n_groups={n_groups}, return={return_col})...")

    results = []

    for factor in factors:
        if factor not in df.columns:
            continue

        group_returns = group_backtest(df, factor, n_groups, return_col)
        if len(group_returns) < 10:
            continue

        monotonicity, group_means = calculate_monotonicity(group_returns, n_groups)

        # 多空收益
        long_short = group_means.get(f'G{n_groups}', 0) - group_means.get('G1', 0)

        factor_name = FACTOR_DEFINITIONS.get(factor, (factor, ''))[0]

        results.append({
            '因子': factor,
            '因子名称': factor_name,
            'G1(最小)': round(group_means.get('G1', np.nan), 4),
            'G2': round(group_means.get('G2', np.nan), 4),
            'G3': round(group_means.get('G3', np.nan), 4),
            'G4': round(group_means.get('G4', np.nan), 4),
            'G5(最大)': round(group_means.get('G5', np.nan), 4),
            '多空收益': round(long_short, 4),
            '单调性': round(monotonicity, 4) if not np.isnan(monotonicity) else np.nan,
        })

    return pd.DataFrame(results).sort_values('单调性', key=abs, ascending=False)


def calculate_ic_autocorr(df, factors, return_col='ret_5d', lags=[1, 5, 10, 20]):
    """计算IC自相关性"""
    print("正在计算IC自相关性...")

    results = []

    for factor in factors:
        if factor not in df.columns:
            continue

        # 计算每日IC
        ic_series = df.groupby('trade_date').apply(
            lambda x: x[factor].corr(x[return_col], method='spearman')
            if len(x.dropna(subset=[factor, return_col])) > 30 else np.nan
        ).dropna()

        if len(ic_series) < max(lags) + 10:
            continue

        autocorr = {}
        for lag in lags:
            autocorr[f'lag{lag}'] = round(ic_series.autocorr(lag=lag), 4)

        factor_name = FACTOR_DEFINITIONS.get(factor, (factor, ''))[0]

        results.append({
            '因子': factor,
            '因子名称': factor_name,
            **autocorr
        })

    return pd.DataFrame(results)


def generate_factor_ranking(ic_df, group_df):
    """生成因子综合排名"""
    print("正在生成因子综合排名...")

    # 合并IC和分组回测结果
    ranking = ic_df[['因子', '因子名称', 'IC均值', 'IR', 'IC>0占比%', '显著性']].merge(
        group_df[['因子', '单调性', '多空收益']],
        on='因子',
        how='outer'
    )

    # 计算综合得分
    # IR绝对值 + 单调性绝对值 + IC正向比例/100
    ranking['综合得分'] = (
        ranking['IR'].abs() * 2 +  # IR权重较高
        ranking['单调性'].abs() +
        ranking['IC>0占比%'] / 100
    )

    ranking = ranking.sort_values('综合得分', ascending=False)
    ranking['排名'] = range(1, len(ranking) + 1)

    return ranking


def generate_report(stats_df, corr_matrix, high_corr_df, cluster_result,
                   ic_df, ic_decay_df, group_df, ic_autocorr_df, ranking_df):
    """生成分析报告"""
    print("正在生成分析报告...")

    report = []

    # 标题
    report.append("# 因子有效性分析报告")
    report.append("")
    report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("**数据源**: stk_factor_pro 表")
    report.append("")
    report.append("**分析周期**: 2020-01-01 至今")
    report.append("")

    # 目录
    report.append("## 目录")
    report.append("")
    report.append("1. [因子列表和定义](#1-因子列表和定义)")
    report.append("2. [描述性统计](#2-描述性统计)")
    report.append("3. [相关性分析](#3-相关性分析)")
    report.append("4. [IC/IR分析](#4-icir分析)")
    report.append("5. [分组回测](#5-分组回测)")
    report.append("6. [因子衰减分析](#6-因子衰减分析)")
    report.append("7. [有效因子排名](#7-有效因子排名)")
    report.append("8. [因子使用建议](#8-因子使用建议)")
    report.append("")

    # 1. 因子列表和定义
    report.append("## 1. 因子列表和定义")
    report.append("")
    report.append("| 因子代码 | 因子名称 | 因子描述 |")
    report.append("|---------|---------|---------|")
    for factor in CORE_FACTORS:
        if factor in FACTOR_DEFINITIONS:
            name, desc = FACTOR_DEFINITIONS[factor]
            report.append(f"| {factor} | {name} | {desc} |")
    report.append("")

    # 2. 描述性统计
    report.append("## 2. 描述性统计")
    report.append("")
    report.append("### 2.1 基本统计量")
    report.append("")

    # 简化表格，只显示核心统计量
    stats_simple = stats_df[['因子', '因子名称', '样本量', '缺失率%', '均值', '标准差', '偏度', '峰度', '异常值%']]
    report.append(stats_simple.to_markdown(index=False))
    report.append("")

    report.append("### 2.2 分布特征分析")
    report.append("")

    # 分析偏度和峰度
    high_skew = stats_df[stats_df['偏度'].abs() > 2]['因子'].tolist()
    high_kurt = stats_df[stats_df['峰度'].abs() > 7]['因子'].tolist()

    if high_skew:
        report.append(f"**高偏度因子** (|偏度|>2): {', '.join(high_skew)}")
        report.append("")
    if high_kurt:
        report.append(f"**高峰度因子** (|峰度|>7): {', '.join(high_kurt)}")
        report.append("")

    report.append("**建议**: 高偏度和高峰度因子在使用前建议进行标准化或分位数转换处理。")
    report.append("")

    # 3. 相关性分析
    report.append("## 3. 相关性分析")
    report.append("")

    report.append("### 3.1 高相关因子对 (|r|>=0.7)")
    report.append("")
    if len(high_corr_df) > 0:
        report.append(high_corr_df.to_markdown(index=False))
    else:
        report.append("没有发现高度相关的因子对。")
    report.append("")

    report.append("### 3.2 因子聚类结果")
    report.append("")
    report.append("基于相关性的因子聚类 (相关性阈值=0.7):")
    report.append("")
    for cluster_id, factors in sorted(cluster_result.items()):
        if len(factors) > 1:
            factor_names = [FACTOR_DEFINITIONS.get(f, (f, ''))[0] for f in factors]
            report.append(f"- **聚类 {cluster_id}**: {', '.join(factors)}")
            report.append(f"  - 因子名称: {', '.join(factor_names)}")
    report.append("")

    report.append("**建议**: 同一聚类内的因子高度相关，在构建多因子模型时应避免同时使用，可选择IC最高的因子作为代表。")
    report.append("")

    # 4. IC/IR分析
    report.append("## 4. IC/IR分析")
    report.append("")
    report.append("### 4.1 IC/IR统计 (5日收益)")
    report.append("")
    report.append("| 因子 | 因子名称 | IC均值 | IC标准差 | IR | IC>0占比% | 显著性 |")
    report.append("|-----|---------|--------|---------|-----|----------|--------|")
    for _, row in ic_df.head(20).iterrows():
        report.append(f"| {row['因子']} | {row['因子名称']} | {row['IC均值']} | {row['IC标准差']} | {row['IR']} | {row['IC>0占比%']} | {row['显著性']} |")
    report.append("")

    report.append("**IC评价标准**:")
    report.append("- |IC| > 0.05: 因子有一定预测能力")
    report.append("- |IC| > 0.1: 因子预测能力较强")
    report.append("- |IR| > 0.5: 因子较为稳定")
    report.append("- IC>0占比 > 55%: 因子方向一致性较好")
    report.append("")

    # 5. 分组回测
    report.append("## 5. 分组回测")
    report.append("")
    report.append("### 5.1 分组收益 (5日)")
    report.append("")
    report.append("将股票按因子值从小到大分为5组，计算每组平均收益:")
    report.append("")
    report.append(group_df.to_markdown(index=False))
    report.append("")

    report.append("### 5.2 单调性分析")
    report.append("")
    strong_mono = group_df[group_df['单调性'].abs() > 0.8]['因子'].tolist()
    if strong_mono:
        strong_mono_names = [FACTOR_DEFINITIONS.get(f, (f, ''))[0] for f in strong_mono]
        report.append(f"**高单调性因子** (|单调性|>0.8): {', '.join(strong_mono_names)}")
        report.append("")

    report.append("**单调性评价标准**:")
    report.append("- |单调性| > 0.8: 因子与收益有强单调关系")
    report.append("- |单调性| > 0.6: 因子与收益有中等单调关系")
    report.append("- 单调性为正: 因子值越大，收益越高")
    report.append("- 单调性为负: 因子值越小，收益越高")
    report.append("")

    # 6. 因子衰减分析
    report.append("## 6. 因子衰减分析")
    report.append("")
    report.append("### 6.1 不同持有期IC")
    report.append("")
    report.append(ic_decay_df.to_markdown(index=False))
    report.append("")

    report.append("### 6.2 IC自相关性")
    report.append("")
    if len(ic_autocorr_df) > 0:
        report.append(ic_autocorr_df.to_markdown(index=False))
        report.append("")

        report.append("**IC自相关性解读**:")
        report.append("- 高自相关: 因子预测能力稳定，变化缓慢")
        report.append("- 低自相关: 因子预测能力波动大，需要频繁调仓")
    report.append("")

    # 7. 有效因子排名
    report.append("## 7. 有效因子排名")
    report.append("")
    report.append("综合考虑IC、IR、单调性等指标的因子排名:")
    report.append("")
    ranking_cols = ['排名', '因子', '因子名称', 'IC均值', 'IR', 'IC>0占比%', '单调性', '多空收益', '显著性']
    ranking_display = ranking_df[[c for c in ranking_cols if c in ranking_df.columns]]
    report.append(ranking_display.to_markdown(index=False))
    report.append("")

    # 8. 因子使用建议
    report.append("## 8. 因子使用建议")
    report.append("")

    # 有效因子
    effective_factors = ranking_df[
        (ranking_df['IR'].abs() > 0.3) &
        (ranking_df['单调性'].abs() > 0.5)
    ]['因子'].tolist()

    report.append("### 8.1 推荐使用的因子")
    report.append("")
    if effective_factors:
        for f in effective_factors[:10]:
            name = FACTOR_DEFINITIONS.get(f, (f, ''))[0]
            row = ranking_df[ranking_df['因子'] == f].iloc[0]
            direction = "正向" if row.get('单调性', 0) > 0 else "反向"
            report.append(f"- **{name}** ({f}): IR={row['IR']}, 单调性={row['单调性']}, 建议使用方向: {direction}")
    else:
        report.append("未找到显著有效的因子。")
    report.append("")

    # 建议剔除的因子
    ineffective_factors = ranking_df[
        (ranking_df['IR'].abs() < 0.1) |
        (ranking_df['单调性'].abs() < 0.2)
    ]['因子'].tolist()

    report.append("### 8.2 建议剔除或谨慎使用的因子")
    report.append("")
    if ineffective_factors:
        for f in ineffective_factors[:5]:
            name = FACTOR_DEFINITIONS.get(f, (f, ''))[0]
            report.append(f"- **{name}** ({f}): 预测能力弱或单调性差")
    else:
        report.append("所有因子都有一定的预测能力。")
    report.append("")

    # 多因子组合建议
    report.append("### 8.3 多因子组合建议")
    report.append("")
    report.append("基于因子聚类和有效性分析，建议的多因子组合:")
    report.append("")

    # 从每个聚类中选择最有效的因子
    recommended_combo = []
    used_clusters = set()

    for _, row in ranking_df.iterrows():
        factor = row['因子']
        # 找出因子所属聚类
        for cluster_id, factors in cluster_result.items():
            if factor in factors and cluster_id not in used_clusters:
                recommended_combo.append(factor)
                used_clusters.add(cluster_id)
                break
        if len(recommended_combo) >= 5:
            break

    if recommended_combo:
        report.append("**推荐因子组合**:")
        for f in recommended_combo:
            name = FACTOR_DEFINITIONS.get(f, (f, ''))[0]
            desc = FACTOR_DEFINITIONS.get(f, ('', ''))[1]
            report.append(f"- {name} ({f}): {desc}")
    report.append("")

    report.append("### 8.4 注意事项")
    report.append("")
    report.append("1. **因子预处理**: 建议对因子进行去极值、标准化处理后再使用")
    report.append("2. **行业中性化**: 部分因子可能存在行业偏露，建议进行行业中性化处理")
    report.append("3. **市值中性化**: 部分因子与市值相关性高，建议进行市值中性化处理")
    report.append("4. **动态监控**: 因子有效性会随时间变化，建议定期重新评估")
    report.append("5. **交易成本**: 高换手率因子需考虑交易成本对收益的影响")
    report.append("")

    return '\n'.join(report)


def main():
    """主函数"""
    print("=" * 60)
    print("因子有效性分析")
    print("=" * 60)

    # 连接数据库
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 加载数据并计算未来收益
    df = calculate_future_returns(conn)

    # 1. 描述性统计
    stats_df = descriptive_statistics(df, CORE_FACTORS)
    print(f"描述性统计完成: {len(stats_df)} 个因子")

    # 2. 相关性分析
    corr_matrix = calculate_correlation_matrix(df, CORE_FACTORS)
    high_corr_df = find_high_correlations(corr_matrix, threshold=0.7)
    cluster_result = cluster_factors(corr_matrix, threshold=0.7)
    print(f"相关性分析完成: 发现 {len(high_corr_df)} 对高相关因子")

    # 3. IC/IR分析
    ic_df = calculate_ic(df, CORE_FACTORS, return_col='ret_5d')
    print(f"IC分析完成: {len(ic_df)} 个因子")

    # 4. 分组回测
    group_df = batch_group_backtest(df, CORE_FACTORS, n_groups=5, return_col='ret_5d')
    print(f"分组回测完成: {len(group_df)} 个因子")

    # 5. 因子衰减分析
    ic_decay_df = calculate_ic_decay(df, CORE_FACTORS)
    ic_autocorr_df = calculate_ic_autocorr(df, CORE_FACTORS)
    print(f"因子衰减分析完成")

    # 6. 综合排名
    ranking_df = generate_factor_ranking(ic_df, group_df)
    print(f"综合排名完成")

    # 7. 生成报告
    report = generate_report(
        stats_df, corr_matrix, high_corr_df, cluster_result,
        ic_df, ic_decay_df, group_df, ic_autocorr_df, ranking_df
    )

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print("=" * 60)
    print(f"分析报告已保存至: {REPORT_PATH}")
    print("=" * 60)

    conn.close()


if __name__ == '__main__':
    main()
