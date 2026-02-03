#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
申万行业分类深度分析
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/sw_industry_analysis.md'

def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

def query(sql):
    conn = get_conn()
    df = conn.execute(sql).fetchdf()
    conn.close()
    return df

# =============================================================================
# 1. 行业分类体系分析
# =============================================================================
print("=" * 60)
print("1. 行业分类体系分析")
print("=" * 60)

# 一级行业
l1_industries = query("""
    SELECT index_code, industry_name, industry_code, src
    FROM index_classify
    WHERE level='L1'
    ORDER BY industry_code
""")
print(f"\n一级行业数量: {len(l1_industries)}")
print(l1_industries[['industry_name', 'index_code', 'src']].to_string())

# 二级行业
l2_industries = query("""
    SELECT index_code, industry_name, industry_code, parent_code, src
    FROM index_classify
    WHERE level='L2'
    ORDER BY industry_code
""")
print(f"\n二级行业数量: {len(l2_industries)}")

# 三级行业
l3_industries = query("""
    SELECT index_code, industry_name, industry_code, parent_code, src
    FROM index_classify
    WHERE level='L3'
    ORDER BY industry_code
""")
print(f"三级行业数量: {len(l3_industries)}")

# 各一级行业下的二三级行业数量
l1_l2_count = query("""
    SELECT
        l1.industry_name as l1_name,
        COUNT(DISTINCT l2.industry_code) as l2_count,
        COUNT(DISTINCT l3.industry_code) as l3_count
    FROM index_classify l1
    LEFT JOIN index_classify l2 ON l2.parent_code = l1.industry_code AND l2.level='L2'
    LEFT JOIN index_classify l3 ON l3.parent_code = l2.industry_code AND l3.level='L3'
    WHERE l1.level='L1'
    GROUP BY l1.industry_name, l1.industry_code
    ORDER BY l1.industry_code
""")
print("\n各一级行业下的二级、三级行业数量:")
print(l1_l2_count.to_string())

# 各一级行业股票数量分布
stock_by_l1 = query("""
    SELECT
        l1_name,
        COUNT(DISTINCT ts_code) as stock_count
    FROM index_member_all
    WHERE is_new = 'Y'
    GROUP BY l1_name
    ORDER BY stock_count DESC
""")
print("\n各一级行业股票数量分布:")
print(stock_by_l1.to_string())

# 最新市值分布
latest_date = query("SELECT MAX(trade_date) as d FROM sw_daily").iloc[0]['d']
print(f"\n数据最新日期: {latest_date}")

market_value_by_l1 = query(f"""
    SELECT
        ic.industry_name,
        SUM(sd.total_mv) / 100000000 as total_mv_billion,
        AVG(sd.total_mv) / 100000000 as avg_mv_billion,
        AVG(sd.pe) as avg_pe,
        AVG(sd.pb) as avg_pb
    FROM sw_daily sd
    JOIN index_classify ic ON sd.ts_code = ic.index_code
    WHERE sd.trade_date = '{latest_date}'
    AND ic.level = 'L1'
    GROUP BY ic.industry_name
    ORDER BY total_mv_billion DESC
""")
print("\n各一级行业市值与估值(亿元):")
print(market_value_by_l1.to_string())

# =============================================================================
# 2. 行业财务特征
# =============================================================================
print("\n" + "=" * 60)
print("2. 行业财务特征")
print("=" * 60)

# 获取最新财报数据
latest_fina_date = query("SELECT MAX(end_date) as d FROM fina_indicator_vip WHERE end_date LIKE '%1231'").iloc[0]['d']
print(f"\n最新年报日期: {latest_fina_date}")

# 各行业ROE分布
roe_by_industry = query(f"""
    WITH stock_roe AS (
        SELECT
            f.ts_code,
            f.roe,
            f.roe_waa,
            f.roe_dt,
            f.gross_margin,
            f.netprofit_margin,
            f.current_ratio,
            f.quick_ratio,
            f.debt_to_assets,
            im.l1_name
        FROM fina_indicator_vip f
        JOIN index_member_all im ON f.ts_code = im.ts_code
        WHERE f.end_date = '{latest_fina_date}'
        AND im.is_new = 'Y'
        AND f.roe IS NOT NULL
        AND f.roe > -100 AND f.roe < 100
    )
    SELECT
        l1_name,
        COUNT(*) as stock_count,
        AVG(roe) as avg_roe,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roe) as median_roe,
        STDDEV(roe) as std_roe,
        MIN(roe) as min_roe,
        MAX(roe) as max_roe,
        AVG(gross_margin) as avg_gross_margin,
        AVG(netprofit_margin) as avg_net_margin,
        AVG(debt_to_assets) as avg_debt_ratio
    FROM stock_roe
    GROUP BY l1_name
    ORDER BY avg_roe DESC
""")
print("\n各行业ROE分布(%):")
print(roe_by_industry.to_string())

# 行业盈利能力排名
profit_ranking = query(f"""
    WITH stock_profit AS (
        SELECT
            f.ts_code,
            f.roe,
            f.roa,
            f.gross_margin,
            f.netprofit_margin,
            f.op_of_gr,
            im.l1_name
        FROM fina_indicator_vip f
        JOIN index_member_all im ON f.ts_code = im.ts_code
        WHERE f.end_date = '{latest_fina_date}'
        AND im.is_new = 'Y'
        AND f.roe IS NOT NULL
    )
    SELECT
        l1_name,
        ROUND(AVG(roe), 2) as avg_roe,
        ROUND(AVG(roa), 2) as avg_roa,
        ROUND(AVG(gross_margin), 2) as avg_gross_margin,
        ROUND(AVG(netprofit_margin), 2) as avg_net_margin
    FROM stock_profit
    GROUP BY l1_name
    ORDER BY avg_roe DESC
""")
print("\n行业盈利能力排名:")
print(profit_ranking.to_string())

# 行业估值分位数(使用行业指数PE/PB历史数据)
valuation_percentile = query(f"""
    WITH hist_pe AS (
        SELECT
            ts_code,
            AVG(pe) as avg_pe,
            PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY pe) as pe_20pct,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe) as pe_50pct,
            PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY pe) as pe_80pct
        FROM sw_daily
        WHERE pe > 0 AND pe < 500
        AND trade_date >= '20210101'
        GROUP BY ts_code
    ),
    current_pe AS (
        SELECT ts_code, pe as current_pe, pb as current_pb
        FROM sw_daily
        WHERE trade_date = '{latest_date}'
    )
    SELECT
        ic.industry_name,
        ROUND(cp.current_pe, 2) as current_pe,
        ROUND(hp.avg_pe, 2) as avg_pe,
        ROUND(hp.pe_20pct, 2) as pe_20pct,
        ROUND(hp.pe_50pct, 2) as pe_median,
        ROUND(hp.pe_80pct, 2) as pe_80pct,
        CASE
            WHEN cp.current_pe < hp.pe_20pct THEN '低估'
            WHEN cp.current_pe > hp.pe_80pct THEN '高估'
            ELSE '合理'
        END as valuation_status
    FROM hist_pe hp
    JOIN current_pe cp ON hp.ts_code = cp.ts_code
    JOIN index_classify ic ON hp.ts_code = ic.index_code
    WHERE ic.level = 'L1'
    ORDER BY current_pe
""")
print("\n行业估值分位数分析:")
print(valuation_percentile.to_string())

# =============================================================================
# 3. 行业收益分析
# =============================================================================
print("\n" + "=" * 60)
print("3. 行业收益分析")
print("=" * 60)

# 计算各行业收益率和波动率
industry_returns = query("""
    WITH daily_returns AS (
        SELECT
            sd.ts_code,
            sd.trade_date,
            sd.pct_change / 100 as daily_return,
            ic.industry_name
        FROM sw_daily sd
        JOIN index_classify ic ON sd.ts_code = ic.index_code
        WHERE ic.level = 'L1'
        AND sd.trade_date >= '20210101'
    ),
    yearly_stats AS (
        SELECT
            industry_name,
            SUBSTR(trade_date, 1, 4) as year,
            EXP(SUM(LN(1 + daily_return))) - 1 as annual_return,
            STDDEV(daily_return) * SQRT(252) as annual_vol,
            COUNT(*) as trading_days
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY industry_name, SUBSTR(trade_date, 1, 4)
    )
    SELECT
        industry_name,
        year,
        ROUND(annual_return * 100, 2) as return_pct,
        ROUND(annual_vol * 100, 2) as vol_pct,
        ROUND(annual_return / NULLIF(annual_vol, 0), 3) as sharpe
    FROM yearly_stats
    ORDER BY industry_name, year
""")
print("\n各行业年度收益统计:")
# 转为宽表
returns_pivot = industry_returns.pivot(index='industry_name', columns='year', values='return_pct')
print(returns_pivot.to_string())

# 整体期间统计
overall_stats = query("""
    WITH daily_returns AS (
        SELECT
            sd.ts_code,
            sd.trade_date,
            sd.pct_change / 100 as daily_return,
            ic.industry_name
        FROM sw_daily sd
        JOIN index_classify ic ON sd.ts_code = ic.index_code
        WHERE ic.level = 'L1'
        AND sd.trade_date >= '20210101'
    )
    SELECT
        industry_name,
        ROUND((EXP(SUM(LN(1 + daily_return))) - 1) * 100, 2) as total_return_pct,
        ROUND(AVG(daily_return) * 252 * 100, 2) as annualized_return_pct,
        ROUND(STDDEV(daily_return) * SQRT(252) * 100, 2) as annualized_vol_pct,
        ROUND(AVG(daily_return) * 252 / NULLIF(STDDEV(daily_return) * SQRT(252), 0), 3) as sharpe_ratio,
        COUNT(*) as trading_days
    FROM daily_returns
    WHERE daily_return IS NOT NULL
    GROUP BY industry_name
    ORDER BY total_return_pct DESC
""")
print("\n2021年以来行业整体表现:")
print(overall_stats.to_string())

# 计算最大回撤
max_drawdown = query("""
    WITH daily_data AS (
        SELECT
            sd.ts_code,
            sd.trade_date,
            sd.close,
            ic.industry_name,
            MAX(sd.close) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date ROWS UNBOUNDED PRECEDING) as running_max
        FROM sw_daily sd
        JOIN index_classify ic ON sd.ts_code = ic.index_code
        WHERE ic.level = 'L1'
        AND sd.trade_date >= '20210101'
    ),
    drawdowns AS (
        SELECT
            industry_name,
            trade_date,
            (close - running_max) / running_max as drawdown
        FROM daily_data
    )
    SELECT
        industry_name,
        ROUND(MIN(drawdown) * 100, 2) as max_drawdown_pct
    FROM drawdowns
    GROUP BY industry_name
    ORDER BY max_drawdown_pct DESC
""")
print("\n行业最大回撤:")
print(max_drawdown.to_string())

# =============================================================================
# 4. 行业相关性分析
# =============================================================================
print("\n" + "=" * 60)
print("4. 行业相关性分析")
print("=" * 60)

# 获取所有一级行业的日收益率
returns_data = query("""
    SELECT
        sd.trade_date,
        ic.industry_name,
        sd.pct_change
    FROM sw_daily sd
    JOIN index_classify ic ON sd.ts_code = ic.index_code
    WHERE ic.level = 'L1'
    AND sd.trade_date >= '20210101'
    ORDER BY sd.trade_date, ic.industry_name
""")

# 透视表
returns_matrix = returns_data.pivot(index='trade_date', columns='industry_name', values='pct_change')
returns_matrix = returns_matrix.dropna()

# 相关性矩阵
corr_matrix = returns_matrix.corr()
print("\n行业相关性矩阵(部分):")
# 只显示相关性较高的组合
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        high_corr.append({
            'industry1': corr_matrix.columns[i],
            'industry2': corr_matrix.columns[j],
            'correlation': round(corr_val, 3)
        })

high_corr_df = pd.DataFrame(high_corr).sort_values('correlation', ascending=False)
print("\n高相关性行业组合 (>0.7):")
print(high_corr_df[high_corr_df['correlation'] > 0.7].to_string())

print("\n低相关性行业组合 (<0.4):")
print(high_corr_df[high_corr_df['correlation'] < 0.4].head(20).to_string())

# 行业聚类分析
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 将相关性转换为距离
distance_matrix = 1 - corr_matrix
linkage_matrix = linkage(squareform(distance_matrix), method='ward')
clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

cluster_df = pd.DataFrame({
    'industry': corr_matrix.columns,
    'cluster': clusters
}).sort_values('cluster')
print("\n行业聚类结果(3类):")
for c in sorted(cluster_df['cluster'].unique()):
    industries = cluster_df[cluster_df['cluster'] == c]['industry'].tolist()
    print(f"  类别{c}: {', '.join(industries)}")

# 防御性 vs 周期性分析
# 使用与大盘相关性来区分
# 先计算各行业与整体市场的beta
market_return = returns_matrix.mean(axis=1)  # 简单用等权平均代替市场
betas = {}
for col in returns_matrix.columns:
    cov = np.cov(returns_matrix[col].dropna(), market_return.loc[returns_matrix[col].dropna().index])[0, 1]
    var = np.var(market_return.loc[returns_matrix[col].dropna().index])
    betas[col] = cov / var if var > 0 else 0

beta_df = pd.DataFrame({'industry': list(betas.keys()), 'beta': list(betas.values())})
beta_df = beta_df.sort_values('beta')
print("\n行业Beta值(防御性 vs 周期性):")
print("防御性行业 (Beta < 0.9):")
print(beta_df[beta_df['beta'] < 0.9].to_string())
print("\n周期性行业 (Beta > 1.1):")
print(beta_df[beta_df['beta'] > 1.1].to_string())

# =============================================================================
# 5. 行业动量分析
# =============================================================================
print("\n" + "=" * 60)
print("5. 行业动量分析")
print("=" * 60)

# 不同时间窗口的动量
momentum = query(f"""
    WITH price_data AS (
        SELECT
            sd.ts_code,
            sd.trade_date,
            sd.close,
            ic.industry_name,
            LAG(sd.close, 5) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date) as close_5d,
            LAG(sd.close, 20) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date) as close_20d,
            LAG(sd.close, 60) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date) as close_60d,
            LAG(sd.close, 120) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date) as close_120d,
            LAG(sd.close, 250) OVER (PARTITION BY sd.ts_code ORDER BY sd.trade_date) as close_250d
        FROM sw_daily sd
        JOIN index_classify ic ON sd.ts_code = ic.index_code
        WHERE ic.level = 'L1'
    )
    SELECT
        industry_name,
        ROUND((close / close_5d - 1) * 100, 2) as mom_1w,
        ROUND((close / close_20d - 1) * 100, 2) as mom_1m,
        ROUND((close / close_60d - 1) * 100, 2) as mom_3m,
        ROUND((close / close_120d - 1) * 100, 2) as mom_6m,
        ROUND((close / close_250d - 1) * 100, 2) as mom_1y
    FROM price_data
    WHERE trade_date = '{latest_date}'
    ORDER BY mom_3m DESC
""")
print("\n各行业动量(%):")
print(momentum.to_string())

# 近期领涨行业
print("\n近1周领涨行业:")
print(momentum.sort_values('mom_1w', ascending=False).head(5)[['industry_name', 'mom_1w']].to_string())

print("\n近1月领涨行业:")
print(momentum.sort_values('mom_1m', ascending=False).head(5)[['industry_name', 'mom_1m']].to_string())

print("\n近3月领涨行业:")
print(momentum.sort_values('mom_3m', ascending=False).head(5)[['industry_name', 'mom_3m']].to_string())

# 行业轮动分析 - 计算月度行业排名变化
monthly_ranking = query("""
    WITH monthly_returns AS (
        SELECT
            sd.ts_code,
            ic.industry_name,
            SUBSTR(sd.trade_date, 1, 6) as month,
            (LAST_VALUE(sd.close) OVER (
                PARTITION BY sd.ts_code, SUBSTR(sd.trade_date, 1, 6)
                ORDER BY sd.trade_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) / FIRST_VALUE(sd.close) OVER (
                PARTITION BY sd.ts_code, SUBSTR(sd.trade_date, 1, 6)
                ORDER BY sd.trade_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) - 1) * 100 as monthly_return
        FROM sw_daily sd
        JOIN index_classify ic ON sd.ts_code = ic.index_code
        WHERE ic.level = 'L1'
        AND sd.trade_date >= '20240101'
    ),
    distinct_returns AS (
        SELECT DISTINCT industry_name, month, monthly_return
        FROM monthly_returns
    ),
    ranked AS (
        SELECT
            industry_name,
            month,
            monthly_return,
            RANK() OVER (PARTITION BY month ORDER BY monthly_return DESC) as rank
        FROM distinct_returns
    )
    SELECT industry_name, month, rank
    FROM ranked
    WHERE industry_name IN ('煤炭', '银行', '电子', '计算机', '医药生物')
    ORDER BY month, rank
""")
# 转为宽表显示排名变化
ranking_pivot = monthly_ranking.pivot(index='industry_name', columns='month', values='rank')
print("\n部分行业月度排名变化(2024年以来):")
print(ranking_pivot.to_string())

# =============================================================================
# 6. 行业配置建议
# =============================================================================
print("\n" + "=" * 60)
print("6. 行业配置建议")
print("=" * 60)

# 综合评分 - 结合估值、动量、ROE
# 1. 估值评分 (越低越好)
valuation_score = valuation_percentile.copy()
valuation_score['valuation_score'] = 100 - (valuation_score['current_pe'].rank(pct=True) * 100)

# 2. 动量评分
momentum_score = momentum.copy()
momentum_score['momentum_score'] = momentum_score['mom_3m'].rank(pct=True) * 100

# 3. ROE评分
roe_score = roe_by_industry.copy()
roe_score['roe_score'] = roe_score['avg_roe'].rank(pct=True) * 100

# 合并评分
comprehensive = pd.merge(
    valuation_score[['industry_name', 'current_pe', 'valuation_status', 'valuation_score']],
    momentum_score[['industry_name', 'mom_3m', 'momentum_score']],
    on='industry_name'
)
comprehensive = pd.merge(
    comprehensive,
    roe_score[['l1_name', 'avg_roe', 'roe_score']],
    left_on='industry_name',
    right_on='l1_name'
)

# 综合评分 (等权)
comprehensive['total_score'] = (comprehensive['valuation_score'] + comprehensive['momentum_score'] + comprehensive['roe_score']) / 3
comprehensive = comprehensive.sort_values('total_score', ascending=False)

print("\n行业综合评分(估值+动量+ROE):")
print(comprehensive[['industry_name', 'current_pe', 'valuation_status', 'mom_3m', 'avg_roe', 'total_score']].head(15).to_string())

# 配置建议
print("\n" + "=" * 60)
print("配置建议")
print("=" * 60)

# 低估值高ROE行业
print("\n推荐配置(低估值+高ROE):")
recommended = comprehensive[
    (comprehensive['valuation_status'] == '低估') &
    (comprehensive['roe_score'] > 50)
].sort_values('total_score', ascending=False)
if len(recommended) > 0:
    print(recommended[['industry_name', 'current_pe', 'avg_roe', 'total_score']].to_string())
else:
    # 放宽条件
    recommended = comprehensive[comprehensive['valuation_status'] == '低估'].sort_values('total_score', ascending=False).head(5)
    print(recommended[['industry_name', 'current_pe', 'avg_roe', 'total_score']].to_string())

# 强动量行业
print("\n强动量行业:")
strong_momentum = comprehensive.sort_values('mom_3m', ascending=False).head(5)
print(strong_momentum[['industry_name', 'mom_3m', 'current_pe', 'total_score']].to_string())

# 防御性配置
print("\n防御性配置建议(低Beta + 低波动):")
# 合并beta数据
comprehensive_with_beta = pd.merge(comprehensive, beta_df, left_on='industry_name', right_on='industry')
# 合并波动率数据
comprehensive_with_beta = pd.merge(comprehensive_with_beta, overall_stats[['industry_name', 'annualized_vol_pct']], on='industry_name')
defensive = comprehensive_with_beta[comprehensive_with_beta['beta'] < 0.95].sort_values('annualized_vol_pct')
print(defensive[['industry_name', 'beta', 'annualized_vol_pct', 'current_pe', 'avg_roe']].head(5).to_string())

# =============================================================================
# 生成报告
# =============================================================================
report = f"""# 申万行业分类深度分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据截止日**: {latest_date}
**分析区间**: 2021-01-01 至 {latest_date}

---

## 一、行业分类体系分析

### 1.1 分类体系概览

申万行业分类采用三级分类体系:
- **一级行业**: {len(l1_industries)} 个
- **二级行业**: {len(l2_industries)} 个
- **三级行业**: {len(l3_industries)} 个
- **覆盖股票**: 5,809 只

### 1.2 一级行业列表

| 行业名称 | 指数代码 | 版本 |
|---------|---------|------|
"""

for _, row in l1_industries.iterrows():
    report += f"| {row['industry_name']} | {row['index_code']} | {row['src']} |\n"

report += f"""
### 1.3 各一级行业子行业数量

| 一级行业 | 二级行业数 | 三级行业数 |
|---------|-----------|-----------|
"""
for _, row in l1_l2_count.iterrows():
    report += f"| {row['l1_name']} | {int(row['l2_count'])} | {int(row['l3_count'])} |\n"

report += f"""
### 1.4 各行业股票数量分布

| 行业 | 股票数量 |
|-----|---------|
"""
for _, row in stock_by_l1.iterrows():
    report += f"| {row['l1_name']} | {row['stock_count']} |\n"

report += f"""
### 1.5 行业市值与估值分布

| 行业 | 总市值(亿) | 平均市值(亿) | 平均PE | 平均PB |
|-----|-----------|-------------|--------|--------|
"""
for _, row in market_value_by_l1.iterrows():
    report += f"| {row['industry_name']} | {row['total_mv_billion']:.0f} | {row['avg_mv_billion']:.0f} | {row['avg_pe']:.1f} | {row['avg_pb']:.2f} |\n"

report += f"""

---

## 二、行业财务特征

### 2.1 行业ROE分布

基于{latest_fina_date[:4]}年年报数据:

| 行业 | 股票数 | 平均ROE | 中位ROE | ROE标准差 |
|-----|-------|--------|--------|----------|
"""
for _, row in roe_by_industry.iterrows():
    report += f"| {row['l1_name']} | {int(row['stock_count'])} | {row['avg_roe']:.2f}% | {row['median_roe']:.2f}% | {row['std_roe']:.2f}% |\n"

report += f"""
### 2.2 行业盈利能力排名

| 行业 | 平均ROE | 平均ROA | 毛利率 | 净利率 |
|-----|--------|--------|-------|-------|
"""
for _, row in profit_ranking.iterrows():
    report += f"| {row['l1_name']} | {row['avg_roe']:.1f}% | {row['avg_roa']:.1f}% | {row['avg_gross_margin']:.1f}% | {row['avg_net_margin']:.1f}% |\n"

report += f"""
### 2.3 行业估值分位数分析

| 行业 | 当前PE | 历史均值 | 20%分位 | 中位数 | 80%分位 | 估值状态 |
|-----|-------|---------|--------|-------|--------|---------|
"""
for _, row in valuation_percentile.iterrows():
    report += f"| {row['industry_name']} | {row['current_pe']:.1f} | {row['avg_pe']:.1f} | {row['pe_20pct']:.1f} | {row['pe_median']:.1f} | {row['pe_80pct']:.1f} | {row['valuation_status']} |\n"

report += f"""

---

## 三、行业收益分析

### 3.1 年度收益率统计

{returns_pivot.to_markdown()}

### 3.2 整体期间表现(2021年以来)

| 行业 | 累计收益 | 年化收益 | 年化波动 | 夏普比率 |
|-----|---------|---------|---------|---------|
"""
for _, row in overall_stats.iterrows():
    report += f"| {row['industry_name']} | {row['total_return_pct']:.1f}% | {row['annualized_return_pct']:.1f}% | {row['annualized_vol_pct']:.1f}% | {row['sharpe_ratio']:.3f} |\n"

report += f"""
### 3.3 最大回撤

| 行业 | 最大回撤 |
|-----|---------|
"""
for _, row in max_drawdown.iterrows():
    report += f"| {row['industry_name']} | {row['max_drawdown_pct']:.1f}% |\n"

report += f"""

---

## 四、行业相关性分析

### 4.1 高相关性行业组合 (相关系数 > 0.7)

| 行业1 | 行业2 | 相关系数 |
|------|------|---------|
"""
for _, row in high_corr_df[high_corr_df['correlation'] > 0.7].head(15).iterrows():
    report += f"| {row['industry1']} | {row['industry2']} | {row['correlation']:.3f} |\n"

report += f"""
### 4.2 低相关性行业组合 (相关系数 < 0.4)

| 行业1 | 行业2 | 相关系数 |
|------|------|---------|
"""
for _, row in high_corr_df[high_corr_df['correlation'] < 0.4].head(15).iterrows():
    report += f"| {row['industry1']} | {row['industry2']} | {row['correlation']:.3f} |\n"

report += f"""
### 4.3 行业聚类分析

**聚类方法**: Ward层次聚类(基于收益率相关性)

"""
for c in sorted(cluster_df['cluster'].unique()):
    industries = cluster_df[cluster_df['cluster'] == c]['industry'].tolist()
    report += f"**类别{c}**: {', '.join(industries)}\n\n"

report += f"""
### 4.4 行业Beta分析(防御性 vs 周期性)

| 行业 | Beta | 属性 |
|-----|------|-----|
"""
for _, row in beta_df.iterrows():
    attr = '防御性' if row['beta'] < 0.9 else ('周期性' if row['beta'] > 1.1 else '中性')
    report += f"| {row['industry']} | {row['beta']:.3f} | {attr} |\n"

report += f"""

---

## 五、行业动量分析

### 5.1 各时间窗口动量

| 行业 | 1周 | 1月 | 3月 | 6月 | 1年 |
|-----|-----|-----|-----|-----|-----|
"""
for _, row in momentum.iterrows():
    report += f"| {row['industry_name']} | {row['mom_1w']:.1f}% | {row['mom_1m']:.1f}% | {row['mom_3m']:.1f}% | {row['mom_6m']:.1f}% | {row['mom_1y']:.1f}% |\n"

report += f"""
### 5.2 近期领涨行业

**近1周领涨**:
"""
for _, row in momentum.sort_values('mom_1w', ascending=False).head(5).iterrows():
    report += f"- {row['industry_name']}: {row['mom_1w']:.1f}%\n"

report += f"""
**近1月领涨**:
"""
for _, row in momentum.sort_values('mom_1m', ascending=False).head(5).iterrows():
    report += f"- {row['industry_name']}: {row['mom_1m']:.1f}%\n"

report += f"""
**近3月领涨**:
"""
for _, row in momentum.sort_values('mom_3m', ascending=False).head(5).iterrows():
    report += f"- {row['industry_name']}: {row['mom_3m']:.1f}%\n"

report += f"""

---

## 六、行业配置建议

### 6.1 综合评分排名

综合考虑估值(PE分位数)、动量(3月涨幅)、盈利能力(ROE)三个维度:

| 行业 | 当前PE | 估值状态 | 3月动量 | ROE | 综合评分 |
|-----|-------|---------|--------|-----|---------|
"""
for _, row in comprehensive[['industry_name', 'current_pe', 'valuation_status', 'mom_3m', 'avg_roe', 'total_score']].head(15).iterrows():
    report += f"| {row['industry_name']} | {row['current_pe']:.1f} | {row['valuation_status']} | {row['mom_3m']:.1f}% | {row['avg_roe']:.1f}% | {row['total_score']:.1f} |\n"

report += f"""
### 6.2 配置策略建议

#### A. 价值策略 - 低估值高ROE行业

"""
if len(recommended) > 0:
    for _, row in recommended.head(5).iterrows():
        report += f"- **{row['industry_name']}**: PE {row['current_pe']:.1f}, ROE {row['avg_roe']:.1f}%\n"
else:
    report += "当前无满足条件的行业\n"

report += f"""
#### B. 动量策略 - 强势行业

"""
for _, row in strong_momentum.iterrows():
    report += f"- **{row['industry_name']}**: 3月涨幅 {row['mom_3m']:.1f}%\n"

report += f"""
#### C. 防御策略 - 低Beta低波动行业

"""
for _, row in defensive.head(5).iterrows():
    report += f"- **{row['industry_name']}**: Beta {row['beta']:.2f}, 年化波动 {row['annualized_vol_pct']:.1f}%\n"

report += f"""
### 6.3 行业分散化组合建议

基于相关性分析,以下行业组合可实现较好的分散化效果:

1. **进取型组合**: 选择高Beta行业,追求超额收益
2. **防御型组合**: 选择低Beta、低波动行业,降低回撤风险
3. **均衡型组合**: 结合价值和动量因子,平衡收益和风险

#### 推荐分散化组合

选择低相关性行业构建组合:
"""
# 找出低相关的行业对
low_corr_industries = set()
for _, row in high_corr_df[high_corr_df['correlation'] < 0.4].head(10).iterrows():
    low_corr_industries.add(row['industry1'])
    low_corr_industries.add(row['industry2'])

for ind in list(low_corr_industries)[:6]:
    report += f"- {ind}\n"

report += f"""

---

## 七、风险提示

1. **数据局限性**: 本报告基于历史数据分析,过去表现不代表未来收益
2. **市场环境变化**: 宏观经济、政策变化可能显著影响行业表现
3. **估值陷阱**: 低估值不等于价值,需结合基本面判断
4. **动量反转**: 强势行业可能面临均值回归风险
5. **行业轮动**: A股行业轮动频繁,需动态调整配置

---

*报告由自动化分析系统生成,仅供参考,不构成投资建议。*
"""

# 保存报告
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n报告已保存至: {REPORT_PATH}")
