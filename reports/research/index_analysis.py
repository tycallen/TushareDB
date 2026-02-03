#!/usr/bin/env python3
"""
A股大盘指数特征研究
分析上证50、沪深300、中证500、中证1000四大指数
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 连接数据库
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
conn = duckdb.connect(DB_PATH, read_only=True)

# 定义四大指数
INDICES = {
    '000016.SH': '上证50',
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000852.SH': '中证1000'
}

print("=" * 80)
print("A股大盘指数特征研究报告")
print("=" * 80)
print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 第一部分：指数对比分析
# ============================================================================
print("\n" + "=" * 80)
print("第一部分：指数对比分析")
print("=" * 80)

# 获取指数日线数据
index_codes = list(INDICES.keys())
query = f"""
SELECT ts_code, trade_date, close, pct_chg, vol, amount
FROM index_daily
WHERE ts_code IN {tuple(index_codes)}
AND trade_date >= '20150101'
ORDER BY ts_code, trade_date
"""
df_daily = conn.execute(query).fetchdf()
df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
df_daily['index_name'] = df_daily['ts_code'].map(INDICES)

# 创建收益率透视表
pivot_close = df_daily.pivot(index='trade_date', columns='ts_code', values='close')
pivot_return = df_daily.pivot(index='trade_date', columns='ts_code', values='pct_chg')

# 1.1 基本信息
print("\n1.1 指数基本信息")
print("-" * 60)
query_basic = f"""
SELECT ts_code, name, base_date, list_date, base_point
FROM index_basic
WHERE ts_code IN {tuple(index_codes)}
"""
df_basic = conn.execute(query_basic).fetchdf()
for _, row in df_basic.iterrows():
    print(f"{row['name']} ({row['ts_code']})")
    print(f"  基期: {row['base_date']}, 上市日期: {row['list_date']}, 基点: {row['base_point']}")

# 1.2 收益风险特征
print("\n1.2 指数收益风险特征 (2015-至今)")
print("-" * 60)

def calculate_metrics(returns):
    """计算收益风险指标"""
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cum_returns = (1 + returns / 100).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()

    # 偏度和峰度
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # 上涨/下跌天数
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()
    win_rate = up_days / (up_days + down_days) * 100

    return {
        '年化收益率': f"{annual_return:.2f}%",
        '年化波动率': f"{annual_vol:.2f}%",
        '夏普比率': f"{sharpe:.3f}",
        '最大回撤': f"{max_drawdown*100:.2f}%",
        '偏度': f"{skewness:.3f}",
        '峰度': f"{kurtosis:.3f}",
        '上涨天数': up_days,
        '下跌天数': down_days,
        '胜率': f"{win_rate:.2f}%"
    }

metrics_list = []
for code in index_codes:
    returns = pivot_return[code].dropna()
    metrics = calculate_metrics(returns)
    metrics['指数'] = INDICES[code]
    metrics_list.append(metrics)

df_metrics = pd.DataFrame(metrics_list)
df_metrics = df_metrics.set_index('指数')
print(df_metrics.to_string())

# 1.3 不同时间段收益对比
print("\n1.3 不同时间段累计收益对比")
print("-" * 60)

periods = {
    '近1年': 252,
    '近3年': 252 * 3,
    '近5年': 252 * 5,
    '近10年': 252 * 10
}

period_returns = {}
for period_name, days in periods.items():
    period_returns[period_name] = {}
    for code in index_codes:
        closes = pivot_close[code].dropna()
        if len(closes) >= days:
            ret = (closes.iloc[-1] / closes.iloc[-days] - 1) * 100
            period_returns[period_name][INDICES[code]] = f"{ret:.2f}%"
        else:
            period_returns[period_name][INDICES[code]] = "N/A"

df_period = pd.DataFrame(period_returns)
print(df_period.to_string())

# 1.4 指数相关性分析
print("\n1.4 指数日收益率相关性矩阵")
print("-" * 60)
corr_matrix = pivot_return.corr()
corr_matrix.columns = [INDICES[c] for c in corr_matrix.columns]
corr_matrix.index = [INDICES[c] for c in corr_matrix.index]
print(corr_matrix.round(4).to_string())

# 1.5 滚动相关性
print("\n1.5 滚动相关性分析 (60日窗口)")
print("-" * 60)
rolling_corr_300_50 = pivot_return['000300.SH'].rolling(60).corr(pivot_return['000016.SH'])
rolling_corr_300_500 = pivot_return['000300.SH'].rolling(60).corr(pivot_return['000905.SH'])
rolling_corr_500_1000 = pivot_return['000905.SH'].rolling(60).corr(pivot_return['000852.SH'])

print(f"沪深300与上证50相关性 - 均值: {rolling_corr_300_50.mean():.4f}, 最小: {rolling_corr_300_50.min():.4f}, 最大: {rolling_corr_300_50.max():.4f}")
print(f"沪深300与中证500相关性 - 均值: {rolling_corr_300_500.mean():.4f}, 最小: {rolling_corr_300_500.min():.4f}, 最大: {rolling_corr_300_500.max():.4f}")
print(f"中证500与中证1000相关性 - 均值: {rolling_corr_500_1000.mean():.4f}, 最小: {rolling_corr_500_1000.min():.4f}, 最大: {rolling_corr_500_1000.max():.4f}")

# 1.6 年度收益对比
print("\n1.6 年度收益对比")
print("-" * 60)
df_daily['year'] = df_daily['trade_date'].dt.year
yearly_first = df_daily.groupby(['ts_code', 'year']).first().reset_index()
yearly_last = df_daily.groupby(['ts_code', 'year']).last().reset_index()

yearly_returns = yearly_last.merge(yearly_first, on=['ts_code', 'year'], suffixes=('_end', '_start'))
yearly_returns['annual_return'] = (yearly_returns['close_end'] / yearly_returns['close_start'] - 1) * 100
yearly_returns['index_name'] = yearly_returns['ts_code'].map(INDICES)

yearly_pivot = yearly_returns.pivot(index='year', columns='index_name', values='annual_return')
yearly_pivot = yearly_pivot[['上证50', '沪深300', '中证500', '中证1000']]
print(yearly_pivot.round(2).to_string())

# ============================================================================
# 第二部分：指数成分分析
# ============================================================================
print("\n\n" + "=" * 80)
print("第二部分：指数成分分析")
print("=" * 80)

# 获取最新权重数据
print("\n2.1 获取最新成分股及权重")
print("-" * 60)

component_stats = {}
for code in index_codes:
    query_weight = f"""
    SELECT iw.index_code, iw.con_code, iw.weight, iw.trade_date,
           sb.name, sb.industry
    FROM index_weight iw
    LEFT JOIN stock_basic sb ON iw.con_code = sb.ts_code
    WHERE iw.index_code = '{code}'
    AND iw.trade_date = (
        SELECT MAX(trade_date) FROM index_weight WHERE index_code = '{code}'
    )
    ORDER BY iw.weight DESC
    """
    df_weight = conn.execute(query_weight).fetchdf()

    if len(df_weight) > 0:
        latest_date = df_weight['trade_date'].iloc[0]
        total_stocks = len(df_weight)
        total_weight = df_weight['weight'].sum()

        component_stats[INDICES[code]] = {
            '最新权重日期': latest_date,
            '成分股数量': total_stocks,
            '总权重': f"{total_weight:.2f}%",
            '前10权重': f"{df_weight['weight'].head(10).sum():.2f}%",
            '前20权重': f"{df_weight['weight'].head(20).sum():.2f}%"
        }

        print(f"\n{INDICES[code]} (截至{latest_date})")
        print(f"  成分股数量: {total_stocks}")
        print(f"  前10大权重股权重合计: {df_weight['weight'].head(10).sum():.2f}%")
        print(f"  前10大权重股:")
        for i, (_, row) in enumerate(df_weight.head(10).iterrows(), 1):
            print(f"    {i}. {row['name']} ({row['con_code']}): {row['weight']:.2f}% - {row['industry']}")

# 2.2 行业分布分析
print("\n2.2 行业分布分析")
print("-" * 60)

for code in index_codes:
    query_weight = f"""
    SELECT iw.index_code, iw.con_code, iw.weight,
           sb.industry
    FROM index_weight iw
    LEFT JOIN stock_basic sb ON iw.con_code = sb.ts_code
    WHERE iw.index_code = '{code}'
    AND iw.trade_date = (
        SELECT MAX(trade_date) FROM index_weight WHERE index_code = '{code}'
    )
    """
    df_weight = conn.execute(query_weight).fetchdf()

    if len(df_weight) > 0:
        # 按行业汇总
        industry_dist = df_weight.groupby('industry').agg({
            'weight': 'sum',
            'con_code': 'count'
        }).rename(columns={'con_code': 'count'}).sort_values('weight', ascending=False)

        print(f"\n{INDICES[code]} 行业分布 (前10):")
        for i, (industry, row) in enumerate(industry_dist.head(10).iterrows(), 1):
            industry_name = industry if industry else '未知'
            print(f"  {i:2d}. {industry_name:12s}: 权重 {row['weight']:6.2f}%, 成分股 {int(row['count']):3d} 只")

# 2.3 市值分布分析
print("\n2.3 市值分布分析")
print("-" * 60)

# 获取最新交易日
latest_trade_date = conn.execute("SELECT MAX(trade_date) FROM daily_basic").fetchone()[0]
print(f"市值数据日期: {latest_trade_date}")

for code in index_codes:
    query_mv = f"""
    SELECT iw.con_code, iw.weight, db.total_mv, db.circ_mv, sb.name
    FROM index_weight iw
    LEFT JOIN daily_basic db ON iw.con_code = db.ts_code
    LEFT JOIN stock_basic sb ON iw.con_code = sb.ts_code
    WHERE iw.index_code = '{code}'
    AND iw.trade_date = (
        SELECT MAX(trade_date) FROM index_weight WHERE index_code = '{code}'
    )
    AND db.trade_date = '{latest_trade_date}'
    """
    df_mv = conn.execute(query_mv).fetchdf()

    if len(df_mv) > 0:
        df_mv['total_mv'] = df_mv['total_mv'] / 10000  # 转换为亿元

        # 市值分布统计
        mv_stats = df_mv['total_mv'].describe()

        # 市值分组
        bins = [0, 100, 500, 1000, 5000, float('inf')]
        labels = ['<100亿', '100-500亿', '500-1000亿', '1000-5000亿', '>5000亿']
        df_mv['mv_group'] = pd.cut(df_mv['total_mv'], bins=bins, labels=labels)
        mv_dist = df_mv.groupby('mv_group', observed=True).agg({
            'con_code': 'count',
            'weight': 'sum'
        }).rename(columns={'con_code': 'count'})

        print(f"\n{INDICES[code]} 市值分布:")
        print(f"  平均市值: {mv_stats['mean']:.2f}亿元")
        print(f"  中位数市值: {mv_stats['50%']:.2f}亿元")
        print(f"  最大市值: {mv_stats['max']:.2f}亿元")
        print(f"  最小市值: {mv_stats['min']:.2f}亿元")
        print(f"  市值分组:")
        for group in labels:
            if group in mv_dist.index:
                row = mv_dist.loc[group]
                print(f"    {group:12s}: {int(row['count']):3d} 只, 权重 {row['weight']:6.2f}%")

# 2.4 权重集中度分析
print("\n2.4 权重集中度分析")
print("-" * 60)

concentration_data = []
for code in index_codes:
    query_weight = f"""
    SELECT weight
    FROM index_weight
    WHERE index_code = '{code}'
    AND trade_date = (
        SELECT MAX(trade_date) FROM index_weight WHERE index_code = '{code}'
    )
    ORDER BY weight DESC
    """
    weights = conn.execute(query_weight).fetchdf()['weight'].values

    if len(weights) > 0:
        # 计算HHI指数 (赫芬达尔指数)
        hhi = np.sum((weights / 100) ** 2) * 10000

        # 计算CR指标
        cr5 = weights[:5].sum()
        cr10 = weights[:10].sum()
        cr20 = weights[:20].sum()

        concentration_data.append({
            '指数': INDICES[code],
            'HHI指数': f"{hhi:.2f}",
            'CR5': f"{cr5:.2f}%",
            'CR10': f"{cr10:.2f}%",
            'CR20': f"{cr20:.2f}%",
            '最大权重': f"{weights[0]:.2f}%",
            '最小权重': f"{weights[-1]:.4f}%"
        })

df_concentration = pd.DataFrame(concentration_data).set_index('指数')
print(df_concentration.to_string())

# ============================================================================
# 第三部分：投资建议
# ============================================================================
print("\n\n" + "=" * 80)
print("第三部分：投资建议")
print("=" * 80)

print("""
3.1 四大指数特征总结
------------------------------------------------------------
指数名称    风格特征          成分股特点                适合场景
------------------------------------------------------------
上证50      超大盘价值       蓝筹龙头、金融占比高       保守型配置、追求稳定
沪深300     大盘均衡         市场核心、行业分散         基准配置、核心仓位
中证500     中盘成长         新兴产业、弹性较大         进取型配置、追求超额
中证1000    小盘成长         小市值、高波动             激进型配置、博取高收益
------------------------------------------------------------

3.2 不同市场环境下的指数选择
------------------------------------------------------------
市场环境              推荐指数              理由
------------------------------------------------------------
牛市初期              中证500/中证1000      小盘股弹性大，涨幅领先
牛市中期              沪深300               市场关注度转向权重股
牛市末期              上证50                避险情绪，资金抱团龙头
熊市初期              上证50                防御属性强，回撤小
熊市末期              中证1000              超跌反弹弹性大
震荡市                沪深300               均衡配置，控制波动
风格轮动期            均衡配置              大小盘轮动频繁
------------------------------------------------------------

3.3 指数增强策略方向
------------------------------------------------------------
指数名称    增强方向                          预期超额收益来源
------------------------------------------------------------
上证50      价值因子、低波动因子              估值回归、股息收益
            基本面选股、打新增厚              财务质量筛选

沪深300     多因子综合                        因子暴露优化
            行业轮动、择时增强                行业配置偏离

中证500     成长因子、动量因子                成长股挖掘
            量化选股、事件驱动                信息不对称

中证1000    动量反转、流动性因子              市场微观结构
            小市值因子、壳价值                重组概念
------------------------------------------------------------

3.4 配置建议
------------------------------------------------------------
投资者类型        上证50    沪深300    中证500    中证1000
------------------------------------------------------------
保守型            40%       40%        15%        5%
稳健型            25%       35%        25%        15%
进取型            15%       25%        30%        30%
激进型            10%       20%        30%        40%
------------------------------------------------------------

3.5 风险提示
------------------------------------------------------------
1. 上证50: 金融行业占比过高，对政策敏感度大
2. 沪深300: 与上证50重叠度高，分散效果有限
3. 中证500: 波动较大，回撤控制是关键
4. 中证1000: 流动性风险，大资金难以操作
5. 历史业绩不代表未来表现，市场风格轮动难以预测
------------------------------------------------------------

3.6 指数投资工具选择
------------------------------------------------------------
类型            特点                    代表产品
------------------------------------------------------------
ETF             流动性好、成本低        各指数对应ETF
指数基金        申赎灵活                场外指数基金
增强型基金      追求超额收益            量化增强产品
指数期货        杠杆工具、对冲          IF/IC/IM期货
期权            策略灵活                上证50ETF期权等
------------------------------------------------------------
""")

# ============================================================================
# 附录：详细数据
# ============================================================================
print("\n" + "=" * 80)
print("附录：详细统计数据")
print("=" * 80)

# 月度收益统计
print("\n附录1: 月度收益统计 (2020-至今)")
print("-" * 60)
df_recent = df_daily[df_daily['trade_date'] >= '2020-01-01'].copy()
df_recent['month'] = df_recent['trade_date'].dt.to_period('M')
monthly_first = df_recent.groupby(['ts_code', 'month']).first().reset_index()
monthly_last = df_recent.groupby(['ts_code', 'month']).last().reset_index()
monthly_returns = monthly_last.merge(monthly_first, on=['ts_code', 'month'], suffixes=('_end', '_start'))
monthly_returns['monthly_return'] = (monthly_returns['close_end'] / monthly_returns['close_start'] - 1) * 100

for code in index_codes:
    mr = monthly_returns[monthly_returns['ts_code'] == code]['monthly_return']
    print(f"{INDICES[code]}: 月均收益 {mr.mean():.2f}%, 月收益标准差 {mr.std():.2f}%, 正收益月占比 {(mr > 0).mean()*100:.1f}%")

# 极端收益日统计
print("\n附录2: 极端收益日统计 (2015-至今)")
print("-" * 60)
for code in index_codes:
    returns = pivot_return[code].dropna()
    extreme_up = (returns > 3).sum()
    extreme_down = (returns < -3).sum()
    print(f"{INDICES[code]}: 涨幅>3%天数 {extreme_up}, 跌幅>3%天数 {extreme_down}")

print("\n" + "=" * 80)
print("报告完成")
print("=" * 80)

conn.close()
