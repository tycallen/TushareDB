#!/usr/bin/env python3
"""
成长股特征研究报告

研究内容:
1. 成长股筛选 - 高营收增长、高利润增长、高ROE
2. 成长股特征 - 行业分布、估值特征、收益风险特征
3. 投资策略 - GARP策略、PEG选股、成长陷阱规避

Author: AI Research
Date: 2024
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def get_growth_stocks_data():
    """获取成长股筛选所需数据"""
    conn = get_connection()

    # 获取最近3年的年报数据用于计算成长性
    query = """
    WITH latest_stocks AS (
        -- 获取当前上市的股票
        SELECT ts_code, name, industry
        FROM stock_basic
        WHERE list_status = 'L'
    ),
    fina_data AS (
        -- 获取最近3年的财务数据
        SELECT
            f.ts_code,
            f.end_date,
            f.tr_yoy,           -- 营收同比增长率
            f.netprofit_yoy,    -- 净利润同比增长率
            f.or_yoy,           -- 营业收入同比
            f.roe,              -- ROE
            f.roe_yearly,       -- 年化ROE
            f.roe_waa,          -- 加权ROE
            f.grossprofit_margin, -- 毛利率
            f.netprofit_margin,   -- 净利率
            f.basic_eps_yoy,      -- EPS同比
            f.ocf_yoy,            -- 经营现金流同比
            f.debt_to_assets,     -- 资产负债率
            f.current_ratio       -- 流动比率
        FROM fina_indicator_vip f
        WHERE f.end_date IN ('20231231', '20221231', '20211231', '20201231')
        AND f.end_date LIKE '%1231'  -- 只取年报
    )
    SELECT
        s.ts_code,
        s.name,
        s.industry,
        f.end_date,
        f.tr_yoy,
        f.netprofit_yoy,
        f.or_yoy,
        f.roe,
        f.roe_yearly,
        f.roe_waa,
        f.grossprofit_margin,
        f.netprofit_margin,
        f.basic_eps_yoy,
        f.ocf_yoy,
        f.debt_to_assets,
        f.current_ratio
    FROM latest_stocks s
    JOIN fina_data f ON s.ts_code = f.ts_code
    ORDER BY s.ts_code, f.end_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def get_valuation_data():
    """获取估值数据"""
    conn = get_connection()

    query = """
    SELECT
        ts_code,
        trade_date,
        close,
        pe_ttm,
        pb,
        ps_ttm,
        total_mv,
        circ_mv
    FROM daily_basic
    WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic)
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def get_price_data(stock_list=None):
    """获取价格数据用于计算收益"""
    conn = get_connection()

    # 只获取指定股票列表的数据以减少内存占用
    if stock_list is not None:
        stock_filter = "AND ts_code IN ('" + "','".join(stock_list) + "')"
    else:
        stock_filter = ""

    query = f"""
    SELECT
        ts_code,
        trade_date,
        close,
        pct_chg
    FROM daily
    WHERE trade_date >= '20240101'
    {stock_filter}
    ORDER BY ts_code, trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def get_industry_classification():
    """获取行业分类数据"""
    conn = get_connection()

    query = """
    SELECT
        ts_code,
        l1_name as sw_l1,
        l2_name as sw_l2,
        l3_name as sw_l3
    FROM index_member_all
    WHERE is_new = 'Y' OR out_date IS NULL
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def screen_growth_stocks(fina_df):
    """
    成长股筛选
    标准:
    1. 连续3年营收增长 > 15%
    2. 连续3年净利润增长 > 15%
    3. 连续3年ROE > 10%
    """

    # 按股票和年份聚合
    pivot_df = fina_df.pivot_table(
        index=['ts_code', 'name', 'industry'],
        columns='end_date',
        values=['tr_yoy', 'netprofit_yoy', 'roe_yearly'],
        aggfunc='first'
    ).reset_index()

    # 扁平化列名
    pivot_df.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
                        for col in pivot_df.columns]

    # 筛选高成长股票
    growth_stocks = pivot_df.copy()

    # 条件1: 3年营收增长都大于15%
    rev_cols = [c for c in growth_stocks.columns if 'tr_yoy' in c]
    if len(rev_cols) >= 3:
        growth_stocks['avg_rev_growth'] = growth_stocks[rev_cols[-3:]].mean(axis=1)
        growth_stocks['min_rev_growth'] = growth_stocks[rev_cols[-3:]].min(axis=1)

    # 条件2: 3年利润增长都大于15%
    profit_cols = [c for c in growth_stocks.columns if 'netprofit_yoy' in c]
    if len(profit_cols) >= 3:
        growth_stocks['avg_profit_growth'] = growth_stocks[profit_cols[-3:]].mean(axis=1)
        growth_stocks['min_profit_growth'] = growth_stocks[profit_cols[-3:]].min(axis=1)

    # 条件3: 3年ROE都大于10%
    roe_cols = [c for c in growth_stocks.columns if 'roe_yearly' in c]
    if len(roe_cols) >= 3:
        growth_stocks['avg_roe'] = growth_stocks[roe_cols[-3:]].mean(axis=1)
        growth_stocks['min_roe'] = growth_stocks[roe_cols[-3:]].min(axis=1)

    # 应用筛选条件
    strict_growth = growth_stocks[
        (growth_stocks['min_rev_growth'] > 15) &
        (growth_stocks['min_profit_growth'] > 15) &
        (growth_stocks['min_roe'] > 10)
    ].copy()

    # 宽松条件 - 平均值标准
    moderate_growth = growth_stocks[
        (growth_stocks['avg_rev_growth'] > 15) &
        (growth_stocks['avg_profit_growth'] > 15) &
        (growth_stocks['avg_roe'] > 10)
    ].copy()

    # 基础成长股 - 至少1年达标
    basic_growth = growth_stocks[
        (growth_stocks['avg_rev_growth'] > 10) &
        (growth_stocks['avg_profit_growth'] > 10) &
        (growth_stocks['avg_roe'] > 8)
    ].copy()

    return {
        'all_data': growth_stocks,
        'strict_growth': strict_growth,
        'moderate_growth': moderate_growth,
        'basic_growth': basic_growth
    }

def analyze_industry_distribution(growth_df, industry_df):
    """分析成长股的行业分布"""
    # 合并行业分类
    merged = growth_df.merge(industry_df, on='ts_code', how='left')

    # 行业分布统计
    industry_stats = merged.groupby('sw_l1').agg({
        'ts_code': 'count',
        'avg_rev_growth': 'mean',
        'avg_profit_growth': 'mean',
        'avg_roe': 'mean'
    }).rename(columns={'ts_code': 'count'}).sort_values('count', ascending=False)

    return industry_stats

def analyze_valuation(growth_stocks, valuation_df):
    """分析成长股的估值特征"""
    # 合并估值数据
    merged = growth_stocks.merge(valuation_df, on='ts_code', how='inner')

    # 过滤异常值
    merged = merged[
        (merged['pe_ttm'] > 0) &
        (merged['pe_ttm'] < 500) &
        (merged['pb'] > 0) &
        (merged['pb'] < 50)
    ]

    # 计算PEG (PE / 利润增速)
    merged['peg'] = merged['pe_ttm'] / merged['avg_profit_growth']
    merged = merged[merged['peg'] > 0]

    valuation_stats = {
        'pe_ttm': {
            'mean': merged['pe_ttm'].mean(),
            'median': merged['pe_ttm'].median(),
            'std': merged['pe_ttm'].std(),
            'q25': merged['pe_ttm'].quantile(0.25),
            'q75': merged['pe_ttm'].quantile(0.75)
        },
        'pb': {
            'mean': merged['pb'].mean(),
            'median': merged['pb'].median(),
            'std': merged['pb'].std(),
            'q25': merged['pb'].quantile(0.25),
            'q75': merged['pb'].quantile(0.75)
        },
        'peg': {
            'mean': merged['peg'].mean(),
            'median': merged['peg'].median(),
            'std': merged['peg'].std(),
            'q25': merged['peg'].quantile(0.25),
            'q75': merged['peg'].quantile(0.75)
        }
    }

    return merged, valuation_stats

def analyze_risk_return(growth_stocks, price_df):
    """分析成长股的收益风险特征"""
    # 计算每只股票的年化收益和波动率
    results = []

    for ts_code in growth_stocks['ts_code'].unique():
        stock_prices = price_df[price_df['ts_code'] == ts_code].copy()
        if len(stock_prices) < 20:
            continue

        stock_prices = stock_prices.sort_values('trade_date')
        returns = stock_prices['pct_chg'] / 100

        # 年化收益
        total_return = (1 + returns).prod() - 1
        trading_days = len(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率 (假设无风险利率3%)
        sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        results.append({
            'ts_code': ts_code,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        })

    return pd.DataFrame(results)

def garp_strategy(growth_stocks, valuation_df):
    """
    GARP策略 (Growth at Reasonable Price)
    选择具有合理估值的成长股
    """
    merged = growth_stocks.merge(valuation_df, on='ts_code', how='inner')

    # 过滤有效数据
    merged = merged[
        (merged['pe_ttm'] > 0) &
        (merged['pe_ttm'] < 100) &
        (merged['avg_profit_growth'] > 0)
    ]

    # 计算PEG
    merged['peg'] = merged['pe_ttm'] / merged['avg_profit_growth']

    # GARP筛选条件:
    # 1. PEG < 1 (合理估值)
    # 2. ROE > 15% (高盈利能力)
    # 3. 利润增速 > 20%
    garp_stocks = merged[
        (merged['peg'] > 0) &
        (merged['peg'] < 1) &
        (merged['avg_roe'] > 15) &
        (merged['avg_profit_growth'] > 20)
    ].sort_values('peg')

    return garp_stocks

def peg_selection(growth_stocks, valuation_df):
    """
    PEG选股策略
    PEG < 1: 被低估
    PEG = 1: 合理估值
    PEG > 1: 被高估
    """
    merged = growth_stocks.merge(valuation_df, on='ts_code', how='inner')

    # 过滤有效数据
    merged = merged[
        (merged['pe_ttm'] > 0) &
        (merged['pe_ttm'] < 200) &
        (merged['avg_profit_growth'] > 0)
    ]

    merged['peg'] = merged['pe_ttm'] / merged['avg_profit_growth']

    # PEG分组
    merged['peg_category'] = pd.cut(
        merged['peg'],
        bins=[-np.inf, 0.5, 1, 1.5, 2, np.inf],
        labels=['极低估(PEG<0.5)', '低估(0.5-1)', '合理(1-1.5)', '高估(1.5-2)', '极高估(>2)']
    )

    return merged

def identify_growth_traps(fina_df, growth_stocks):
    """
    识别成长陷阱
    特征:
    1. 利润增长但现金流恶化
    2. 营收增长但毛利率下降
    3. ROE下滑
    4. 高负债扩张
    """
    # 获取最新两年的数据对比
    recent_data = fina_df[fina_df['end_date'].isin(['20231231', '20221231'])].copy()

    pivot = recent_data.pivot_table(
        index=['ts_code', 'name'],
        columns='end_date',
        values=['grossprofit_margin', 'netprofit_margin', 'roe_yearly',
                'ocf_yoy', 'debt_to_assets'],
        aggfunc='first'
    ).reset_index()

    pivot.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
                     for col in pivot.columns]

    # 合并成长股数据
    merged = growth_stocks[['ts_code', 'avg_profit_growth', 'avg_rev_growth']].merge(
        pivot, on='ts_code', how='inner'
    )

    traps = []

    # 陷阱1: 利润增长但现金流恶化
    if 'ocf_yoy_20231231' in merged.columns and 'ocf_yoy_20221231' in merged.columns:
        trap1 = merged[
            (merged['avg_profit_growth'] > 20) &
            (merged['ocf_yoy_20231231'] < -20)
        ]['ts_code'].tolist()
        traps.append(('现金流陷阱', trap1))

    # 陷阱2: 毛利率下降
    gm_2023 = 'grossprofit_margin_20231231'
    gm_2022 = 'grossprofit_margin_20221231'
    if gm_2023 in merged.columns and gm_2022 in merged.columns:
        merged['gm_change'] = merged[gm_2023] - merged[gm_2022]
        trap2 = merged[
            (merged['avg_rev_growth'] > 20) &
            (merged['gm_change'] < -3)
        ]['ts_code'].tolist()
        traps.append(('毛利率陷阱', trap2))

    # 陷阱3: ROE下滑
    roe_2023 = 'roe_yearly_20231231'
    roe_2022 = 'roe_yearly_20221231'
    if roe_2023 in merged.columns and roe_2022 in merged.columns:
        merged['roe_change'] = merged[roe_2023] - merged[roe_2022]
        trap3 = merged[
            (merged['avg_profit_growth'] > 15) &
            (merged['roe_change'] < -5)
        ]['ts_code'].tolist()
        traps.append(('ROE陷阱', trap3))

    # 陷阱4: 高负债扩张
    debt_col = 'debt_to_assets_20231231'
    if debt_col in merged.columns:
        trap4 = merged[
            (merged['avg_rev_growth'] > 30) &
            (merged[debt_col] > 70)
        ]['ts_code'].tolist()
        traps.append(('高负债陷阱', trap4))

    return traps, merged

def generate_report():
    """生成完整研究报告"""
    print("=" * 60)
    print("成长股特征研究报告")
    print("=" * 60)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 数据准备
    print("正在获取数据...")
    fina_df = get_growth_stocks_data()
    valuation_df = get_valuation_data()
    industry_df = get_industry_classification()

    print(f"财务数据: {len(fina_df)} 条记录")
    print(f"估值数据: {len(valuation_df)} 条记录")
    print(f"行业数据: {len(industry_df)} 条记录")
    print()

    # 2. 成长股筛选
    print("=" * 60)
    print("一、成长股筛选")
    print("=" * 60)

    screening_results = screen_growth_stocks(fina_df)

    print(f"\n筛选标准说明:")
    print(f"  严格标准: 连续3年营收增长>15%, 利润增长>15%, ROE>10%")
    print(f"  中等标准: 3年平均营收增长>15%, 利润增长>15%, ROE>10%")
    print(f"  基础标准: 3年平均营收增长>10%, 利润增长>10%, ROE>8%")

    print(f"\n筛选结果:")
    print(f"  严格成长股数量: {len(screening_results['strict_growth'])}")
    print(f"  中等成长股数量: {len(screening_results['moderate_growth'])}")
    print(f"  基础成长股数量: {len(screening_results['basic_growth'])}")

    # 显示严格成长股TOP20
    if len(screening_results['strict_growth']) > 0:
        top_growth = screening_results['strict_growth'].nlargest(20, 'avg_profit_growth')
        print(f"\n严格成长股TOP20 (按利润增速排序):")
        print("-" * 80)
        for _, row in top_growth.iterrows():
            print(f"  {row['ts_code']} {row['name'][:6]:6s} | "
                  f"营收增长:{row['avg_rev_growth']:6.1f}% | "
                  f"利润增长:{row['avg_profit_growth']:6.1f}% | "
                  f"ROE:{row['avg_roe']:5.1f}%")

    # 3. 行业分布分析
    print("\n" + "=" * 60)
    print("二、成长股行业分布")
    print("=" * 60)

    growth_for_analysis = screening_results['moderate_growth'] if len(screening_results['moderate_growth']) > 50 else screening_results['basic_growth']

    if len(growth_for_analysis) > 0:
        industry_stats = analyze_industry_distribution(growth_for_analysis, industry_df)
        print(f"\n成长股行业分布 (共{len(growth_for_analysis)}只):")
        print("-" * 80)
        print(f"{'行业':<15} | {'数量':>6} | {'平均营收增长':>12} | {'平均利润增长':>12} | {'平均ROE':>10}")
        print("-" * 80)
        for idx, row in industry_stats.head(15).iterrows():
            if pd.notna(idx):
                print(f"{str(idx)[:14]:<15} | {row['count']:>6.0f} | {row['avg_rev_growth']:>12.1f}% | {row['avg_profit_growth']:>12.1f}% | {row['avg_roe']:>9.1f}%")

    # 4. 估值特征分析
    print("\n" + "=" * 60)
    print("三、成长股估值特征")
    print("=" * 60)

    if len(growth_for_analysis) > 0:
        valuation_merged, val_stats = analyze_valuation(growth_for_analysis, valuation_df)

        print(f"\n成长股估值统计 (有效样本: {len(valuation_merged)}只):")
        print("-" * 60)
        for metric, stats in val_stats.items():
            print(f"\n{metric.upper()}:")
            print(f"  均值: {stats['mean']:.2f}")
            print(f"  中位数: {stats['median']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  25分位: {stats['q25']:.2f}")
            print(f"  75分位: {stats['q75']:.2f}")

        # PEG分布
        if 'peg' in valuation_merged.columns:
            print(f"\nPEG分布:")
            peg_dist = valuation_merged['peg'].describe()
            print(f"  PEG < 0.5: {len(valuation_merged[valuation_merged['peg'] < 0.5])} 只 (极度低估)")
            print(f"  0.5 <= PEG < 1: {len(valuation_merged[(valuation_merged['peg'] >= 0.5) & (valuation_merged['peg'] < 1)])} 只 (低估)")
            print(f"  1 <= PEG < 1.5: {len(valuation_merged[(valuation_merged['peg'] >= 1) & (valuation_merged['peg'] < 1.5)])} 只 (合理)")
            print(f"  PEG >= 1.5: {len(valuation_merged[valuation_merged['peg'] >= 1.5])} 只 (高估)")

    # 5. 收益风险特征
    print("\n" + "=" * 60)
    print("四、成长股收益风险特征")
    print("=" * 60)

    risk_return = pd.DataFrame()  # 初始化
    if len(growth_for_analysis) > 0:
        # 只获取成长股的价格数据以节省内存
        stock_list = growth_for_analysis['ts_code'].unique().tolist()
        print(f"正在获取{len(stock_list)}只成长股的价格数据...")
        price_df = get_price_data(stock_list)
        print(f"价格数据: {len(price_df)} 条记录")
        risk_return = analyze_risk_return(growth_for_analysis, price_df)

        if len(risk_return) > 0:
            print(f"\n收益风险统计 (有效样本: {len(risk_return)}只):")
            print("-" * 60)
            print(f"\n年化收益率:")
            print(f"  均值: {risk_return['annual_return'].mean()*100:.1f}%")
            print(f"  中位数: {risk_return['annual_return'].median()*100:.1f}%")
            print(f"  正收益比例: {(risk_return['annual_return']>0).mean()*100:.1f}%")

            print(f"\n年化波动率:")
            print(f"  均值: {risk_return['annual_vol'].mean()*100:.1f}%")
            print(f"  中位数: {risk_return['annual_vol'].median()*100:.1f}%")

            print(f"\n夏普比率:")
            print(f"  均值: {risk_return['sharpe'].mean():.2f}")
            print(f"  中位数: {risk_return['sharpe'].median():.2f}")
            print(f"  夏普>1比例: {(risk_return['sharpe']>1).mean()*100:.1f}%")

            print(f"\n最大回撤:")
            print(f"  均值: {risk_return['max_drawdown'].mean()*100:.1f}%")
            print(f"  中位数: {risk_return['max_drawdown'].median()*100:.1f}%")

    # 6. GARP策略
    print("\n" + "=" * 60)
    print("五、GARP策略选股")
    print("=" * 60)

    if len(growth_for_analysis) > 0:
        garp_stocks = garp_strategy(growth_for_analysis, valuation_df)

        print(f"\nGARP策略条件:")
        print(f"  1. PEG < 1 (合理估值)")
        print(f"  2. ROE > 15% (高盈利能力)")
        print(f"  3. 利润增速 > 20%")

        print(f"\n符合GARP策略股票: {len(garp_stocks)}只")

        if len(garp_stocks) > 0:
            print(f"\nGARP精选股票TOP15:")
            print("-" * 100)
            print(f"{'代码':<12} | {'名称':<8} | {'PEG':>6} | {'PE':>8} | {'利润增速':>10} | {'ROE':>8} | {'市值(亿)':>10}")
            print("-" * 100)
            for _, row in garp_stocks.head(15).iterrows():
                mv = row['total_mv'] / 10000 if pd.notna(row['total_mv']) else 0
                print(f"{row['ts_code']:<12} | {str(row['name'])[:8]:<8} | {row['peg']:>6.2f} | {row['pe_ttm']:>8.1f} | {row['avg_profit_growth']:>9.1f}% | {row['avg_roe']:>7.1f}% | {mv:>10.0f}")

    # 7. PEG选股
    print("\n" + "=" * 60)
    print("六、PEG选股分析")
    print("=" * 60)

    if len(growth_for_analysis) > 0:
        peg_analysis = peg_selection(growth_for_analysis, valuation_df)

        print(f"\nPEG选股说明:")
        print(f"  PEG = PE / 利润增速")
        print(f"  PEG < 1: 被低估 (彼得林奇推荐)")
        print(f"  PEG = 1: 合理估值")
        print(f"  PEG > 1: 可能被高估")

        if 'peg_category' in peg_analysis.columns:
            print(f"\nPEG分布统计:")
            peg_dist = peg_analysis['peg_category'].value_counts()
            for cat, cnt in peg_dist.items():
                print(f"  {cat}: {cnt}只")

        # 低PEG股票
        low_peg = peg_analysis[peg_analysis['peg'] < 1].nsmallest(15, 'peg')
        if len(low_peg) > 0:
            print(f"\n低PEG成长股TOP15 (PEG<1):")
            print("-" * 100)
            print(f"{'代码':<12} | {'名称':<8} | {'PEG':>6} | {'PE':>8} | {'利润增速':>10} | {'ROE':>8}")
            print("-" * 100)
            for _, row in low_peg.iterrows():
                print(f"{row['ts_code']:<12} | {str(row['name'])[:8]:<8} | {row['peg']:>6.2f} | {row['pe_ttm']:>8.1f} | {row['avg_profit_growth']:>9.1f}% | {row['avg_roe']:>7.1f}%")

    # 8. 成长陷阱识别
    print("\n" + "=" * 60)
    print("七、成长陷阱识别")
    print("=" * 60)

    if len(growth_for_analysis) > 0:
        traps, trap_data = identify_growth_traps(fina_df, growth_for_analysis)

        print(f"\n成长陷阱类型说明:")
        print(f"  1. 现金流陷阱: 利润增长但经营现金流大幅下降")
        print(f"  2. 毛利率陷阱: 营收增长但毛利率持续下滑")
        print(f"  3. ROE陷阱: 利润增长但ROE下降(靠杠杆或资产膨胀)")
        print(f"  4. 高负债陷阱: 高速扩张伴随高负债率")

        print(f"\n陷阱识别结果:")
        all_trap_stocks = set()
        for trap_name, trap_stocks in traps:
            print(f"  {trap_name}: {len(trap_stocks)}只股票")
            all_trap_stocks.update(trap_stocks)

        print(f"\n总计可能存在陷阱的股票: {len(all_trap_stocks)}只")
        print(f"建议规避比例: {len(all_trap_stocks)/len(growth_for_analysis)*100:.1f}%")

    # 9. 投资建议总结
    print("\n" + "=" * 60)
    print("八、投资建议总结")
    print("=" * 60)

    print(f"""
成长股投资策略建议:

1. 筛选标准建议
   - 优先选择连续3年保持高增长的严格成长股
   - ROE>15%是优质成长股的重要标志
   - 关注营收和利润同步增长的公司

2. 估值策略
   - 使用PEG估值法,优选PEG<1的成长股
   - GARP策略结合成长性和估值,风险收益比更优
   - 避免追逐高PE的热门成长股

3. 风险控制
   - 警惕四大成长陷阱: 现金流、毛利率、ROE、高负债
   - 分散投资于多个成长行业
   - 定期复查成长股的财务质量

4. 行业配置
   - 关注成长股集中的优势行业
   - 新兴产业往往成长股更多
   - 传统行业的成长股可能更稳健

5. 组合构建
   - 核心仓位: GARP策略精选股
   - 卫星仓位: 高成长但估值较高的潜力股
   - 定期再平衡,保持组合成长属性
    """)

    return {
        'screening_results': screening_results,
        'growth_for_analysis': growth_for_analysis,
        'valuation_merged': valuation_merged if len(growth_for_analysis) > 0 else None,
        'garp_stocks': garp_stocks if len(growth_for_analysis) > 0 else None,
        'risk_return': risk_return if len(risk_return) > 0 else None
    }

def save_report_to_file():
    """保存报告到文件"""
    import sys
    from io import StringIO

    # 捕获print输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    results = generate_report()

    report_content = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # 打印到控制台
    print(report_content)

    # 保存到文件
    report_file = f"{REPORT_PATH}/growth_stock_research_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\n报告已保存至: {report_file}")

    # 保存CSV数据
    if results.get('garp_stocks') is not None and len(results['garp_stocks']) > 0:
        garp_file = f"{REPORT_PATH}/garp_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
        results['garp_stocks'].to_csv(garp_file, index=False, encoding='utf-8-sig')
        print(f"GARP股票列表已保存至: {garp_file}")

    return results

if __name__ == '__main__':
    save_report_to_file()
