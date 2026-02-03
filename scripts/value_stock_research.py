#!/usr/bin/env python3
"""
主板价值股特征研究

研究内容:
1. 价值股筛选: 低PE/低PB、高股息、稳定盈利
2. 价值股特征: 行业分布、收益风险特征、与成长股对比
3. 投资策略: 深度价值策略、价值陷阱规避、配置建议

作者: AI Research
日期: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def get_connection():
    """获取只读数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def get_mainboard_stocks(conn):
    """获取主板上市股票列表"""
    query = """
    SELECT ts_code, name, industry, area, list_date
    FROM stock_basic
    WHERE market = '主板' AND list_status = 'L'
    """
    return conn.execute(query).df()

def get_latest_valuation(conn, trade_date=None):
    """获取最新估值数据"""
    if trade_date is None:
        date_query = "SELECT MAX(trade_date) FROM daily_basic"
        trade_date = conn.execute(date_query).fetchone()[0]

    query = f"""
    SELECT
        db.ts_code,
        db.trade_date,
        db.close,
        db.pe,
        db.pe_ttm,
        db.pb,
        db.ps,
        db.ps_ttm,
        db.dv_ratio,
        db.dv_ttm,
        db.total_mv,
        db.circ_mv,
        db.turnover_rate
    FROM daily_basic db
    WHERE db.trade_date = '{trade_date}'
    """
    return conn.execute(query).df()

def get_financial_indicators(conn, end_date=None):
    """获取财务指标数据"""
    if end_date is None:
        # 获取最新有充足数据的年报 (选择数据量>1000的最新年份)
        end_date_query = """
        SELECT end_date
        FROM fina_indicator_vip
        WHERE end_date LIKE '%1231'
        GROUP BY end_date
        HAVING COUNT(*) > 1000
        ORDER BY end_date DESC
        LIMIT 1
        """
        end_date = conn.execute(end_date_query).fetchone()[0]
        print(f"  使用年报日期: {end_date}")

    query = f"""
    SELECT
        ts_code,
        end_date,
        eps,
        dt_eps,
        roe,
        roe_dt,
        roa,
        npta,
        gross_margin,
        netprofit_margin,
        debt_to_assets,
        current_ratio,
        quick_ratio,
        netprofit_yoy,
        dt_netprofit_yoy,
        or_yoy,
        roe_yoy,
        ocf_to_debt
    FROM fina_indicator_vip
    WHERE end_date = '{end_date}'
    """
    return conn.execute(query).df()

def get_historical_financials(conn, years=5):
    """获取历史财务数据用于计算盈利稳定性"""
    # 获取最近5年的年报数据
    query = """
    SELECT
        ts_code,
        end_date,
        eps,
        roe,
        netprofit_yoy
    FROM fina_indicator_vip
    WHERE end_date LIKE '%1231'
    ORDER BY ts_code, end_date DESC
    """
    df = conn.execute(query).df()
    return df

def calculate_profit_stability(historical_df):
    """计算盈利稳定性指标"""
    stability_list = []

    for ts_code, group in historical_df.groupby('ts_code'):
        group = group.sort_values('end_date', ascending=False).head(5)

        if len(group) >= 3:
            # 计算EPS标准差/均值 (波动系数)
            eps_values = group['eps'].dropna()
            if len(eps_values) >= 3 and eps_values.mean() != 0:
                eps_cv = eps_values.std() / abs(eps_values.mean()) if eps_values.mean() != 0 else np.nan
            else:
                eps_cv = np.nan

            # 计算ROE均值和标准差
            roe_values = group['roe'].dropna()
            if len(roe_values) >= 3:
                roe_mean = roe_values.mean()
                roe_std = roe_values.std()
            else:
                roe_mean = np.nan
                roe_std = np.nan

            # 盈利连续为正的年数
            positive_years = (eps_values > 0).sum()

            stability_list.append({
                'ts_code': ts_code,
                'eps_cv': eps_cv,
                'roe_mean': roe_mean,
                'roe_std': roe_std,
                'positive_years': positive_years,
                'data_years': len(group)
            })

    return pd.DataFrame(stability_list)

def get_dividend_history(conn, years=5):
    """获取历史分红数据"""
    # 先检查有哪些div_proc状态
    query = """
    SELECT
        ts_code,
        end_date,
        cash_div,
        stk_div,
        cash_div_tax
    FROM dividend
    WHERE cash_div > 0 OR cash_div_tax > 0
    ORDER BY ts_code, end_date DESC
    """
    df = conn.execute(query).df()
    # 如果cash_div为空但cash_div_tax有值,使用cash_div_tax
    if len(df) > 0:
        df['cash_div'] = df['cash_div'].fillna(0) + df['cash_div_tax'].fillna(0)
    return df

def calculate_dividend_consistency(dividend_df, years=5):
    """计算分红连续性"""
    if len(dividend_df) == 0:
        # 返回空DataFrame但带有正确的列
        return pd.DataFrame(columns=['ts_code', 'div_years', 'cash_div_count', 'cash_div_total', 'div_consistency'])

    consistency_list = []

    for ts_code, group in dividend_df.groupby('ts_code'):
        # 取最近5年的分红记录
        recent = group.sort_values('end_date', ascending=False).copy()

        # 提取年份
        recent['year'] = recent['end_date'].str[:4].astype(int)
        recent = recent.drop_duplicates('year').head(years)

        # 计算现金分红次数和总额
        cash_div_count = (recent['cash_div'] > 0).sum()
        cash_div_total = recent['cash_div'].sum()

        consistency_list.append({
            'ts_code': ts_code,
            'div_years': len(recent),
            'cash_div_count': cash_div_count,
            'cash_div_total': cash_div_total,
            'div_consistency': cash_div_count / years if years > 0 else 0
        })

    return pd.DataFrame(consistency_list)

def screen_value_stocks(valuation_df, financial_df, stability_df, dividend_df, stocks_df):
    """
    价值股筛选

    筛选标准:
    1. PE_TTM > 0 且 < 20 (盈利且估值较低)
    2. PB > 0 且 < 3 (资产价值合理)
    3. 股息率 > 2% (高股息)
    4. ROE > 8% (合理盈利能力)
    5. 盈利稳定 (EPS波动系数 < 1)
    """
    # 合并数据
    df = valuation_df.merge(stocks_df[['ts_code', 'name', 'industry']], on='ts_code', how='inner')
    df = df.merge(financial_df[['ts_code', 'roe', 'gross_margin', 'debt_to_assets', 'current_ratio']],
                  on='ts_code', how='left')
    df = df.merge(stability_df, on='ts_code', how='left')
    if len(dividend_df) > 0 and 'div_consistency' in dividend_df.columns:
        df = df.merge(dividend_df[['ts_code', 'div_consistency', 'cash_div_count']], on='ts_code', how='left')
    else:
        df['div_consistency'] = 0
        df['cash_div_count'] = 0

    # 填充缺失值
    df['div_consistency'] = df['div_consistency'].fillna(0)
    df['cash_div_count'] = df['cash_div_count'].fillna(0)

    # 基础筛选条件
    conditions = (
        (df['pe_ttm'] > 0) & (df['pe_ttm'] < 100) &  # 排除亏损和极端PE
        (df['pb'] > 0) & (df['pb'] < 50) &  # 排除负净资产和极端PB
        (df['total_mv'] > 0)  # 有效市值
    )
    df_valid = df[conditions].copy()

    # 价值股筛选 (宽松条件)
    value_conditions = (
        (df_valid['pe_ttm'] <= 20) &  # 低PE
        (df_valid['pb'] <= 3) &  # 低PB
        (df_valid['dv_ttm'] >= 2)  # 股息率 >= 2%
    )

    df_value = df_valid[value_conditions].copy()

    # 深度价值股筛选 (严格条件)
    deep_value_conditions = (
        (df_valid['pe_ttm'] <= 10) &  # 更低PE
        (df_valid['pb'] <= 1.5) &  # 更低PB
        (df_valid['dv_ttm'] >= 4) &  # 更高股息
        (df_valid['roe'] >= 8)  # 合理ROE
    )

    df_deep_value = df_valid[deep_value_conditions].copy()

    # 成长股筛选 (对比组)
    growth_conditions = (
        (df_valid['pe_ttm'] > 30) &  # 高PE
        (df_valid['pb'] > 3)  # 高PB
    )

    df_growth = df_valid[growth_conditions].copy()

    return df_valid, df_value, df_deep_value, df_growth

def get_historical_returns(conn, ts_codes, periods=[20, 60, 120, 250]):
    """计算历史收益率"""
    ts_codes_str = "', '".join(ts_codes)

    # 获取最新交易日
    date_query = "SELECT MAX(trade_date) FROM daily"
    latest_date = conn.execute(date_query).fetchone()[0]

    query = f"""
    WITH ranked_data AS (
        SELECT
            ts_code,
            trade_date,
            close,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
        FROM daily
        WHERE ts_code IN ('{ts_codes_str}')
        AND trade_date <= '{latest_date}'
    )
    SELECT ts_code, trade_date, close, rn
    FROM ranked_data
    WHERE rn <= 260
    ORDER BY ts_code, rn
    """

    df = conn.execute(query).df()

    returns_list = []
    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('rn')
        if len(group) < 2:
            continue

        current_price = group.iloc[0]['close']
        returns_dict = {'ts_code': ts_code}

        for period in periods:
            if len(group) > period:
                past_price = group.iloc[period]['close']
                returns_dict[f'return_{period}d'] = (current_price / past_price - 1) * 100
            else:
                returns_dict[f'return_{period}d'] = np.nan

        # 计算波动率 (日收益率标准差年化)
        if len(group) >= 20:
            prices = group['close'].values
            daily_returns = np.diff(prices) / prices[:-1]
            returns_dict['volatility'] = np.std(daily_returns) * np.sqrt(250) * 100
        else:
            returns_dict['volatility'] = np.nan

        returns_list.append(returns_dict)

    return pd.DataFrame(returns_list)

def analyze_industry_distribution(df_value, df_all):
    """分析行业分布"""
    # 价值股行业分布
    value_industry = df_value.groupby('industry').agg({
        'ts_code': 'count',
        'pe_ttm': 'median',
        'pb': 'median',
        'dv_ttm': 'median',
        'total_mv': 'sum'
    }).reset_index()
    value_industry.columns = ['industry', 'count', 'median_pe', 'median_pb', 'median_div', 'total_mv']
    value_industry['pct'] = value_industry['count'] / value_industry['count'].sum() * 100
    value_industry = value_industry.sort_values('count', ascending=False)

    # 全市场行业分布
    all_industry = df_all.groupby('industry').agg({
        'ts_code': 'count',
        'pe_ttm': 'median',
        'pb': 'median',
        'dv_ttm': 'median'
    }).reset_index()
    all_industry.columns = ['industry', 'all_count', 'all_median_pe', 'all_median_pb', 'all_median_div']

    # 合并计算价值股占比
    industry_analysis = value_industry.merge(all_industry, on='industry', how='left')
    industry_analysis['value_ratio'] = industry_analysis['count'] / industry_analysis['all_count'] * 100

    return industry_analysis

def identify_value_traps(df_value, financial_df, stability_df):
    """识别潜在价值陷阱"""
    df = df_value.copy()

    # 安全地合并财务数据
    fin_cols = ['ts_code']
    for col in ['netprofit_yoy', 'or_yoy', 'debt_to_assets', 'ocf_to_debt']:
        if col in financial_df.columns:
            fin_cols.append(col)

    if len(fin_cols) > 1:
        df = df.merge(financial_df[fin_cols], on='ts_code', how='left')

    # 合并稳定性数据
    stab_cols = ['ts_code']
    for col in ['eps_cv', 'positive_years']:
        if col in stability_df.columns:
            stab_cols.append(col)

    if len(stab_cols) > 1:
        df = df.merge(stability_df[stab_cols], on='ts_code', how='left')

    # 初始化trap_conditions
    df['is_trap_risk'] = False
    df['trap_reasons'] = ''

    # 价值陷阱特征检查 (安全处理列不存在的情况)
    if 'netprofit_yoy' in df.columns:
        mask = df['netprofit_yoy'] < -20
        df.loc[mask, 'is_trap_risk'] = True
        df.loc[mask, 'trap_reasons'] += '利润下滑;'

    if 'debt_to_assets' in df.columns:
        mask = df['debt_to_assets'] > 70
        df.loc[mask, 'is_trap_risk'] = True
        df.loc[mask, 'trap_reasons'] += '高负债;'

    if 'positive_years' in df.columns:
        mask = df['positive_years'] < 3
        df.loc[mask, 'is_trap_risk'] = True
        df.loc[mask, 'trap_reasons'] += '盈利不稳;'

    if 'ocf_to_debt' in df.columns:
        mask = (df['ocf_to_debt'] < 0.1) & (df['ocf_to_debt'].notna())
        df.loc[mask, 'is_trap_risk'] = True
        df.loc[mask, 'trap_reasons'] += '现金流弱;'

    return df

def create_visualizations(df_value, df_growth, df_all, industry_analysis, returns_df, save_dir):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. PE分布对比
    ax1 = axes[0, 0]
    df_value_pe = df_value['pe_ttm'].dropna()
    df_growth_pe = df_growth['pe_ttm'].dropna()
    df_value_pe = df_value_pe[df_value_pe < 50]
    df_growth_pe = df_growth_pe[df_growth_pe < 100]

    ax1.hist(df_value_pe, bins=30, alpha=0.7, label='价值股', color='blue')
    ax1.hist(df_growth_pe, bins=30, alpha=0.7, label='成长股', color='orange')
    ax1.set_xlabel('PE TTM')
    ax1.set_ylabel('数量')
    ax1.set_title('PE分布对比')
    ax1.legend()

    # 2. PB分布对比
    ax2 = axes[0, 1]
    df_value_pb = df_value['pb'].dropna()
    df_growth_pb = df_growth['pb'].dropna()
    df_value_pb = df_value_pb[df_value_pb < 10]
    df_growth_pb = df_growth_pb[df_growth_pb < 20]

    ax2.hist(df_value_pb, bins=30, alpha=0.7, label='价值股', color='blue')
    ax2.hist(df_growth_pb, bins=30, alpha=0.7, label='成长股', color='orange')
    ax2.set_xlabel('PB')
    ax2.set_ylabel('数量')
    ax2.set_title('PB分布对比')
    ax2.legend()

    # 3. 股息率分布
    ax3 = axes[0, 2]
    df_value_div = df_value['dv_ttm'].dropna()
    df_value_div = df_value_div[df_value_div < 15]

    ax3.hist(df_value_div, bins=30, alpha=0.7, color='green')
    ax3.axvline(df_value_div.median(), color='red', linestyle='--', label=f'中位数: {df_value_div.median():.2f}%')
    ax3.set_xlabel('股息率 TTM (%)')
    ax3.set_ylabel('数量')
    ax3.set_title('价值股股息率分布')
    ax3.legend()

    # 4. 行业分布 (Top 15)
    ax4 = axes[1, 0]
    top_industries = industry_analysis.head(15)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_industries)))
    bars = ax4.barh(top_industries['industry'], top_industries['count'], color=colors)
    ax4.set_xlabel('股票数量')
    ax4.set_title('价值股行业分布 (Top 15)')
    ax4.invert_yaxis()

    # 5. 收益率对比 (如果有数据)
    ax5 = axes[1, 1]
    if returns_df is not None and len(returns_df) > 0:
        value_returns = returns_df[returns_df['ts_code'].isin(df_value['ts_code'])]
        growth_returns = returns_df[returns_df['ts_code'].isin(df_growth['ts_code'])]

        periods = ['return_20d', 'return_60d', 'return_120d', 'return_250d']
        period_labels = ['20日', '60日', '120日', '250日']

        value_means = [value_returns[p].mean() for p in periods if p in value_returns.columns]
        growth_means = [growth_returns[p].mean() for p in periods if p in growth_returns.columns]

        x = np.arange(len(period_labels[:len(value_means)]))
        width = 0.35

        ax5.bar(x - width/2, value_means, width, label='价值股', color='blue', alpha=0.7)
        ax5.bar(x + width/2, growth_means, width, label='成长股', color='orange', alpha=0.7)
        ax5.set_ylabel('平均收益率 (%)')
        ax5.set_title('不同周期收益率对比')
        ax5.set_xticks(x)
        ax5.set_xticklabels(period_labels[:len(value_means)])
        ax5.legend()
        ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    else:
        ax5.text(0.5, 0.5, '收益率数据不足', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('收益率对比')

    # 6. 风险收益散点图
    ax6 = axes[1, 2]
    if returns_df is not None and 'volatility' in returns_df.columns and 'return_250d' in returns_df.columns:
        value_risk = returns_df[returns_df['ts_code'].isin(df_value['ts_code'])][['volatility', 'return_250d']].dropna()
        growth_risk = returns_df[returns_df['ts_code'].isin(df_growth['ts_code'])][['volatility', 'return_250d']].dropna()

        if len(value_risk) > 0:
            ax6.scatter(value_risk['volatility'], value_risk['return_250d'], alpha=0.5, label='价值股', color='blue', s=20)
        if len(growth_risk) > 0:
            ax6.scatter(growth_risk['volatility'], growth_risk['return_250d'], alpha=0.5, label='成长股', color='orange', s=20)

        ax6.set_xlabel('波动率 (%)')
        ax6.set_ylabel('年收益率 (%)')
        ax6.set_title('风险收益特征')
        ax6.legend()
        ax6.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    else:
        ax6.text(0.5, 0.5, '风险数据不足', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('风险收益特征')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/value_stock_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {save_dir}/value_stock_analysis.png")

def generate_report(df_all, df_value, df_deep_value, df_growth, industry_analysis,
                   value_traps_df, returns_df, stability_df, save_dir):
    """生成研究报告"""

    report = []
    report.append("=" * 80)
    report.append("主板价值股特征研究报告")
    report.append("=" * 80)
    report.append(f"\n生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据日期: {df_value['trade_date'].iloc[0] if len(df_value) > 0 else 'N/A'}")

    # ==================== 第一部分: 价值股筛选 ====================
    report.append("\n" + "=" * 80)
    report.append("第一部分: 价值股筛选")
    report.append("=" * 80)

    report.append("\n一、筛选标准")
    report.append("-" * 40)
    report.append("""
价值股定义 (宽松标准):
  - PE TTM <= 20 (低市盈率)
  - PB <= 3 (低市净率)
  - 股息率 TTM >= 2% (较高股息)

深度价值股定义 (严格标准):
  - PE TTM <= 10 (极低市盈率)
  - PB <= 1.5 (极低市净率)
  - 股息率 TTM >= 4% (高股息)
  - ROE >= 8% (合理盈利能力)

成长股定义 (对比组):
  - PE TTM > 30 (高市盈率)
  - PB > 3 (高市净率)
""")

    report.append("\n二、筛选结果统计")
    report.append("-" * 40)
    report.append(f"主板有效样本股票数: {len(df_all)}")
    report.append(f"价值股数量: {len(df_value)} (占比 {len(df_value)/len(df_all)*100:.1f}%)")
    report.append(f"深度价值股数量: {len(df_deep_value)} (占比 {len(df_deep_value)/len(df_all)*100:.1f}%)")
    report.append(f"成长股数量: {len(df_growth)} (占比 {len(df_growth)/len(df_all)*100:.1f}%)")

    report.append("\n三、价值股估值特征")
    report.append("-" * 40)
    value_stats = df_value[['pe_ttm', 'pb', 'dv_ttm', 'total_mv']].describe()
    report.append(f"""
                    PE TTM      PB      股息率(%)    市值(亿)
平均值            {df_value['pe_ttm'].mean():8.2f}  {df_value['pb'].mean():6.2f}    {df_value['dv_ttm'].mean():6.2f}    {df_value['total_mv'].mean()/10000:8.1f}
中位数            {df_value['pe_ttm'].median():8.2f}  {df_value['pb'].median():6.2f}    {df_value['dv_ttm'].median():6.2f}    {df_value['total_mv'].median()/10000:8.1f}
最小值            {df_value['pe_ttm'].min():8.2f}  {df_value['pb'].min():6.2f}    {df_value['dv_ttm'].min():6.2f}    {df_value['total_mv'].min()/10000:8.1f}
最大值            {df_value['pe_ttm'].max():8.2f}  {df_value['pb'].max():6.2f}    {df_value['dv_ttm'].max():6.2f}    {df_value['total_mv'].max()/10000:8.1f}
""")

    report.append("\n四、深度价值股列表 (Top 30)")
    report.append("-" * 40)
    if len(df_deep_value) > 0:
        top_deep_value = df_deep_value.nsmallest(30, 'pe_ttm')[
            ['ts_code', 'name', 'industry', 'pe_ttm', 'pb', 'dv_ttm', 'total_mv']
        ].copy()
        top_deep_value['total_mv'] = top_deep_value['total_mv'] / 10000  # 转换为亿
        top_deep_value.columns = ['代码', '名称', '行业', 'PE', 'PB', '股息率%', '市值(亿)']
        report.append(top_deep_value.to_string(index=False))
    else:
        report.append("没有满足深度价值条件的股票")

    # ==================== 第二部分: 价值股特征 ====================
    report.append("\n\n" + "=" * 80)
    report.append("第二部分: 价值股特征分析")
    report.append("=" * 80)

    report.append("\n一、行业分布分析")
    report.append("-" * 40)
    report.append("\n价值股行业分布 (Top 20):")
    top_industries = industry_analysis.head(20)[
        ['industry', 'count', 'pct', 'median_pe', 'median_pb', 'median_div', 'value_ratio']
    ].copy()
    top_industries.columns = ['行业', '数量', '占比%', 'PE中位数', 'PB中位数', '股息率中位数', '行业价值股占比%']
    report.append(top_industries.to_string(index=False))

    report.append("\n\n行业分布特征总结:")
    top3_industries = industry_analysis.head(3)['industry'].tolist()
    report.append(f"  - 价值股主要集中在: {', '.join(top3_industries)}")
    high_value_ratio = industry_analysis[industry_analysis['value_ratio'] > 50].head(5)
    if len(high_value_ratio) > 0:
        report.append(f"  - 价值股占比超过50%的行业: {', '.join(high_value_ratio['industry'].tolist())}")

    report.append("\n二、收益风险特征")
    report.append("-" * 40)

    if returns_df is not None and len(returns_df) > 0:
        value_returns = returns_df[returns_df['ts_code'].isin(df_value['ts_code'])]
        growth_returns = returns_df[returns_df['ts_code'].isin(df_growth['ts_code'])]

        report.append("\n不同周期收益率对比:")
        report.append(f"""
                        价值股          成长股
20日收益率 (%)        {value_returns['return_20d'].mean():8.2f}        {growth_returns['return_20d'].mean():8.2f}
60日收益率 (%)        {value_returns['return_60d'].mean():8.2f}        {growth_returns['return_60d'].mean():8.2f}
120日收益率 (%)       {value_returns['return_120d'].mean():8.2f}        {growth_returns['return_120d'].mean():8.2f}
250日收益率 (%)       {value_returns['return_250d'].mean():8.2f}        {growth_returns['return_250d'].mean():8.2f}
波动率 (%)            {value_returns['volatility'].mean():8.2f}        {growth_returns['volatility'].mean():8.2f}
""")

        # 计算夏普比率 (简化版)
        value_sharpe = value_returns['return_250d'].mean() / value_returns['volatility'].mean() if value_returns['volatility'].mean() > 0 else 0
        growth_sharpe = growth_returns['return_250d'].mean() / growth_returns['volatility'].mean() if growth_returns['volatility'].mean() > 0 else 0
        report.append(f"\n风险调整收益 (年化收益/波动率):")
        report.append(f"  - 价值股: {value_sharpe:.3f}")
        report.append(f"  - 成长股: {growth_sharpe:.3f}")
    else:
        report.append("\n收益率数据不足,无法进行对比分析")

    report.append("\n三、价值股 vs 成长股对比")
    report.append("-" * 40)
    report.append(f"""
                        价值股          成长股
数量                  {len(df_value):8d}        {len(df_growth):8d}
平均PE TTM            {df_value['pe_ttm'].mean():8.2f}        {df_growth['pe_ttm'].mean():8.2f}
平均PB                {df_value['pb'].mean():8.2f}        {df_growth['pb'].mean():8.2f}
平均股息率(%)         {df_value['dv_ttm'].mean():8.2f}        {df_growth['dv_ttm'].mean():8.2f}
平均市值(亿)          {df_value['total_mv'].mean()/10000:8.1f}        {df_growth['total_mv'].mean()/10000:8.1f}
""")

    # ==================== 第三部分: 投资策略 ====================
    report.append("\n\n" + "=" * 80)
    report.append("第三部分: 投资策略建议")
    report.append("=" * 80)

    report.append("\n一、深度价值策略")
    report.append("-" * 40)
    report.append("""
1. 策略定义:
   选取估值极低、股息稳定的优质股票,长期持有等待价值回归

2. 选股标准:
   - PE TTM < 10
   - PB < 1.5
   - 股息率 > 4%
   - ROE > 8%
   - 连续3年以上盈利

3. 持仓管理:
   - 分散持有10-20只股票
   - 单只股票仓位不超过10%
   - 行业分散,避免过度集中

4. 买入/卖出时机:
   - 买入: 当股价跌至历史估值低位时分批建仓
   - 卖出: PE回升至15以上,或基本面恶化
""")

    report.append("\n二、价值陷阱识别与规避")
    report.append("-" * 40)

    if value_traps_df is not None:
        trap_count = value_traps_df['is_trap_risk'].sum()
        report.append(f"\n价值股中存在陷阱风险的股票数: {trap_count} (占比 {trap_count/len(value_traps_df)*100:.1f}%)")

        report.append("""
价值陷阱识别标准:
  1. 利润大幅下滑 (同比下降超过20%)
  2. 高负债风险 (资产负债率超过70%)
  3. 盈利不稳定 (近5年盈利为正不足3年)
  4. 现金流弱 (经营现金流/带息债务 < 10%)
""")

        # 列出高风险股票
        high_risk = value_traps_df[value_traps_df['is_trap_risk'] == True].head(20)
        if len(high_risk) > 0:
            report.append("\n高风险价值陷阱股票示例 (Top 20):")
            risk_display = high_risk[['ts_code', 'name', 'industry', 'pe_ttm', 'pb', 'trap_reasons']].copy()
            risk_display.columns = ['代码', '名称', '行业', 'PE', 'PB', '风险原因']
            report.append(risk_display.to_string(index=False))

    report.append("""
规避价值陷阱的方法:
  1. 关注盈利质量: 经营现金流应能覆盖净利润
  2. 检查行业趋势: 避免夕阳行业和产能过剩行业
  3. 分析负债结构: 警惕高杠杆和短期债务占比高的公司
  4. 跟踪管理层: 关注管理层的战略调整和减持行为
  5. 设置止损: 即使是价值股也需要风控
""")

    report.append("\n三、配置建议")
    report.append("-" * 40)
    report.append("""
1. 组合配置比例建议:
   - 深度价值股: 30-40% (高股息、低估值龙头)
   - 稳健价值股: 30-40% (中等估值、稳定增长)
   - 成长股: 20-30% (高增长、合理估值)
   - 现金: 5-10% (应对波动和加仓机会)

2. 行业配置建议:
""")

    # 基于行业分析给出建议
    recommended_industries = industry_analysis[
        (industry_analysis['median_div'] > 3) &
        (industry_analysis['count'] >= 5)
    ].head(5)

    if len(recommended_industries) > 0:
        report.append("   推荐配置行业 (高股息、股票数量充足):")
        for _, row in recommended_industries.iterrows():
            report.append(f"   - {row['industry']}: 股息率中位数 {row['median_div']:.1f}%, 可选股票 {row['count']}只")

    report.append("""
3. 风险提示:
   - 价值股可能长期低估,需要耐心持有
   - 周期性行业在下行周期时价值陷阱风险高
   - 高股息可能是因为市场预期未来盈利下降
   - 建议结合宏观经济周期调整配置
""")

    # ==================== 第四部分: 附录 ====================
    report.append("\n\n" + "=" * 80)
    report.append("附录: 价值股完整列表")
    report.append("=" * 80)

    report.append(f"\n价值股完整列表 (共 {len(df_value)} 只):")
    report.append("-" * 40)

    value_full = df_value[['ts_code', 'name', 'industry', 'pe_ttm', 'pb', 'dv_ttm', 'total_mv']].copy()
    value_full['total_mv'] = value_full['total_mv'] / 10000
    value_full = value_full.sort_values('dv_ttm', ascending=False)
    value_full.columns = ['代码', '名称', '行业', 'PE', 'PB', '股息率%', '市值(亿)']

    # 保存完整列表到CSV
    value_full.to_csv(f'{save_dir}/value_stocks_list.csv', index=False, encoding='utf-8-sig')
    report.append(f"完整列表已保存到: {save_dir}/value_stocks_list.csv")

    # 在报告中只显示前50只
    report.append("\n股息率最高的50只价值股:")
    report.append(value_full.head(50).to_string(index=False))

    # 保存报告
    report_text = '\n'.join(report)
    report_path = f'{save_dir}/value_stock_research_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存到: {report_path}")

    return report_text

def main():
    """主函数"""
    print("=" * 60)
    print("主板价值股特征研究")
    print("=" * 60)

    conn = get_connection()

    # 1. 获取基础数据
    print("\n[1/7] 获取主板股票列表...")
    stocks_df = get_mainboard_stocks(conn)
    print(f"  主板上市股票数: {len(stocks_df)}")

    print("\n[2/7] 获取最新估值数据...")
    valuation_df = get_latest_valuation(conn)
    # 只保留主板股票
    valuation_df = valuation_df[valuation_df['ts_code'].isin(stocks_df['ts_code'])]
    print(f"  有估值数据的股票数: {len(valuation_df)}")

    print("\n[3/7] 获取财务指标数据...")
    financial_df = get_financial_indicators(conn)
    print(f"  有财务数据的股票数: {len(financial_df)}")

    print("\n[4/7] 计算盈利稳定性...")
    historical_df = get_historical_financials(conn)
    stability_df = calculate_profit_stability(historical_df)
    print(f"  有稳定性数据的股票数: {len(stability_df)}")

    print("\n[5/7] 计算分红连续性...")
    dividend_df = get_dividend_history(conn)
    dividend_consistency = calculate_dividend_consistency(dividend_df)
    print(f"  有分红数据的股票数: {len(dividend_consistency)}")

    # 2. 筛选价值股
    print("\n[6/7] 筛选价值股...")
    df_all, df_value, df_deep_value, df_growth = screen_value_stocks(
        valuation_df, financial_df, stability_df, dividend_consistency, stocks_df
    )
    print(f"  有效样本数: {len(df_all)}")
    print(f"  价值股数: {len(df_value)}")
    print(f"  深度价值股数: {len(df_deep_value)}")
    print(f"  成长股数: {len(df_growth)}")

    # 3. 分析行业分布
    print("\n[7/7] 分析特征并生成报告...")
    industry_analysis = analyze_industry_distribution(df_value, df_all)

    # 4. 识别价值陷阱
    value_traps_df = identify_value_traps(df_value, financial_df, stability_df)

    # 5. 计算历史收益
    all_codes = list(df_value['ts_code'].unique()) + list(df_growth['ts_code'].unique())
    returns_df = get_historical_returns(conn, all_codes[:500])  # 限制数量避免查询太慢

    # 6. 创建可视化
    create_visualizations(df_value, df_growth, df_all, industry_analysis, returns_df, REPORT_DIR)

    # 7. 生成报告
    report = generate_report(
        df_all, df_value, df_deep_value, df_growth,
        industry_analysis, value_traps_df, returns_df, stability_df, REPORT_DIR
    )

    # 8. 保存深度价值股列表
    if len(df_deep_value) > 0:
        # 安全选择存在的列
        save_cols = []
        for col in ['ts_code', 'name', 'industry', 'pe_ttm', 'pb', 'dv_ttm', 'total_mv', 'roe']:
            if col in df_deep_value.columns:
                save_cols.append(col)

        deep_value_save = df_deep_value[save_cols].copy()
        if 'total_mv' in deep_value_save.columns:
            deep_value_save['total_mv'] = deep_value_save['total_mv'] / 10000

        # 按股息率排序(如果存在)
        if 'dv_ttm' in deep_value_save.columns:
            deep_value_save = deep_value_save.sort_values('dv_ttm', ascending=False)

        deep_value_save.to_csv(f'{REPORT_DIR}/deep_value_stocks_list.csv', index=False, encoding='utf-8-sig')
        print(f"\n深度价值股列表已保存到: {REPORT_DIR}/deep_value_stocks_list.csv")

    conn.close()
    print("\n" + "=" * 60)
    print("研究完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
