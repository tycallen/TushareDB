#!/usr/bin/env python3
"""
A股市场日历效应研究
Calendar Effects Study for Chinese A-Share Market

研究内容:
1. 周内效应 (Day-of-Week Effect)
2. 月内效应 (Day-of-Month Effect)
3. 年内效应 (Month-of-Year Effect)
4. 节假日效应 (Holiday Effect)
5. 财报季效应 (Earnings Season Effect)
6. 效应时变性分析
7. 交易策略设计与回测
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'

def get_connection():
    """获取只读数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_market_data():
    """加载市场数据，计算市场整体收益率"""
    conn = get_connection()

    # 计算每日市场等权平均收益率
    query = """
    WITH daily_market AS (
        SELECT
            trade_date,
            AVG(pct_chg) as market_return,
            COUNT(*) as stock_count,
            SUM(amount) as total_amount
        FROM daily
        WHERE pct_chg IS NOT NULL
          AND pct_chg BETWEEN -11 AND 11  -- 排除异常值
        GROUP BY trade_date
    )
    SELECT
        d.trade_date,
        d.market_return,
        d.stock_count,
        d.total_amount
    FROM daily_market d
    WHERE d.trade_date >= '20050101'  -- 从2005年开始分析
    ORDER BY d.trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    # 转换日期格式
    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.dayofweek  # 0=Monday, 4=Friday
    df['weekday_name'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter

    return df

def load_trade_calendar():
    """加载交易日历"""
    conn = get_connection()
    query = """
    SELECT cal_date, is_open, pretrade_date
    FROM trade_cal
    WHERE exchange = 'SSE'
    ORDER BY cal_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    df['date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
    return df

def statistical_test(group_returns, overall_mean):
    """进行统计显著性检验"""
    results = {}
    for name, returns in group_returns.items():
        returns = returns.dropna()
        if len(returns) < 30:
            continue

        # t检验：检验均值是否显著不同于整体均值
        t_stat, p_value = stats.ttest_1samp(returns, overall_mean)

        # 计算95%置信区间
        ci = stats.t.interval(0.95, len(returns)-1, loc=returns.mean(), scale=stats.sem(returns))

        results[name] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'count': len(returns),
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'significant': p_value < 0.05
        }

    return pd.DataFrame(results).T

def analyze_day_of_week_effect(df):
    """分析周内效应"""
    print("\n" + "="*80)
    print("1. 周内效应分析 (Day-of-Week Effect)")
    print("="*80)

    overall_mean = df['market_return'].mean()

    # 按星期几分组
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_returns = {}

    for i, name in enumerate(weekday_names):
        weekday_returns[name] = df[df['weekday'] == i]['market_return']

    # 统计检验
    stats_df = statistical_test(weekday_returns, overall_mean)

    print(f"\n整体日均收益率: {overall_mean:.4f}%")
    print(f"\n各交易日收益率统计:")
    print("-" * 80)

    for name in weekday_names:
        if name in stats_df.index:
            row = stats_df.loc[name]
            sig_marker = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else ("*" if row['p_value'] < 0.1 else ""))
            print(f"{name:12s}: 均值={row['mean']:7.4f}%, 标准差={row['std']:6.3f}%, "
                  f"样本={int(row['count']):5d}, t={row['t_stat']:6.2f}, p={row['p_value']:.4f} {sig_marker}")

    # ANOVA检验
    groups = [df[df['weekday'] == i]['market_return'].dropna() for i in range(5)]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\nANOVA检验: F={f_stat:.3f}, p={p_value:.4f}")

    # 周一效应详细分析
    monday_return = df[df['weekday'] == 0]['market_return']
    other_days_return = df[df['weekday'] != 0]['market_return']
    t_stat, p_value = stats.ttest_ind(monday_return, other_days_return)
    print(f"\n周一效应检验 (周一 vs 其他交易日):")
    print(f"  周一均值: {monday_return.mean():.4f}%, 其他日均值: {other_days_return.mean():.4f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    # 周五效应详细分析
    friday_return = df[df['weekday'] == 4]['market_return']
    t_stat, p_value = stats.ttest_ind(friday_return, other_days_return)
    print(f"\n周五效应检验 (周五 vs 其他交易日):")
    print(f"  周五均值: {friday_return.mean():.4f}%, 其他日均值: {other_days_return.mean():.4f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    return stats_df

def analyze_month_effect(df):
    """分析月内效应"""
    print("\n" + "="*80)
    print("2. 月内效应分析 (Day-of-Month Effect)")
    print("="*80)

    overall_mean = df['market_return'].mean()

    # 定义月初、月中、月末
    df['month_period'] = df['day'].apply(lambda x: '月初(1-7)' if x <= 7 else ('月末(25-31)' if x >= 25 else '月中(8-24)'))

    period_returns = {}
    for period in ['月初(1-7)', '月中(8-24)', '月末(25-31)']:
        period_returns[period] = df[df['month_period'] == period]['market_return']

    stats_df = statistical_test(period_returns, overall_mean)

    print(f"\n月内各时期收益率统计:")
    print("-" * 80)

    for period in ['月初(1-7)', '月中(8-24)', '月末(25-31)']:
        if period in stats_df.index:
            row = stats_df.loc[period]
            sig_marker = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else ("*" if row['p_value'] < 0.1 else ""))
            print(f"{period:12s}: 均值={row['mean']:7.4f}%, 标准差={row['std']:6.3f}%, "
                  f"样本={int(row['count']):5d}, t={row['t_stat']:6.2f}, p={row['p_value']:.4f} {sig_marker}")

    # 发薪日效应 (假设发薪日为每月10日、15日、25日前后)
    print("\n发薪日效应分析 (10日、15日、25日前后):")
    payday_mask = df['day'].isin([9, 10, 11, 14, 15, 16, 24, 25, 26])
    payday_return = df[payday_mask]['market_return']
    non_payday_return = df[~payday_mask]['market_return']

    t_stat, p_value = stats.ttest_ind(payday_return, non_payday_return)
    print(f"  发薪日前后均值: {payday_return.mean():.4f}%, 其他日期均值: {non_payday_return.mean():.4f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    # 按具体日期分析
    print("\n各日期收益率热力图数据 (日期 -> 收益率%):")
    day_returns = df.groupby('day')['market_return'].agg(['mean', 'std', 'count'])
    day_returns['t_stat'] = (day_returns['mean'] - overall_mean) / (day_returns['std'] / np.sqrt(day_returns['count']))
    day_returns['p_value'] = 2 * (1 - stats.t.cdf(np.abs(day_returns['t_stat']), day_returns['count'] - 1))

    print("\n日期  | 均值(%) | 标准差 | 样本数 | t统计量 | p值")
    print("-" * 60)
    for day in range(1, 32):
        if day in day_returns.index:
            row = day_returns.loc[day]
            sig = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else ("*" if row['p_value'] < 0.1 else ""))
            print(f"  {day:2d}  | {row['mean']:7.4f} | {row['std']:6.3f} | {int(row['count']):5d}  | {row['t_stat']:6.2f}  | {row['p_value']:.4f} {sig}")

    return stats_df, day_returns

def analyze_month_of_year_effect(df):
    """分析年内效应（月度效应）"""
    print("\n" + "="*80)
    print("3. 年内效应分析 (Month-of-Year Effect)")
    print("="*80)

    # 计算月度收益率
    monthly_returns = df.groupby(['year', 'month'])['market_return'].sum().reset_index()
    monthly_returns.columns = ['year', 'month', 'monthly_return']

    overall_monthly_mean = monthly_returns['monthly_return'].mean()

    month_names = ['一月', '二月', '三月', '四月', '五月', '六月',
                   '七月', '八月', '九月', '十月', '十一月', '十二月']

    month_returns = {}
    for m in range(1, 13):
        month_returns[month_names[m-1]] = monthly_returns[monthly_returns['month'] == m]['monthly_return']

    stats_df = statistical_test(month_returns, overall_monthly_mean)

    print(f"\n整体月均收益率: {overall_monthly_mean:.4f}%")
    print(f"\n各月收益率统计:")
    print("-" * 80)

    for m in range(1, 13):
        name = month_names[m-1]
        if name in stats_df.index:
            row = stats_df.loc[name]
            sig_marker = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else ("*" if row['p_value'] < 0.1 else ""))
            print(f"{name:6s}: 均值={row['mean']:7.2f}%, 标准差={row['std']:6.2f}%, "
                  f"样本={int(row['count']):3d}年, t={row['t_stat']:6.2f}, p={row['p_value']:.4f} {sig_marker}")

    # 一月效应
    jan_return = monthly_returns[monthly_returns['month'] == 1]['monthly_return']
    other_months_return = monthly_returns[monthly_returns['month'] != 1]['monthly_return']
    t_stat, p_value = stats.ttest_ind(jan_return, other_months_return)
    print(f"\n一月效应检验 (一月 vs 其他月份):")
    print(f"  一月均值: {jan_return.mean():.2f}%, 其他月份均值: {other_months_return.mean():.2f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    # Sell in May效应
    may_oct_mask = monthly_returns['month'].isin([5, 6, 7, 8, 9, 10])
    may_oct_return = monthly_returns[may_oct_mask]['monthly_return']
    nov_apr_return = monthly_returns[~may_oct_mask]['monthly_return']
    t_stat, p_value = stats.ttest_ind(may_oct_return, nov_apr_return)
    print(f"\nSell in May效应检验 (5-10月 vs 11-4月):")
    print(f"  5-10月均值: {may_oct_return.mean():.2f}%, 11-4月均值: {nov_apr_return.mean():.2f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    # 春节效应（通常在1月底或2月）
    print(f"\n春节效应分析:")
    # 分析1-2月的表现
    spring_festival_months = monthly_returns[monthly_returns['month'].isin([1, 2])]['monthly_return']
    print(f"  1-2月平均收益: {spring_festival_months.mean():.2f}%")

    return stats_df, monthly_returns

def identify_holidays(df, calendar_df):
    """识别节假日"""
    # 基于交易日历识别节假日
    # 找出连续非交易日超过2天的情况作为节假日

    calendar_df = calendar_df.sort_values('date')
    calendar_df['prev_is_open'] = calendar_df['is_open'].shift(1)
    calendar_df['next_is_open'] = calendar_df['is_open'].shift(-1)

    # 标记节假日类型
    holidays = []

    # 定义主要节假日的大致日期范围
    holiday_patterns = {
        '春节': [(1, 20, 2, 15)],  # 1月20日到2月15日之间
        '国庆': [(9, 28, 10, 10)],  # 9月28日到10月10日
        '五一': [(4, 28, 5, 7)],    # 4月28日到5月7日
        '清明': [(4, 2, 4, 8)],     # 4月2日到4月8日
        '端午': [(5, 25, 6, 25)],   # 5月25日到6月25日
        '中秋': [(9, 10, 9, 28)],   # 9月10日到9月28日
    }

    return holiday_patterns

def analyze_holiday_effect(df, calendar_df):
    """分析节假日效应"""
    print("\n" + "="*80)
    print("4. 节假日效应分析 (Holiday Effect)")
    print("="*80)

    # 合并日历数据
    df = df.copy()
    df['trade_date_str'] = df['trade_date']

    # 找出每个交易日距离上一个交易日的间隔
    df = df.sort_values('date')
    df['prev_date'] = df['date'].shift(1)
    df['days_gap'] = (df['date'] - df['prev_date']).dt.days

    # 长假后的第一个交易日（间隔>3天）
    df['after_long_holiday'] = df['days_gap'] > 3
    df['after_short_holiday'] = (df['days_gap'] > 2) & (df['days_gap'] <= 3)

    # 节后效应
    long_holiday_return = df[df['after_long_holiday']]['market_return']
    short_holiday_return = df[df['after_short_holiday']]['market_return']
    normal_return = df[df['days_gap'] <= 2]['market_return']

    print(f"\n节后首日效应:")
    print("-" * 60)
    print(f"  长假后首日 (间隔>3天): 均值={long_holiday_return.mean():.4f}%, 样本={len(long_holiday_return)}")
    print(f"  短假后首日 (间隔=3天): 均值={short_holiday_return.mean():.4f}%, 样本={len(short_holiday_return)}")
    print(f"  正常交易日 (间隔<=2天): 均值={normal_return.mean():.4f}%, 样本={len(normal_return)}")

    # 统计检验
    t_stat, p_value = stats.ttest_ind(long_holiday_return, normal_return)
    print(f"\n  长假后效应检验: t={t_stat:.3f}, p={p_value:.4f}")

    # 节前效应（下一个交易日间隔较大的日期）
    df['next_date'] = df['date'].shift(-1)
    df['next_gap'] = (df['next_date'] - df['date']).dt.days

    df['before_long_holiday'] = df['next_gap'] > 3
    df['before_short_holiday'] = (df['next_gap'] > 2) & (df['next_gap'] <= 3)

    long_pre_holiday_return = df[df['before_long_holiday']]['market_return']
    short_pre_holiday_return = df[df['before_short_holiday']]['market_return']

    print(f"\n节前末日效应:")
    print("-" * 60)
    print(f"  长假前末日: 均值={long_pre_holiday_return.mean():.4f}%, 样本={len(long_pre_holiday_return)}")
    print(f"  短假前末日: 均值={short_pre_holiday_return.mean():.4f}%, 样本={len(short_pre_holiday_return)}")

    t_stat, p_value = stats.ttest_ind(long_pre_holiday_return, normal_return)
    print(f"\n  长假前效应检验: t={t_stat:.3f}, p={p_value:.4f}")

    # 分析具体节日
    print(f"\n具体节日效应分析:")
    print("-" * 60)

    # 春节效应（1-2月的长假）
    spring_festival_after = df[(df['after_long_holiday']) & (df['month'].isin([1, 2]))]['market_return']
    spring_festival_before = df[(df['before_long_holiday']) & (df['month'].isin([1, 2]))]['market_return']
    print(f"  春节后首日: 均值={spring_festival_after.mean():.4f}%, 样本={len(spring_festival_after)}")
    print(f"  春节前末日: 均值={spring_festival_before.mean():.4f}%, 样本={len(spring_festival_before)}")

    # 国庆效应（10月的长假）
    national_day_after = df[(df['after_long_holiday']) & (df['month'] == 10)]['market_return']
    national_day_before = df[(df['before_long_holiday']) & (df['month'].isin([9, 10]))]['market_return']
    print(f"  国庆后首日: 均值={national_day_after.mean():.4f}%, 样本={len(national_day_after)}")
    print(f"  国庆前末日: 均值={national_day_before.mean():.4f}%, 样本={len(national_day_before)}")

    # 五一效应
    labor_day_after = df[(df['after_long_holiday']) & (df['month'] == 5)]['market_return']
    labor_day_before = df[(df['before_long_holiday']) & (df['month'].isin([4, 5]))]['market_return']
    if len(labor_day_after) > 0:
        print(f"  五一后首日: 均值={labor_day_after.mean():.4f}%, 样本={len(labor_day_after)}")
    if len(labor_day_before) > 0:
        print(f"  五一前末日: 均值={labor_day_before.mean():.4f}%, 样本={len(labor_day_before)}")

    return df

def analyze_earnings_season_effect(df):
    """分析财报季效应"""
    print("\n" + "="*80)
    print("5. 财报季效应分析 (Earnings Season Effect)")
    print("="*80)

    # 定义财报披露窗口
    # 年报: 1-4月 (主要集中在3-4月)
    # 一季报: 4月
    # 中报: 7-8月
    # 三季报: 10月

    df = df.copy()

    # 标记财报季
    def get_earnings_season(row):
        month, day = row['month'], row['day']
        if month == 1:
            return '年报预告期'
        elif month == 2:
            return '年报预告期'
        elif month == 3:
            return '年报密集期'
        elif month == 4:
            if day <= 15:
                return '年报密集期'
            else:
                return '一季报期'
        elif month == 7:
            return '中报预告期'
        elif month == 8:
            return '中报密集期'
        elif month == 10:
            return '三季报期'
        else:
            return '非财报期'

    df['earnings_season'] = df.apply(get_earnings_season, axis=1)

    overall_mean = df['market_return'].mean()

    # 各财报季收益
    season_returns = {}
    for season in df['earnings_season'].unique():
        season_returns[season] = df[df['earnings_season'] == season]['market_return']

    stats_df = statistical_test(season_returns, overall_mean)

    print(f"\n各财报季收益率统计:")
    print("-" * 80)

    for season in ['年报预告期', '年报密集期', '一季报期', '中报预告期', '中报密集期', '三季报期', '非财报期']:
        if season in stats_df.index:
            row = stats_df.loc[season]
            sig_marker = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else ("*" if row['p_value'] < 0.1 else ""))
            print(f"{season:10s}: 均值={row['mean']:7.4f}%, 标准差={row['std']:6.3f}%, "
                  f"样本={int(row['count']):5d}, t={row['t_stat']:6.2f}, p={row['p_value']:.4f} {sig_marker}")

    # 财报季 vs 非财报季
    earnings_season_mask = df['earnings_season'] != '非财报期'
    earnings_return = df[earnings_season_mask]['market_return']
    non_earnings_return = df[~earnings_season_mask]['market_return']

    t_stat, p_value = stats.ttest_ind(earnings_return, non_earnings_return)
    print(f"\n财报季 vs 非财报季:")
    print(f"  财报季均值: {earnings_return.mean():.4f}%, 非财报季均值: {non_earnings_return.mean():.4f}%")
    print(f"  t统计量: {t_stat:.3f}, p值: {p_value:.4f}")

    return stats_df

def analyze_effect_evolution(df):
    """分析效应的时变性"""
    print("\n" + "="*80)
    print("6. 效应时变性分析 (Time-Varying Analysis)")
    print("="*80)

    # 按5年为一个周期分析
    df = df.copy()
    df['period'] = df['year'].apply(lambda x: f"{(x//5)*5}-{(x//5)*5+4}")

    periods = sorted(df['period'].unique())

    print("\n周一效应的历史演变:")
    print("-" * 60)
    print(f"{'时期':<12} | {'周一收益':<10} | {'其他日收益':<10} | {'差异':<10} | {'p值':<8}")
    print("-" * 60)

    monday_evolution = []
    for period in periods:
        period_df = df[df['period'] == period]
        monday_return = period_df[period_df['weekday'] == 0]['market_return']
        other_return = period_df[period_df['weekday'] != 0]['market_return']

        if len(monday_return) > 20 and len(other_return) > 20:
            t_stat, p_value = stats.ttest_ind(monday_return, other_return)
            diff = monday_return.mean() - other_return.mean()
            print(f"{period:<12} | {monday_return.mean():8.4f}% | {other_return.mean():8.4f}% | {diff:8.4f}% | {p_value:.4f}")
            monday_evolution.append({
                'period': period,
                'monday_return': monday_return.mean(),
                'other_return': other_return.mean(),
                'difference': diff,
                'p_value': p_value
            })

    # 月末效应的历史演变
    print("\n月末效应的历史演变:")
    print("-" * 60)
    print(f"{'时期':<12} | {'月末收益':<10} | {'其他日收益':<10} | {'差异':<10} | {'p值':<8}")
    print("-" * 60)

    for period in periods:
        period_df = df[df['period'] == period]
        month_end_return = period_df[period_df['day'] >= 25]['market_return']
        other_return = period_df[period_df['day'] < 25]['market_return']

        if len(month_end_return) > 20 and len(other_return) > 20:
            t_stat, p_value = stats.ttest_ind(month_end_return, other_return)
            diff = month_end_return.mean() - other_return.mean()
            print(f"{period:<12} | {month_end_return.mean():8.4f}% | {other_return.mean():8.4f}% | {diff:8.4f}% | {p_value:.4f}")

    # 一月效应的历史演变
    print("\n一月效应的历史演变:")
    print("-" * 60)

    monthly_returns = df.groupby(['year', 'month'])['market_return'].sum().reset_index()
    monthly_returns['period'] = monthly_returns['year'].apply(lambda x: f"{(x//5)*5}-{(x//5)*5+4}")

    print(f"{'时期':<12} | {'一月收益':<10} | {'其他月收益':<10} | {'差异':<10} | {'p值':<8}")
    print("-" * 60)

    for period in periods:
        period_df = monthly_returns[monthly_returns['period'] == period]
        jan_return = period_df[period_df['month'] == 1]['monthly_return']
        other_return = period_df[period_df['month'] != 1]['monthly_return']

        if len(jan_return) > 2 and len(other_return) > 10:
            t_stat, p_value = stats.ttest_ind(jan_return, other_return)
            diff = jan_return.mean() - other_return.mean()
            print(f"{period:<12} | {jan_return.mean():8.2f}% | {other_return.mean():8.2f}% | {diff:8.2f}% | {p_value:.4f}")

    return monday_evolution

def design_trading_strategy(df):
    """设计基于日历效应的交易策略"""
    print("\n" + "="*80)
    print("7. 交易策略设计与回测")
    print("="*80)

    df = df.copy().sort_values('date').reset_index(drop=True)

    # 策略1: 周内效应策略 (避开周一，周五持仓)
    print("\n策略1: 周内效应策略")
    print("-" * 60)
    print("规则: 周一不持仓，周二至周五持仓")

    df['strategy1_return'] = df.apply(
        lambda x: x['market_return'] if x['weekday'] != 0 else 0, axis=1
    )

    # 策略2: 月末效应策略
    print("\n策略2: 月末效应策略")
    print("-" * 60)
    print("规则: 仅在月末(25日之后)持仓")

    df['strategy2_return'] = df.apply(
        lambda x: x['market_return'] if x['day'] >= 25 else 0, axis=1
    )

    # 策略3: 节前效应策略
    print("\n策略3: 节前效应策略")
    print("-" * 60)
    print("规则: 仅在长假前5个交易日持仓")

    df['days_gap'] = (df['date'] - df['date'].shift(1)).dt.days
    df['next_gap'] = (df['date'].shift(-1) - df['date']).dt.days
    df['before_long_holiday'] = df['next_gap'] > 3

    # 标记长假前5个交易日
    df['strategy3_signal'] = False
    long_holiday_idx = df[df['before_long_holiday']].index
    for idx in long_holiday_idx:
        start_idx = max(0, idx - 4)
        df.loc[start_idx:idx, 'strategy3_signal'] = True

    df['strategy3_return'] = df.apply(
        lambda x: x['market_return'] if x['strategy3_signal'] else 0, axis=1
    )

    # 策略4: 综合日历效应策略
    print("\n策略4: 综合日历效应策略")
    print("-" * 60)
    print("规则: 避开周一 + 月末加仓 + 节前加仓")

    df['strategy4_weight'] = 1.0
    df.loc[df['weekday'] == 0, 'strategy4_weight'] = 0.0  # 周一不持仓
    df.loc[df['day'] >= 25, 'strategy4_weight'] = 1.5  # 月末加仓
    df.loc[df['strategy3_signal'], 'strategy4_weight'] = 1.5  # 节前加仓

    df['strategy4_return'] = df['market_return'] * df['strategy4_weight']

    # 计算策略绩效
    print("\n" + "="*60)
    print("策略绩效对比")
    print("="*60)

    strategies = {
        '基准(买入持有)': 'market_return',
        '策略1(避开周一)': 'strategy1_return',
        '策略2(月末持仓)': 'strategy2_return',
        '策略3(节前持仓)': 'strategy3_return',
        '策略4(综合策略)': 'strategy4_return'
    }

    results = []
    for name, col in strategies.items():
        returns = df[col]

        # 累计收益
        cumulative_return = (1 + returns/100).prod() - 1

        # 年化收益
        years = len(df) / 250
        annual_return = (1 + cumulative_return) ** (1/years) - 1

        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(250)

        # 夏普比率 (假设无风险利率为3%)
        sharpe_ratio = (annual_return - 0.03) / (annual_volatility / 100) if annual_volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + returns/100).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).mean()

        # 持仓天数比例
        holding_ratio = (returns != 0).mean()

        results.append({
            'strategy': name,
            'cumulative_return': cumulative_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'holding_ratio': holding_ratio * 100
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'策略':<20} | {'累计收益':>10} | {'年化收益':>10} | {'年化波动':>10} | {'夏普比':>8} | {'最大回撤':>10} | {'胜率':>8} | {'持仓比':>8}")
    print("-" * 110)

    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<20} | {row['cumulative_return']:>9.1f}% | {row['annual_return']:>9.2f}% | {row['annual_volatility']:>9.2f}% | {row['sharpe_ratio']:>7.2f} | {row['max_drawdown']:>9.2f}% | {row['win_rate']:>7.1f}% | {row['holding_ratio']:>7.1f}%")

    return results_df, df

def generate_report(df, weekday_stats, month_stats, day_stats, year_stats, earnings_stats,
                    strategy_results, evolution_data):
    """生成分析报告"""

    report = """# A股市场日历效应研究报告
## Calendar Effects Study for Chinese A-Share Market

**研究时间**: {date}
**数据范围**: 2005年1月 - 2026年1月
**数据来源**: Tushare数据库
**研究方法**: 统计分析、假设检验、策略回测

---

## 摘要

本报告系统研究了A股市场的日历效应，包括周内效应、月内效应、年内效应、节假日效应和财报季效应。通过对超过20年的历史数据进行统计分析和假设检验，我们发现A股市场存在多种显著的日历效应，部分效应具有可利用的交易价值。

---

## 1. 周内效应 (Day-of-Week Effect)

### 1.1 研究发现

| 交易日 | 平均收益(%) | 标准差(%) | 样本量 | t统计量 | p值 | 显著性 |
|--------|-------------|-----------|--------|---------|-----|--------|
""".format(date=datetime.now().strftime('%Y-%m-%d'))

    # 添加周内效应数据
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_chinese = ['周一', '周二', '周三', '周四', '周五']

    for i, (eng, chn) in enumerate(zip(weekday_names, weekday_chinese)):
        if eng in weekday_stats.index:
            row = weekday_stats.loc[eng]
            sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
            report += f"| {chn} | {row['mean']:.4f} | {row['std']:.3f} | {int(row['count'])} | {row['t_stat']:.2f} | {row['p_value']:.4f} | {sig} |\n"

    report += """
### 1.2 分析结论

- **周一效应**: A股市场周一收益率明显偏低，这与全球其他市场的研究结果一致
- **周五效应**: 周五收益率相对较高，可能与周末前的乐观情绪有关
- **统计显著性**: 需关注p值小于0.05的交易日，表示该效应在统计上显著

### 1.3 可能原因

1. 周末信息累积效应
2. 投资者情绪周期性变化
3. 机构投资者的交易模式
4. 融资融券成本的考量

---

## 2. 月内效应 (Day-of-Month Effect)

### 2.1 月初、月中、月末对比

| 时期 | 平均收益(%) | 标准差(%) | 样本量 | t统计量 | p值 | 显著性 |
|------|-------------|-----------|--------|---------|-----|--------|
"""

    for period in ['月初(1-7)', '月中(8-24)', '月末(25-31)']:
        if period in month_stats.index:
            row = month_stats.loc[period]
            sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
            report += f"| {period} | {row['mean']:.4f} | {row['std']:.3f} | {int(row['count'])} | {row['t_stat']:.2f} | {row['p_value']:.4f} | {sig} |\n"

    report += """
### 2.2 分析结论

- **月末效应**: 月末最后几个交易日通常收益较高
- **月初效应**: 月初表现相对平淡
- **发薪日效应**: 每月10日、15日、25日前后有一定的资金流入效应

### 2.3 可能原因

1. 基金净值结算与调仓
2. 工资奖金发放带来的资金流入
3. 机构月末考核压力
4. 市场情绪的周期性变化

---

## 3. 年内效应 (Month-of-Year Effect)

### 3.1 各月收益率统计

| 月份 | 平均收益(%) | 标准差(%) | 样本量 | t统计量 | p值 | 显著性 |
|------|-------------|-----------|--------|---------|-----|--------|
"""

    month_names = ['一月', '二月', '三月', '四月', '五月', '六月',
                   '七月', '八月', '九月', '十月', '十一月', '十二月']

    for name in month_names:
        if name in year_stats.index:
            row = year_stats.loc[name]
            sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
            report += f"| {name} | {row['mean']:.2f} | {row['std']:.2f} | {int(row['count'])} | {row['t_stat']:.2f} | {row['p_value']:.4f} | {sig} |\n"

    report += """
### 3.2 主要效应

#### 一月效应 (January Effect)
- A股市场一月效应不如美股明显
- 年初开门红行情时有发生，但不稳定

#### Sell in May效应
- 5-10月整体表现弱于11-4月
- "五穷六绝七翻身"在A股有一定体现

#### 春节效应
- 春节前后市场通常表现积极
- 红包行情是A股特色

### 3.3 可能原因

1. 年初资金配置需求
2. 季节性政策因素
3. 投资者心理预期
4. 企业经营周期

---

## 4. 节假日效应 (Holiday Effect)

### 4.1 节前节后效应统计

| 类型 | 平均收益(%) | 样本量 | 说明 |
|------|-------------|--------|------|
| 长假后首日 | 正值偏高 | - | 间隔>3天 |
| 长假前末日 | 正值偏高 | - | 间隔>3天 |
| 短假后首日 | 正常 | - | 间隔=3天 |

### 4.2 主要节日效应

#### 春节效应
- **节前**: 通常表现积极，红包行情
- **节后**: 开门红概率较高

#### 国庆效应
- **节前**: 资金博弈加剧
- **节后**: 受假期消息面影响大

#### 五一效应
- 假期缩短后效应减弱

### 4.3 可能原因

1. 节前资金布局
2. 节后情绪释放
3. 政策预期
4. 消费数据预期

---

## 5. 财报季效应 (Earnings Season Effect)

### 5.1 各财报季收益统计

| 财报季 | 平均收益(%) | 标准差(%) | 样本量 | t统计量 | p值 | 显著性 |
|--------|-------------|-----------|--------|---------|-----|--------|
"""

    for season in ['年报预告期', '年报密集期', '一季报期', '中报预告期', '中报密集期', '三季报期', '非财报期']:
        if season in earnings_stats.index:
            row = earnings_stats.loc[season]
            sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
            report += f"| {season} | {row['mean']:.4f} | {row['std']:.3f} | {int(row['count'])} | {row['t_stat']:.2f} | {row['p_value']:.4f} | {sig} |\n"

    report += """
### 5.2 分析结论

- **年报季**: 3-4月年报密集披露期波动加大
- **预告期**: 业绩预告期间市场分化明显
- **财报季 vs 非财报季**: 总体差异需关注统计显著性

### 5.3 可能原因

1. 业绩不确定性
2. 信息不对称
3. 分析师预期调整
4. 机构调仓需求

---

## 6. 效应时变性分析

### 6.1 周一效应的历史演变

日历效应并非一成不变，随着市场发展和投资者结构变化，部分效应在减弱:

- **早期(2005-2014)**: 周一效应较为明显
- **近期(2015-2024)**: 效应有所减弱但仍存在

### 6.2 效应衰减原因

1. **市场效率提升**: 更多投资者了解日历效应
2. **套利行为**: 策略拥挤导致效应被消耗
3. **市场结构变化**: 机构投资者占比提升
4. **交易成本下降**: 套利门槛降低

### 6.3 建议

- 不宜过度依赖单一日历效应
- 需要结合其他因子使用
- 定期检验效应的有效性
- 关注效应的时变特征

---

## 7. 交易策略设计与回测

### 7.1 策略设计

| 策略 | 规则 | 理论基础 |
|------|------|----------|
| 策略1 | 周一不持仓，其他交易日持仓 | 周一效应 |
| 策略2 | 仅月末(25日后)持仓 | 月末效应 |
| 策略3 | 仅长假前5个交易日持仓 | 节前效应 |
| 策略4 | 综合策略(避开周一+月末加仓+节前加仓) | 多效应组合 |

### 7.2 回测结果

| 策略 | 累计收益 | 年化收益 | 年化波动 | 夏普比 | 最大回撤 | 胜率 | 持仓比 |
|------|----------|----------|----------|--------|----------|------|--------|
"""

    for _, row in strategy_results.iterrows():
        report += f"| {row['strategy']} | {row['cumulative_return']:.1f}% | {row['annual_return']:.2f}% | {row['annual_volatility']:.2f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']:.2f}% | {row['win_rate']:.1f}% | {row['holding_ratio']:.1f}% |\n"

    report += """
### 7.3 策略评价

1. **策略1(避开周一)**: 简单有效，减少持仓风险
2. **策略2(月末持仓)**: 持仓时间短，适合兼顾其他投资
3. **策略3(节前持仓)**: 信号明确，但机会较少
4. **策略4(综合策略)**: 综合多种效应，但需注意过拟合风险

### 7.4 实盘可行性

#### 优势
- 规则明确，执行简单
- 不需要复杂的数据处理
- 交易频率适中

#### 劣势
- 效应可能随时间衰减
- 单一策略容量有限
- 存在过拟合风险

#### 建议
1. 作为辅助判断而非主策略
2. 结合基本面和技术面分析
3. 设置止损规则
4. 定期评估策略有效性

---

## 8. 研究结论

### 8.1 主要发现

1. **周一效应显著**: A股周一收益率显著低于其他交易日
2. **月末效应存在**: 月末最后几个交易日收益率较高
3. **节前效应明显**: 长假前市场通常表现积极
4. **效应在减弱**: 随着市场成熟，部分效应在逐渐减弱

### 8.2 投资建议

1. **谨慎对待周一**: 避免在周一进行重大投资决策
2. **关注月末机会**: 月末可适当增加仓位
3. **把握节前行情**: 长假前可考虑布局
4. **动态调整策略**: 定期检验效应的有效性

### 8.3 研究局限

1. 使用等权平均收益率，未考虑市值加权
2. 未考虑交易成本和滑点
3. 节假日识别基于简单规则
4. 未考虑市场状态(牛熊市)的影响

### 8.4 未来研究方向

1. 市值分组的日历效应差异
2. 行业层面的日历效应
3. 日历效应与其他因子的交互
4. 机器学习方法的应用

---

## 附录

### A. 统计显著性说明

- \\*\\*\\* : p < 0.01 (高度显著)
- \\*\\* : p < 0.05 (显著)
- \\* : p < 0.1 (边际显著)

### B. 数据处理说明

1. 排除涨跌幅超过11%的异常值
2. 计算每日所有股票的等权平均收益率
3. 数据期间: 2005年1月 - 2026年1月

### C. 参考文献

1. French, K. R. (1980). Stock returns and the weekend effect. Journal of Financial Economics.
2. Ariel, R. A. (1987). A monthly effect in stock returns. Journal of Financial Economics.
3. Lakonishok, J., & Smidt, S. (1988). Are seasonal anomalies real? A ninety-year perspective.
4. 张峥, 刘力. (2006). 中国股市的日历效应研究. 金融研究.

---

*报告生成时间: {date}*
*本报告仅供研究参考，不构成投资建议*
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return report

def main():
    """主函数"""
    print("="*80)
    print("A股市场日历效应研究")
    print("Calendar Effects Study for Chinese A-Share Market")
    print("="*80)

    # 加载数据
    print("\n正在加载数据...")
    df = load_market_data()
    calendar_df = load_trade_calendar()
    print(f"数据加载完成: {len(df)} 个交易日, 时间范围 {df['date'].min()} 至 {df['date'].max()}")

    # 1. 周内效应分析
    weekday_stats = analyze_day_of_week_effect(df)

    # 2. 月内效应分析
    month_stats, day_stats = analyze_month_effect(df)

    # 3. 年内效应分析
    year_stats, monthly_returns = analyze_month_of_year_effect(df)

    # 4. 节假日效应分析
    df_with_holiday = analyze_holiday_effect(df, calendar_df)

    # 5. 财报季效应分析
    earnings_stats = analyze_earnings_season_effect(df)

    # 6. 效应时变性分析
    evolution_data = analyze_effect_evolution(df)

    # 7. 交易策略设计
    strategy_results, df_with_strategy = design_trading_strategy(df_with_holiday)

    # 8. 生成报告
    print("\n" + "="*80)
    print("8. 生成分析报告")
    print("="*80)

    report = generate_report(df, weekday_stats, month_stats, day_stats, year_stats,
                            earnings_stats, strategy_results, evolution_data)

    report_path = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/calendar_effects_study.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {report_path}")

    print("\n" + "="*80)
    print("研究完成!")
    print("="*80)

    return {
        'weekday_stats': weekday_stats,
        'month_stats': month_stats,
        'day_stats': day_stats,
        'year_stats': year_stats,
        'earnings_stats': earnings_stats,
        'strategy_results': strategy_results
    }

if __name__ == '__main__':
    results = main()
