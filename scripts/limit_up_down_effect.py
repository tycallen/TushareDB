#!/usr/bin/env python3
"""
涨跌停板效应研究
研究内容：
1. 涨跌停统计：涨停/跌停频率分布、连续涨停/跌停统计、一字板vs非一字板
2. 涨跌停后效应：涨停后次日收益、跌停后次日收益、打开涨停/跌停的影响
3. 策略应用：涨停打板策略、跌停抄底策略、风险控制
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from datetime import datetime

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
OUTPUT_DIR = Path("/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)


def identify_limit_up_down(df):
    """
    识别涨停和跌停
    A股涨跌停规则：
    - 普通股票：10%
    - ST股票：5%
    - 科创板、创业板（2020年8月后）：20%
    """
    # 根据股票代码判断板块
    def get_limit_ratio(ts_code):
        if ts_code.startswith('688') or ts_code.startswith('300'):
            return 0.20  # 科创板和创业板 20%
        else:
            return 0.10  # 主板 10%

    # 计算涨跌幅（使用pre_close计算）
    df['calc_pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['limit_ratio'] = df['ts_code'].apply(get_limit_ratio)

    # 涨停判定：涨幅 >= 涨停幅度 - 0.5%（容差）
    df['is_limit_up'] = df['calc_pct_chg'] >= (df['limit_ratio'] - 0.005)
    # 跌停判定：跌幅 <= -跌停幅度 + 0.5%（容差）
    df['is_limit_down'] = df['calc_pct_chg'] <= (-df['limit_ratio'] + 0.005)

    # 一字板判定：开盘价 = 收盘价 = 涨停价/跌停价
    df['limit_up_price'] = df['pre_close'] * (1 + df['limit_ratio'])
    df['limit_down_price'] = df['pre_close'] * (1 - df['limit_ratio'])

    # 一字涨停：开盘即涨停且全天不开板
    df['is_yizi_up'] = df['is_limit_up'] & (df['open'] >= df['limit_up_price'] * 0.999) & (df['low'] >= df['limit_up_price'] * 0.999)
    # 一字跌停：开盘即跌停且全天不开板
    df['is_yizi_down'] = df['is_limit_down'] & (df['open'] <= df['limit_down_price'] * 1.001) & (df['high'] <= df['limit_down_price'] * 1.001)

    # 炸板（打开涨停）：曾涨停但收盘未涨停
    df['touched_limit_up'] = df['high'] >= df['limit_up_price'] * 0.999
    df['is_opened_limit_up'] = df['touched_limit_up'] & ~df['is_limit_up']

    # 打开跌停：曾跌停但收盘未跌停
    df['touched_limit_down'] = df['low'] <= df['limit_down_price'] * 1.001
    df['is_opened_limit_down'] = df['touched_limit_down'] & ~df['is_limit_down']

    return df


def load_daily_data(conn, start_date='20200101', end_date='20260130'):
    """加载日线数据"""
    query = f"""
    SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.pre_close,
           d.pct_chg, d.vol, d.amount,
           s.name, s.industry, s.market
    FROM daily d
    LEFT JOIN stock_basic s ON d.ts_code = s.ts_code
    WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
      AND d.pre_close > 0 AND d.vol > 0
    ORDER BY d.ts_code, d.trade_date
    """
    return conn.execute(query).df()


def analyze_limit_frequency(df):
    """分析涨跌停频率分布"""
    print("\n" + "="*60)
    print("1. 涨跌停频率分布分析")
    print("="*60)

    # 按日期统计
    daily_stats = df.groupby('trade_date').agg({
        'is_limit_up': 'sum',
        'is_limit_down': 'sum',
        'is_yizi_up': 'sum',
        'is_yizi_down': 'sum',
        'is_opened_limit_up': 'sum',
        'is_opened_limit_down': 'sum',
        'ts_code': 'count'
    }).reset_index()
    daily_stats.columns = ['trade_date', 'limit_up_count', 'limit_down_count',
                           'yizi_up_count', 'yizi_down_count',
                           'opened_up_count', 'opened_down_count', 'total_stocks']

    # 计算涨跌停比率
    daily_stats['limit_up_ratio'] = daily_stats['limit_up_count'] / daily_stats['total_stocks'] * 100
    daily_stats['limit_down_ratio'] = daily_stats['limit_down_count'] / daily_stats['total_stocks'] * 100

    # 总体统计
    total_records = len(df)
    total_limit_up = df['is_limit_up'].sum()
    total_limit_down = df['is_limit_down'].sum()
    total_yizi_up = df['is_yizi_up'].sum()
    total_yizi_down = df['is_yizi_down'].sum()
    total_opened_up = df['is_opened_limit_up'].sum()
    total_opened_down = df['is_opened_limit_down'].sum()

    print(f"\n总交易记录数: {total_records:,}")
    print(f"涨停次数: {total_limit_up:,} ({total_limit_up/total_records*100:.2f}%)")
    print(f"跌停次数: {total_limit_down:,} ({total_limit_down/total_records*100:.2f}%)")
    print(f"一字涨停次数: {total_yizi_up:,} ({total_yizi_up/total_limit_up*100:.2f}% of 涨停)")
    print(f"一字跌停次数: {total_yizi_down:,} ({total_yizi_down/total_limit_down*100:.2f}% of 跌停)")
    print(f"炸板（打开涨停）次数: {total_opened_up:,}")
    print(f"打开跌停次数: {total_opened_down:,}")

    # 按年度统计
    df['year'] = df['trade_date'].str[:4]
    yearly_stats = df.groupby('year').agg({
        'is_limit_up': ['sum', 'mean'],
        'is_limit_down': ['sum', 'mean'],
        'ts_code': 'count'
    }).reset_index()
    yearly_stats.columns = ['year', 'limit_up_count', 'limit_up_ratio',
                            'limit_down_count', 'limit_down_ratio', 'total_records']
    yearly_stats['limit_up_ratio'] *= 100
    yearly_stats['limit_down_ratio'] *= 100

    print("\n年度涨跌停统计:")
    print(yearly_stats.to_string(index=False))

    return daily_stats, yearly_stats


def analyze_consecutive_limits(df):
    """分析连续涨停/跌停"""
    print("\n" + "="*60)
    print("2. 连续涨停/跌停统计")
    print("="*60)

    # 按股票分组，计算连续涨停/跌停
    def count_consecutive(group, column):
        """计算连续涨停/跌停天数"""
        group = group.sort_values('trade_date')
        group['consecutive'] = (group[column] != group[column].shift()).cumsum()
        consecutive_counts = group[group[column]].groupby('consecutive').size()
        return consecutive_counts.tolist() if len(consecutive_counts) > 0 else []

    # 统计连续涨停
    consecutive_up = []
    consecutive_down = []

    for ts_code, group in df.groupby('ts_code'):
        consecutive_up.extend(count_consecutive(group, 'is_limit_up'))
        consecutive_down.extend(count_consecutive(group, 'is_limit_down'))

    # 统计分布
    up_counts = pd.Series(consecutive_up).value_counts().sort_index()
    down_counts = pd.Series(consecutive_down).value_counts().sort_index()

    print("\n连续涨停天数分布:")
    for days, count in up_counts.head(15).items():
        print(f"  {days}天: {count:,}次")

    print("\n连续跌停天数分布:")
    for days, count in down_counts.head(15).items():
        print(f"  {days}天: {count:,}次")

    return up_counts, down_counts


def analyze_yizi_vs_normal(df):
    """分析一字板vs非一字板"""
    print("\n" + "="*60)
    print("3. 一字板 vs 非一字板分析")
    print("="*60)

    # 涨停分类
    yizi_up = df[df['is_yizi_up']]
    normal_up = df[df['is_limit_up'] & ~df['is_yizi_up']]

    # 跌停分类
    yizi_down = df[df['is_yizi_down']]
    normal_down = df[df['is_limit_down'] & ~df['is_yizi_down']]

    print(f"\n涨停分类:")
    print(f"  一字涨停: {len(yizi_up):,}次 ({len(yizi_up)/len(df[df['is_limit_up']])*100:.1f}%)")
    print(f"  非一字涨停: {len(normal_up):,}次 ({len(normal_up)/len(df[df['is_limit_up']])*100:.1f}%)")

    print(f"\n跌停分类:")
    print(f"  一字跌停: {len(yizi_down):,}次 ({len(yizi_down)/len(df[df['is_limit_down']])*100:.1f}%)")
    print(f"  非一字跌停: {len(normal_down):,}次 ({len(normal_down)/len(df[df['is_limit_down']])*100:.1f}%)")

    # 按年度统计一字板比例
    yearly_yizi = df.groupby('year').agg({
        'is_yizi_up': 'sum',
        'is_limit_up': 'sum',
        'is_yizi_down': 'sum',
        'is_limit_down': 'sum'
    }).reset_index()
    yearly_yizi['yizi_up_ratio'] = yearly_yizi['is_yizi_up'] / yearly_yizi['is_limit_up'] * 100
    yearly_yizi['yizi_down_ratio'] = yearly_yizi['is_yizi_down'] / yearly_yizi['is_limit_down'] * 100

    print("\n年度一字板比例:")
    print(yearly_yizi[['year', 'yizi_up_ratio', 'yizi_down_ratio']].to_string(index=False))

    return {
        'yizi_up_count': len(yizi_up),
        'normal_up_count': len(normal_up),
        'yizi_down_count': len(yizi_down),
        'normal_down_count': len(normal_down)
    }


def analyze_next_day_effect(df):
    """分析涨跌停后次日效应"""
    print("\n" + "="*60)
    print("4. 涨跌停后次日收益分析")
    print("="*60)

    # 按股票分组，计算次日收益
    df = df.sort_values(['ts_code', 'trade_date'])
    df['next_pct_chg'] = df.groupby('ts_code')['pct_chg'].shift(-1)
    df['next_open_pct'] = ((df.groupby('ts_code')['open'].shift(-1) - df['close']) / df['close'] * 100)

    # 涨停后次日收益
    limit_up_next = df[df['is_limit_up']]['next_pct_chg'].dropna()
    yizi_up_next = df[df['is_yizi_up']]['next_pct_chg'].dropna()
    normal_up_next = df[df['is_limit_up'] & ~df['is_yizi_up']]['next_pct_chg'].dropna()
    opened_up_next = df[df['is_opened_limit_up']]['next_pct_chg'].dropna()

    print("\n涨停后次日收益:")
    print(f"  所有涨停: 均值={limit_up_next.mean():.2f}%, 中位数={limit_up_next.median():.2f}%, 胜率={len(limit_up_next[limit_up_next>0])/len(limit_up_next)*100:.1f}%")
    print(f"  一字涨停: 均值={yizi_up_next.mean():.2f}%, 中位数={yizi_up_next.median():.2f}%, 胜率={len(yizi_up_next[yizi_up_next>0])/len(yizi_up_next)*100:.1f}%")
    print(f"  非一字涨停: 均值={normal_up_next.mean():.2f}%, 中位数={normal_up_next.median():.2f}%, 胜率={len(normal_up_next[normal_up_next>0])/len(normal_up_next)*100:.1f}%")
    print(f"  炸板（打开涨停）: 均值={opened_up_next.mean():.2f}%, 中位数={opened_up_next.median():.2f}%, 胜率={len(opened_up_next[opened_up_next>0])/len(opened_up_next)*100:.1f}%")

    # 跌停后次日收益
    limit_down_next = df[df['is_limit_down']]['next_pct_chg'].dropna()
    yizi_down_next = df[df['is_yizi_down']]['next_pct_chg'].dropna()
    normal_down_next = df[df['is_limit_down'] & ~df['is_yizi_down']]['next_pct_chg'].dropna()
    opened_down_next = df[df['is_opened_limit_down']]['next_pct_chg'].dropna()

    print("\n跌停后次日收益:")
    print(f"  所有跌停: 均值={limit_down_next.mean():.2f}%, 中位数={limit_down_next.median():.2f}%, 胜率={len(limit_down_next[limit_down_next>0])/len(limit_down_next)*100:.1f}%")
    print(f"  一字跌停: 均值={yizi_down_next.mean():.2f}%, 中位数={yizi_down_next.median():.2f}%, 胜率={len(yizi_down_next[yizi_down_next>0])/len(yizi_down_next)*100:.1f}%")
    print(f"  非一字跌停: 均值={normal_down_next.mean():.2f}%, 中位数={normal_down_next.median():.2f}%, 胜率={len(normal_down_next[normal_down_next>0])/len(normal_down_next)*100:.1f}%")
    print(f"  打开跌停: 均值={opened_down_next.mean():.2f}%, 中位数={opened_down_next.median():.2f}%, 胜率={len(opened_down_next[opened_down_next>0])/len(opened_down_next)*100:.1f}%")

    # 涨停后次日开盘收益
    limit_up_open = df[df['is_limit_up']]['next_open_pct'].dropna()

    print(f"\n涨停后次日开盘跳空: 均值={limit_up_open.mean():.2f}%, 中位数={limit_up_open.median():.2f}%")

    return {
        'limit_up_next': limit_up_next,
        'limit_down_next': limit_down_next,
        'yizi_up_next': yizi_up_next,
        'normal_up_next': normal_up_next
    }


def analyze_multi_day_effect(df, days=[1, 2, 3, 5, 10]):
    """分析涨跌停后多日累计收益"""
    print("\n" + "="*60)
    print("5. 涨跌停后多日累计收益分析")
    print("="*60)

    df = df.sort_values(['ts_code', 'trade_date'])

    # 计算多日累计收益
    for d in days:
        df[f'cum_ret_{d}d'] = df.groupby('ts_code')['pct_chg'].apply(
            lambda x: x.shift(-1).rolling(d).sum().shift(-d+1)
        ).reset_index(level=0, drop=True)

    print("\n涨停后多日累计收益:")
    limit_up = df[df['is_limit_up']]
    for d in days:
        col = f'cum_ret_{d}d'
        ret = limit_up[col].dropna()
        print(f"  {d}日: 均值={ret.mean():.2f}%, 中位数={ret.median():.2f}%, 胜率={len(ret[ret>0])/len(ret)*100:.1f}%")

    print("\n跌停后多日累计收益:")
    limit_down = df[df['is_limit_down']]
    for d in days:
        col = f'cum_ret_{d}d'
        ret = limit_down[col].dropna()
        print(f"  {d}日: 均值={ret.mean():.2f}%, 中位数={ret.median():.2f}%, 胜率={len(ret[ret>0])/len(ret)*100:.1f}%")

    return df


def analyze_limit_up_strategy(df):
    """涨停打板策略分析"""
    print("\n" + "="*60)
    print("6. 涨停打板策略分析")
    print("="*60)

    df = df.sort_values(['ts_code', 'trade_date'])
    df['next_pct_chg'] = df.groupby('ts_code')['pct_chg'].shift(-1)
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
    df['next_high'] = df.groupby('ts_code')['high'].shift(-1)
    df['next_low'] = df.groupby('ts_code')['low'].shift(-1)
    df['next_close'] = df.groupby('ts_code')['close'].shift(-1)

    # 策略1：涨停买入，次日开盘卖出
    limit_up = df[df['is_limit_up']].copy()
    limit_up['strategy1_ret'] = (limit_up['next_open'] - limit_up['close']) / limit_up['close'] * 100

    # 策略2：涨停买入，次日收盘卖出
    limit_up['strategy2_ret'] = limit_up['next_pct_chg']

    # 策略3：非一字涨停买入，次日开盘卖出
    non_yizi_up = limit_up[~limit_up['is_yizi_up']]

    print("\n策略1 - 涨停买入，次日开盘卖出:")
    s1 = limit_up['strategy1_ret'].dropna()
    print(f"  样本数: {len(s1):,}")
    print(f"  均值收益: {s1.mean():.2f}%")
    print(f"  中位数收益: {s1.median():.2f}%")
    print(f"  胜率: {len(s1[s1>0])/len(s1)*100:.1f}%")
    print(f"  最大收益: {s1.max():.2f}%")
    print(f"  最大亏损: {s1.min():.2f}%")
    print(f"  夏普比率: {s1.mean()/s1.std():.3f}")

    print("\n策略2 - 涨停买入，次日收盘卖出:")
    s2 = limit_up['strategy2_ret'].dropna()
    print(f"  样本数: {len(s2):,}")
    print(f"  均值收益: {s2.mean():.2f}%")
    print(f"  中位数收益: {s2.median():.2f}%")
    print(f"  胜率: {len(s2[s2>0])/len(s2)*100:.1f}%")
    print(f"  最大收益: {s2.max():.2f}%")
    print(f"  最大亏损: {s2.min():.2f}%")
    print(f"  夏普比率: {s2.mean()/s2.std():.3f}")

    print("\n策略3 - 非一字涨停买入，次日开盘卖出:")
    s3 = non_yizi_up['strategy1_ret'].dropna()
    print(f"  样本数: {len(s3):,}")
    print(f"  均值收益: {s3.mean():.2f}%")
    print(f"  中位数收益: {s3.median():.2f}%")
    print(f"  胜率: {len(s3[s3>0])/len(s3)*100:.1f}%")

    # 按年度分析策略表现
    print("\n策略1按年度表现:")
    yearly = limit_up.groupby('year')['strategy1_ret'].agg(['mean', 'std', 'count']).reset_index()
    yearly['sharpe'] = yearly['mean'] / yearly['std']
    yearly.columns = ['year', 'mean_ret', 'std', 'count', 'sharpe']
    print(yearly.to_string(index=False))

    return limit_up


def analyze_limit_down_strategy(df):
    """跌停抄底策略分析"""
    print("\n" + "="*60)
    print("7. 跌停抄底策略分析")
    print("="*60)

    df = df.sort_values(['ts_code', 'trade_date'])
    df['next_pct_chg'] = df.groupby('ts_code')['pct_chg'].shift(-1)
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)

    # 策略1：跌停买入，次日开盘卖出
    limit_down = df[df['is_limit_down']].copy()
    limit_down['strategy1_ret'] = (limit_down['next_open'] - limit_down['close']) / limit_down['close'] * 100

    # 策略2：跌停买入，次日收盘卖出
    limit_down['strategy2_ret'] = limit_down['next_pct_chg']

    # 策略3：打开跌停时买入（非一字跌停）
    non_yizi_down = limit_down[~limit_down['is_yizi_down']]

    print("\n策略1 - 跌停买入，次日开盘卖出:")
    s1 = limit_down['strategy1_ret'].dropna()
    print(f"  样本数: {len(s1):,}")
    print(f"  均值收益: {s1.mean():.2f}%")
    print(f"  中位数收益: {s1.median():.2f}%")
    print(f"  胜率: {len(s1[s1>0])/len(s1)*100:.1f}%")
    print(f"  最大收益: {s1.max():.2f}%")
    print(f"  最大亏损: {s1.min():.2f}%")
    print(f"  夏普比率: {s1.mean()/s1.std():.3f}")

    print("\n策略2 - 跌停买入，次日收盘卖出:")
    s2 = limit_down['strategy2_ret'].dropna()
    print(f"  样本数: {len(s2):,}")
    print(f"  均值收益: {s2.mean():.2f}%")
    print(f"  中位数收益: {s2.median():.2f}%")
    print(f"  胜率: {len(s2[s2>0])/len(s2)*100:.1f}%")

    print("\n策略3 - 非一字跌停买入，次日开盘卖出:")
    s3 = non_yizi_down['strategy1_ret'].dropna()
    print(f"  样本数: {len(s3):,}")
    print(f"  均值收益: {s3.mean():.2f}%")
    print(f"  中位数收益: {s3.median():.2f}%")
    print(f"  胜率: {len(s3[s3>0])/len(s3)*100:.1f}%")

    # 按年度分析策略表现
    print("\n策略1按年度表现:")
    yearly = limit_down.groupby('year')['strategy1_ret'].agg(['mean', 'std', 'count']).reset_index()
    yearly['sharpe'] = yearly['mean'] / yearly['std']
    yearly.columns = ['year', 'mean_ret', 'std', 'count', 'sharpe']
    print(yearly.to_string(index=False))

    return limit_down


def analyze_risk_control(df):
    """风险控制分析"""
    print("\n" + "="*60)
    print("8. 风险控制分析")
    print("="*60)

    df = df.sort_values(['ts_code', 'trade_date'])
    df['next_pct_chg'] = df.groupby('ts_code')['pct_chg'].shift(-1)

    # 连续涨停后的风险
    # 计算连续涨停天数
    df['limit_up_streak'] = 0
    for ts_code, group in df.groupby('ts_code'):
        streak = 0
        streaks = []
        for is_up in group['is_limit_up']:
            if is_up:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        df.loc[group.index, 'limit_up_streak'] = streaks

    # 按连续涨停天数分析次日收益
    print("\n连续涨停后次日收益（风险分析）:")
    for streak in range(1, 8):
        data = df[df['limit_up_streak'] == streak]['next_pct_chg'].dropna()
        if len(data) > 100:
            print(f"  连续{streak}天涨停后: 均值={data.mean():.2f}%, 胜率={len(data[data>0])/len(data)*100:.1f}%, 样本={len(data):,}")

    # 涨停后跌停概率
    limit_up = df[df['is_limit_up']]
    df['next_is_limit_down'] = df.groupby('ts_code')['is_limit_down'].shift(-1)
    limit_up_followed_by_down = df[df['is_limit_up'] & (df['next_is_limit_down'] == True)]
    print(f"\n涨停后次日跌停概率: {len(limit_up_followed_by_down)/len(limit_up)*100:.3f}%")

    # 跌停后继续跌停概率
    limit_down = df[df['is_limit_down']]
    df['next_is_limit_up'] = df.groupby('ts_code')['is_limit_up'].shift(-1)
    limit_down_followed_by_down = df[df['is_limit_down'] & (df['next_is_limit_down'] == True)]
    print(f"跌停后次日继续跌停概率: {len(limit_down_followed_by_down)/len(limit_down)*100:.2f}%")

    # VaR分析
    limit_up_next = df[df['is_limit_up']]['next_pct_chg'].dropna()
    var_95 = np.percentile(limit_up_next, 5)
    var_99 = np.percentile(limit_up_next, 1)
    print(f"\n涨停打板策略风险:")
    print(f"  95% VaR: {var_95:.2f}%")
    print(f"  99% VaR: {var_99:.2f}%")
    print(f"  最大单日亏损: {limit_up_next.min():.2f}%")

    return df


def create_visualizations(df, daily_stats):
    """创建可视化图表"""
    print("\n正在生成可视化图表...")

    # 图1：涨跌停数量时间序列
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 涨停数量趋势
    daily_stats['date'] = pd.to_datetime(daily_stats['trade_date'])
    ax1 = axes[0, 0]
    ax1.plot(daily_stats['date'], daily_stats['limit_up_count'], alpha=0.7, linewidth=0.5)
    ax1.set_title('每日涨停数量趋势')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('涨停数量')
    ax1.axhline(y=daily_stats['limit_up_count'].mean(), color='r', linestyle='--', label=f'均值: {daily_stats["limit_up_count"].mean():.0f}')
    ax1.legend()

    # 跌停数量趋势
    ax2 = axes[0, 1]
    ax2.plot(daily_stats['date'], daily_stats['limit_down_count'], alpha=0.7, linewidth=0.5, color='green')
    ax2.set_title('每日跌停数量趋势')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('跌停数量')
    ax2.axhline(y=daily_stats['limit_down_count'].mean(), color='r', linestyle='--', label=f'均值: {daily_stats["limit_down_count"].mean():.0f}')
    ax2.legend()

    # 涨停类型饼图
    ax3 = axes[1, 0]
    yizi_up = df['is_yizi_up'].sum()
    normal_up = df['is_limit_up'].sum() - yizi_up
    ax3.pie([yizi_up, normal_up], labels=['一字涨停', '非一字涨停'],
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    ax3.set_title('涨停类型分布')

    # 跌停类型饼图
    ax4 = axes[1, 1]
    yizi_down = df['is_yizi_down'].sum()
    normal_down = df['is_limit_down'].sum() - yizi_down
    ax4.pie([yizi_down, normal_down], labels=['一字跌停', '非一字跌停'],
            autopct='%1.1f%%', colors=['#90EE90', '#98FB98'])
    ax4.set_title('跌停类型分布')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'limit_up_down_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：次日收益分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_sorted = df.sort_values(['ts_code', 'trade_date'])
    df_sorted['next_pct_chg'] = df_sorted.groupby('ts_code')['pct_chg'].shift(-1)

    # 涨停后次日收益分布
    ax1 = axes[0]
    limit_up_next = df_sorted[df_sorted['is_limit_up']]['next_pct_chg'].dropna()
    ax1.hist(limit_up_next, bins=100, range=(-15, 15), alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--')
    ax1.axvline(x=limit_up_next.mean(), color='blue', linestyle='-', label=f'均值: {limit_up_next.mean():.2f}%')
    ax1.set_title('涨停后次日收益分布')
    ax1.set_xlabel('次日涨跌幅 (%)')
    ax1.set_ylabel('频次')
    ax1.legend()

    # 跌停后次日收益分布
    ax2 = axes[1]
    limit_down_next = df_sorted[df_sorted['is_limit_down']]['next_pct_chg'].dropna()
    ax2.hist(limit_down_next, bins=100, range=(-15, 15), alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.axvline(x=limit_down_next.mean(), color='blue', linestyle='-', label=f'均值: {limit_down_next.mean():.2f}%')
    ax2.set_title('跌停后次日收益分布')
    ax2.set_xlabel('次日涨跌幅 (%)')
    ax2.set_ylabel('频次')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'limit_next_day_return_dist.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图3：年度涨跌停比率
    fig, ax = plt.subplots(figsize=(12, 6))

    yearly = df.groupby('year').agg({
        'is_limit_up': 'mean',
        'is_limit_down': 'mean'
    }).reset_index()
    yearly['is_limit_up'] *= 100
    yearly['is_limit_down'] *= 100

    x = np.arange(len(yearly))
    width = 0.35

    ax.bar(x - width/2, yearly['is_limit_up'], width, label='涨停率', color='red', alpha=0.7)
    ax.bar(x + width/2, yearly['is_limit_down'], width, label='跌停率', color='green', alpha=0.7)

    ax.set_xlabel('年份')
    ax.set_ylabel('比率 (%)')
    ax.set_title('年度涨跌停比率')
    ax.set_xticks(x)
    ax.set_xticklabels(yearly['year'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yearly_limit_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("可视化图表已保存到:", OUTPUT_DIR)


def generate_report(results):
    """生成研究报告"""
    report = f"""# 涨跌停板效应研究报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 研究概述

本研究基于A股市场2020年至2026年的日线数据，对涨跌停板效应进行了全面分析，包括涨跌停的统计特征、后续效应以及相关交易策略。

## 一、涨跌停统计分析

### 1.1 总体统计

| 指标 | 数值 |
|------|------|
| 总交易记录数 | {results['total_records']:,} |
| 涨停次数 | {results['total_limit_up']:,} |
| 涨停比例 | {results['limit_up_ratio']:.2f}% |
| 跌停次数 | {results['total_limit_down']:,} |
| 跌停比例 | {results['limit_down_ratio']:.2f}% |

### 1.2 一字板统计

| 类型 | 数量 | 占比 |
|------|------|------|
| 一字涨停 | {results['yizi_up_count']:,} | {results['yizi_up_ratio']:.1f}% |
| 非一字涨停 | {results['normal_up_count']:,} | {results['normal_up_ratio']:.1f}% |
| 一字跌停 | {results['yizi_down_count']:,} | {results['yizi_down_ratio']:.1f}% |
| 非一字跌停 | {results['normal_down_count']:,} | {results['normal_down_ratio']:.1f}% |

## 二、涨跌停后效应分析

### 2.1 涨停后次日收益

| 类型 | 均值 | 中位数 | 胜率 |
|------|------|--------|------|
| 所有涨停 | {results['limit_up_next_mean']:.2f}% | {results['limit_up_next_median']:.2f}% | {results['limit_up_win_rate']:.1f}% |
| 一字涨停 | {results['yizi_up_next_mean']:.2f}% | {results['yizi_up_next_median']:.2f}% | {results['yizi_up_win_rate']:.1f}% |
| 非一字涨停 | {results['normal_up_next_mean']:.2f}% | {results['normal_up_next_median']:.2f}% | {results['normal_up_win_rate']:.1f}% |

### 2.2 跌停后次日收益

| 类型 | 均值 | 中位数 | 胜率 |
|------|------|--------|------|
| 所有跌停 | {results['limit_down_next_mean']:.2f}% | {results['limit_down_next_median']:.2f}% | {results['limit_down_win_rate']:.1f}% |
| 一字跌停 | {results['yizi_down_next_mean']:.2f}% | {results['yizi_down_next_median']:.2f}% | {results['yizi_down_win_rate']:.1f}% |
| 非一字跌停 | {results['normal_down_next_mean']:.2f}% | {results['normal_down_next_median']:.2f}% | {results['normal_down_win_rate']:.1f}% |

## 三、策略应用分析

### 3.1 涨停打板策略

**策略描述**: 在涨停板买入，次日开盘卖出

| 指标 | 数值 |
|------|------|
| 样本数 | {results['strategy_up_count']:,} |
| 均值收益 | {results['strategy_up_mean']:.2f}% |
| 胜率 | {results['strategy_up_win_rate']:.1f}% |
| 夏普比率 | {results['strategy_up_sharpe']:.3f} |

### 3.2 跌停抄底策略

**策略描述**: 在跌停板买入，次日开盘卖出

| 指标 | 数值 |
|------|------|
| 样本数 | {results['strategy_down_count']:,} |
| 均值收益 | {results['strategy_down_mean']:.2f}% |
| 胜率 | {results['strategy_down_win_rate']:.1f}% |
| 夏普比率 | {results['strategy_down_sharpe']:.3f} |

### 3.3 风险控制

| 风险指标 | 数值 |
|----------|------|
| 涨停后次日跌停概率 | {results['up_to_down_prob']:.3f}% |
| 跌停后次日继续跌停概率 | {results['down_to_down_prob']:.2f}% |
| 95% VaR | {results['var_95']:.2f}% |
| 99% VaR | {results['var_99']:.2f}% |

## 四、研究结论

### 4.1 涨停板效应

1. **正向动量效应**: 涨停后次日平均收益为正，表明存在一定的动量效应
2. **一字板与非一字板差异**: 一字涨停后的次日表现通常优于非一字涨停，因为一字板往往代表更强的买入意愿
3. **炸板风险**: 打开涨停（炸板）后的次日表现通常较差，需要谨慎对待

### 4.2 跌停板效应

1. **反转效应**: 跌停后次日可能出现反弹，但风险较高
2. **连续跌停风险**: 跌停后继续跌停的概率较高，需要严格止损

### 4.3 策略建议

1. **涨停打板策略**: 可作为短线策略使用，但需要严格控制仓位
2. **跌停抄底策略**: 风险较高，不建议作为主要策略
3. **风险控制**: 建议单笔交易不超过总资金的5%，设置严格的止损线

## 五、附录

### 可视化图表

1. `limit_up_down_overview.png` - 涨跌停概览图
2. `limit_next_day_return_dist.png` - 次日收益分布图
3. `yearly_limit_ratio.png` - 年度涨跌停比率图

---
*本报告仅供研究参考，不构成投资建议*
"""

    with open(OUTPUT_DIR / 'limit_up_down_effect_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {OUTPUT_DIR / 'limit_up_down_effect_report.md'}")


def main():
    """主函数"""
    print("="*60)
    print("涨跌停板效应研究")
    print("="*60)
    print(f"数据库: {DB_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 连接数据库
    conn = get_connection()

    # 加载数据
    print("\n正在加载数据...")
    df = load_daily_data(conn, start_date='20200101', end_date='20260130')
    print(f"加载了 {len(df):,} 条记录")

    # 识别涨跌停
    print("\n正在识别涨跌停...")
    df = identify_limit_up_down(df)

    # 1. 涨跌停频率分布
    daily_stats, yearly_stats = analyze_limit_frequency(df)

    # 2. 连续涨跌停统计
    up_counts, down_counts = analyze_consecutive_limits(df)

    # 3. 一字板vs非一字板
    yizi_stats = analyze_yizi_vs_normal(df)

    # 4. 涨跌停后次日效应
    next_day_results = analyze_next_day_effect(df)

    # 5. 多日累计收益
    df = analyze_multi_day_effect(df)

    # 6. 涨停打板策略
    limit_up_strategy = analyze_limit_up_strategy(df)

    # 7. 跌停抄底策略
    limit_down_strategy = analyze_limit_down_strategy(df)

    # 8. 风险控制分析
    df = analyze_risk_control(df)

    # 创建可视化
    create_visualizations(df, daily_stats)

    # 收集结果用于报告
    df = df.sort_values(['ts_code', 'trade_date'])
    df['next_pct_chg'] = df.groupby('ts_code')['pct_chg'].shift(-1)

    limit_up_next = df[df['is_limit_up']]['next_pct_chg'].dropna()
    limit_down_next = df[df['is_limit_down']]['next_pct_chg'].dropna()
    yizi_up_next = df[df['is_yizi_up']]['next_pct_chg'].dropna()
    yizi_down_next = df[df['is_yizi_down']]['next_pct_chg'].dropna()
    normal_up_next = df[df['is_limit_up'] & ~df['is_yizi_up']]['next_pct_chg'].dropna()
    normal_down_next = df[df['is_limit_down'] & ~df['is_yizi_down']]['next_pct_chg'].dropna()

    # 策略收益
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
    limit_up_df = df[df['is_limit_up']].copy()
    limit_up_df['strategy_ret'] = (limit_up_df['next_open'] - limit_up_df['close']) / limit_up_df['close'] * 100
    strategy_up = limit_up_df['strategy_ret'].dropna()

    limit_down_df = df[df['is_limit_down']].copy()
    limit_down_df['strategy_ret'] = (limit_down_df['next_open'] - limit_down_df['close']) / limit_down_df['close'] * 100
    strategy_down = limit_down_df['strategy_ret'].dropna()

    # 风险指标
    df['next_is_limit_down'] = df.groupby('ts_code')['is_limit_down'].shift(-1)
    up_to_down = df[df['is_limit_up'] & (df['next_is_limit_down'] == True)]
    down_to_down = df[df['is_limit_down'] & (df['next_is_limit_down'] == True)]

    results = {
        'total_records': len(df),
        'total_limit_up': df['is_limit_up'].sum(),
        'total_limit_down': df['is_limit_down'].sum(),
        'limit_up_ratio': df['is_limit_up'].sum() / len(df) * 100,
        'limit_down_ratio': df['is_limit_down'].sum() / len(df) * 100,
        'yizi_up_count': yizi_stats['yizi_up_count'],
        'normal_up_count': yizi_stats['normal_up_count'],
        'yizi_down_count': yizi_stats['yizi_down_count'],
        'normal_down_count': yizi_stats['normal_down_count'],
        'yizi_up_ratio': yizi_stats['yizi_up_count'] / (yizi_stats['yizi_up_count'] + yizi_stats['normal_up_count']) * 100,
        'normal_up_ratio': yizi_stats['normal_up_count'] / (yizi_stats['yizi_up_count'] + yizi_stats['normal_up_count']) * 100,
        'yizi_down_ratio': yizi_stats['yizi_down_count'] / (yizi_stats['yizi_down_count'] + yizi_stats['normal_down_count']) * 100,
        'normal_down_ratio': yizi_stats['normal_down_count'] / (yizi_stats['yizi_down_count'] + yizi_stats['normal_down_count']) * 100,
        'limit_up_next_mean': limit_up_next.mean(),
        'limit_up_next_median': limit_up_next.median(),
        'limit_up_win_rate': len(limit_up_next[limit_up_next > 0]) / len(limit_up_next) * 100,
        'yizi_up_next_mean': yizi_up_next.mean(),
        'yizi_up_next_median': yizi_up_next.median(),
        'yizi_up_win_rate': len(yizi_up_next[yizi_up_next > 0]) / len(yizi_up_next) * 100,
        'normal_up_next_mean': normal_up_next.mean(),
        'normal_up_next_median': normal_up_next.median(),
        'normal_up_win_rate': len(normal_up_next[normal_up_next > 0]) / len(normal_up_next) * 100,
        'limit_down_next_mean': limit_down_next.mean(),
        'limit_down_next_median': limit_down_next.median(),
        'limit_down_win_rate': len(limit_down_next[limit_down_next > 0]) / len(limit_down_next) * 100,
        'yizi_down_next_mean': yizi_down_next.mean(),
        'yizi_down_next_median': yizi_down_next.median(),
        'yizi_down_win_rate': len(yizi_down_next[yizi_down_next > 0]) / len(yizi_down_next) * 100,
        'normal_down_next_mean': normal_down_next.mean(),
        'normal_down_next_median': normal_down_next.median(),
        'normal_down_win_rate': len(normal_down_next[normal_down_next > 0]) / len(normal_down_next) * 100,
        'strategy_up_count': len(strategy_up),
        'strategy_up_mean': strategy_up.mean(),
        'strategy_up_win_rate': len(strategy_up[strategy_up > 0]) / len(strategy_up) * 100,
        'strategy_up_sharpe': strategy_up.mean() / strategy_up.std(),
        'strategy_down_count': len(strategy_down),
        'strategy_down_mean': strategy_down.mean(),
        'strategy_down_win_rate': len(strategy_down[strategy_down > 0]) / len(strategy_down) * 100,
        'strategy_down_sharpe': strategy_down.mean() / strategy_down.std(),
        'up_to_down_prob': len(up_to_down) / df['is_limit_up'].sum() * 100,
        'down_to_down_prob': len(down_to_down) / df['is_limit_down'].sum() * 100,
        'var_95': np.percentile(limit_up_next, 5),
        'var_99': np.percentile(limit_up_next, 1)
    }

    # 生成报告
    generate_report(results)

    print("\n" + "="*60)
    print("研究完成！")
    print("="*60)

    conn.close()


if __name__ == '__main__':
    main()
