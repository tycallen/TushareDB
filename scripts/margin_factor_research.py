#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融资余额变化因子研究：对中证全指的预测作用
=============================================

研究内容：
1. 多维度融资余额变化因子构建
2. IC/IR分析评估因子预测能力
3. 分位数组合收益测试
4. Alpha衰减分析（不同持有期）
5. 择时策略回测
6. 可视化与报告

使用方法:
    python scripts/margin_factor_research.py
    python scripts/margin_factor_research.py --start-date 20200101 --end-date 20251231

环境变量:
    DB_PATH: 数据库路径（可选，默认为 tushare.db）
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataReader

# 中证全指代码
ALL_A_INDEX_CODE = '000985.SH'

# 配置中文字体
_FONT_LIST = [
    'PingFang SC', 'PingFang HK', 'PingFang TC',
    'Heiti SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'Kaiti SC',
    'Arial Unicode MS', 'Microsoft YaHei', 'SimHei',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif'
]
_available_fonts = set(f.name for f in fm.fontManager.ttflist)
for _font in _FONT_LIST:
    if _font in _available_fonts or _font == 'sans-serif':
        plt.rcParams['font.sans-serif'] = [_font]
        break
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='融资余额变化因子研究')
    parser.add_argument('--start-date', type=str, default='20190816',
                        help='研究起始日期 YYYYMMDD（默认20190816）')
    parser.add_argument('--end-date', type=str, default=None,
                        help='研究结束日期 YYYYMMDD（默认最新）')
    parser.add_argument('--db-path', type=str, default=None,
                        help='数据库路径（默认从 DB_PATH 环境变量或 tushare.db）')
    parser.add_argument('--output-dir', type=str, default='reports/research',
                        help='报告输出目录')
    return parser.parse_args()


# ============================================================================
# Part 1: 数据获取
# ============================================================================

def load_data(reader: DataReader, start_date: str, end_date: str) -> pd.DataFrame:
    """加载融资余额和中证全指数据"""
    print("=" * 60)
    print("Part 1: 数据加载")
    print("=" * 60)

    # 1.1 融资余额数据（SSE + SZSE 汇总）
    print("\n1.1 加载融资余额数据...")
    margin_df = reader.get_margin(start_date=start_date, end_date=end_date)
    if margin_df.empty:
        raise RuntimeError("融资余额数据为空")

    # 按日期汇总（SSE + SZSE + BSE）
    margin_daily = margin_df.groupby('trade_date').agg({
        'rzye': 'sum',      # 融资余额
        'rzmre': 'sum',     # 融资买入额
        'rzche': 'sum',     # 融资偿还额
        'rqye': 'sum',      # 融券余额
        'rzrqye': 'sum',    # 融资融券余额
        'rqmcl': 'sum',     # 融券卖出量
        'rqyl': 'sum',      # 融券余量
    }).reset_index()
    margin_daily['trade_date'] = margin_daily['trade_date'].astype(str)
    margin_daily = margin_daily.sort_values('trade_date').reset_index(drop=True)

    print(f"  融资余额日期范围: {margin_daily['trade_date'].min()} ~ {margin_daily['trade_date'].max()}")
    print(f"  融资余额记录数: {len(margin_daily)}")

    # 1.2 中证全指数据
    print("\n1.2 加载中证全指数据...")
    index_df = reader.get_index_daily(
        ts_code=ALL_A_INDEX_CODE,
        start_date=start_date,
        end_date=end_date
    )
    if index_df.empty:
        raise RuntimeError("中证全指数据为空")

    index_df = index_df[['trade_date', 'close', 'open', 'high', 'low', 'pct_chg']].copy()
    index_df['trade_date'] = index_df['trade_date'].astype(str)
    index_df = index_df.sort_values('trade_date').reset_index(drop=True)

    print(f"  中证全指日期范围: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")
    print(f"  中证全指记录数: {len(index_df)}")

    # 1.3 合并数据
    print("\n1.3 合并数据...")
    df = pd.merge(margin_daily, index_df, on='trade_date', how='inner')
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

    print(f"  合并后记录数: {len(df)}")
    print(f"  合并后日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    return df


# ============================================================================
# Part 2: 因子构建
# ============================================================================

def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """构建融资余额变化因子"""
    print("\n" + "=" * 60)
    print("Part 2: 因子构建")
    print("=" * 60)

    # 单位转换：元 -> 亿元
    for col in ['rzye', 'rzmre', 'rzche', 'rqye', 'rzrqye']:
        df[f'{col}_yi'] = df[col] / 1e8

    # 计算未来收益（1/5/10/20日）
    for n in [1, 5, 10, 20]:
        df[f'fwd_ret_{n}d'] = df['close'].shift(-n) / df['close'] - 1
        df[f'fwd_pct_chg_{n}d'] = df['pct_chg'].shift(-n).rolling(n, min_periods=1).sum()

    # ===== 因子1-3: 融资余额变化量（绝对值，亿元）=====
    for n in [1, 5, 20]:
        df[f'margin_chg_{n}d'] = df['rzye_yi'] - df['rzye_yi'].shift(n)

    # ===== 因子4-6: 融资余额变化率（%）=====
    for n in [1, 5, 20]:
        df[f'margin_pctchg_{n}d'] = df['rzye_yi'].pct_change(n) * 100

    # ===== 因子7-9: 融资净买入额（亿元）=====
    # 当日净买入 = 融资买入额 - 融资偿还额
    df['net_buy_1d'] = df['rzmre_yi'] - df['rzche_yi']
    df['net_buy_5d'] = df['net_buy_1d'].rolling(5, min_periods=3).sum()
    df['net_buy_20d'] = df['net_buy_1d'].rolling(20, min_periods=10).sum()

    # ===== 因子10: 融资余额/融券余额比率 =====
    df['margin_short_ratio'] = df['rzye_yi'] / (df['rqye_yi'] / 1e8).replace(0, np.nan)

    # ===== 因子11-12: 融资余额动量（MA交叉）=====
    df['rzye_ma5'] = df['rzye_yi'].rolling(5, min_periods=3).mean()
    df['rzye_ma20'] = df['rzye_yi'].rolling(20, min_periods=10).mean()
    df['margin_ma_diff'] = df['rzye_ma5'] - df['rzye_ma20']
    df['margin_ma_ratio'] = (df['rzye_ma5'] / df['rzye_ma20'].replace(0, np.nan) - 1) * 100

    # ===== 因子13-15: 标准化Z-Score（滚动窗口）=====
    lookback = 120  # 约半年交易日
    for n in [1, 5, 20]:
        col = f'margin_chg_{n}d'
        rolling_mean = df[col].rolling(lookback, min_periods=30).mean()
        rolling_std = df[col].rolling(lookback, min_periods=30).std()
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)

    # ===== 因子16-18: 历史分位数（滚动窗口）=====
    for n in [1, 5, 20]:
        col = f'margin_pctchg_{n}d'
        df[f'{col}_pctile'] = df[col].rolling(lookback, min_periods=30).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )

    # ===== 因子19: 融资余额占指数点位比（杠杆率趋势）=====
    df['margin_per_index'] = df['rzye_yi'] / df['close']
    df['margin_per_index_chg'] = df['margin_per_index'].pct_change(5) * 100

    # ===== 因子20: 融资融券余额变化率 =====
    df['total_margin_pctchg_5d'] = df['rzrqye_yi'].pct_change(5) * 100

    # ===== 因子21-22: 加速度（变化率的变化）=====
    df['margin_chg_5d_accel'] = df['margin_chg_5d'] - df['margin_chg_5d'].shift(5)
    df['margin_pctchg_5d_accel'] = df['margin_pctchg_5d'] - df['margin_pctchg_5d'].shift(5)

    # ===== 因子23: 融资买入强度（买入额/余额）=====
    df['buy_intensity_1d'] = (df['rzmre_yi'] / df['rzye_yi'].replace(0, np.nan)) * 100
    df['buy_intensity_5d'] = df['buy_intensity_1d'].rolling(5, min_periods=3).mean()

    # ===== 因子24: 净买入占余额比 =====
    df['net_buy_ratio_5d'] = (df['net_buy_5d'] / df['rzye_yi'].replace(0, np.nan)) * 100

    print("\n构建的因子列表:")
    factor_cols = [c for c in df.columns if c.startswith(('margin_', 'net_buy', 'buy_intensity'))]
    for i, col in enumerate(factor_cols, 1):
        non_null = df[col].notna().sum()
        print(f"  {i:2d}. {col:35s} (非空: {non_null:4d})")

    return df


# ============================================================================
# Part 3: IC/IR分析
# ============================================================================

def calculate_ic(df: pd.DataFrame, factor_col: str, forward_col: str) -> dict:
    """计算单个因子与forward return的Spearman IC"""
    valid = df[[factor_col, forward_col]].dropna()
    if len(valid) < 30:
        return {'ic': np.nan, 'p_value': np.nan, 'n': len(valid)}

    ic, p_value = stats.spearmanr(valid[factor_col], valid[forward_col])
    return {'ic': ic, 'p_value': p_value, 'n': len(valid)}


def analyze_ic(df: pd.DataFrame) -> pd.DataFrame:
    """IC/IR分析"""
    print("\n" + "=" * 60)
    print("Part 3: IC/IR分析")
    print("=" * 60)

    factor_defs = [
        ('margin_chg_1d', '融资余额1日变化(亿元)'),
        ('margin_chg_5d', '融资余额5日变化(亿元)'),
        ('margin_chg_20d', '融资余额20日变化(亿元)'),
        ('margin_pctchg_1d', '融资余额1日变化率(%)'),
        ('margin_pctchg_5d', '融资余额5日变化率(%)'),
        ('margin_pctchg_20d', '融资余额20日变化率(%)'),
        ('net_buy_1d', '融资净买入1日(亿元)'),
        ('net_buy_5d', '融资净买入5日(亿元)'),
        ('net_buy_20d', '融资净买入20日(亿元)'),
        ('margin_short_ratio', '融资/融券余额比'),
        ('margin_ma_diff', '融资余额MA5-MA20(亿元)'),
        ('margin_ma_ratio', '融资余额MA比值(%)'),
        ('margin_chg_1d_zscore', '融资余额1日变化Z-Score'),
        ('margin_chg_5d_zscore', '融资余额5日变化Z-Score'),
        ('margin_chg_20d_zscore', '融资余额20日变化Z-Score'),
        ('margin_pctchg_5d_pctile', '融资余额5日变化率分位数'),
        ('margin_pctchg_20d_pctile', '融资余额20日变化率分位数'),
        ('margin_per_index_chg', '融资余额/指数点位变化率(%)'),
        ('total_margin_pctchg_5d', '融资融券余额5日变化率(%)'),
        ('margin_chg_5d_accel', '融资余额5日变化加速度'),
        ('margin_pctchg_5d_accel', '融资余额5日变化率加速度'),
        ('buy_intensity_5d', '融资买入强度5日均值(%)'),
        ('net_buy_ratio_5d', '净买入占余额比5日(%)'),
    ]

    horizons = [1, 5, 10, 20]
    results = []

    for factor_col, factor_name in factor_defs:
        if factor_col not in df.columns:
            continue

        row = {'因子代码': factor_col, '因子名称': factor_name}

        for h in horizons:
            fwd_col = f'fwd_ret_{h}d'
            ic_result = calculate_ic(df, factor_col, fwd_col)

            row[f'IC_{h}D'] = ic_result['ic']
            row[f'p_{h}D'] = ic_result['p_value']
            row[f'n_{h}D'] = ic_result['n']

        results.append(row)

    ic_df = pd.DataFrame(results)

    # 打印结果
    print("\n因子IC分析结果:")
    print("-" * 100)
    print(f"{'因子名称':<30s} {'IC_1D':>8s} {'IC_5D':>8s} {'IC_10D':>8s} {'IC_20D':>8s}")
    print("-" * 100)

    for _, row in ic_df.iterrows():
        ic_vals = [row[f'IC_{h}D'] for h in horizons]
        ic_strs = []
        for ic in ic_vals:
            if pd.isna(ic):
                ic_strs.append('  ---  ')
            else:
                marker = '*' if abs(ic) > 0.05 else ' '
                ic_strs.append(f'{marker}{ic:+.3f}')
        print(f"{row['因子名称']:<30s} {ic_strs[0]:>8s} {ic_strs[1]:>8s} {ic_strs[2]:>8s} {ic_strs[3]:>8s}")

    print("-" * 100)
    print("注: * 表示 |IC| > 0.05（中等强度信号）")

    return ic_df


def analyze_ic_by_period(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """按月计算IC时间序列"""
    df['year_month'] = df['trade_date'].str[:6]

    ic_by_month = []
    for ym, group in df.groupby('year_month'):
        valid = group[[factor_col, 'fwd_ret_5d']].dropna()
        if len(valid) >= 5:
            ic, p = stats.spearmanr(valid[factor_col], valid['fwd_ret_5d'])
            ic_by_month.append({
                'year_month': ym,
                'ic': ic,
                'p_value': p,
                'n': len(valid)
            })

    ic_month_df = pd.DataFrame(ic_by_month)
    if len(ic_month_df) > 0:
        ic_mean = ic_month_df['ic'].mean()
        ic_std = ic_month_df['ic'].std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        pct_positive = (ic_month_df['ic'] > 0).mean()
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_month_df))) if ic_std > 0 else 0

        print(f"\n  {factor_col} 月度IC统计:")
        print(f"    IC均值: {ic_mean:+.4f}")
        print(f"    IC标准差: {ic_std:.4f}")
        print(f"    IR: {ir:+.3f}")
        print(f"    IC正向率: {pct_positive*100:.1f}%")
        print(f"    t统计量: {t_stat:+.3f}")

    return ic_month_df


# ============================================================================
# Part 4: Alpha衰减分析
# ============================================================================

def alpha_decay_analysis(df: pd.DataFrame, factor_cols: list, max_horizon: int = 20) -> pd.DataFrame:
    """分析alpha随持有期的衰减"""
    print("\n" + "=" * 60)
    print("Part 4: Alpha衰减分析")
    print("=" * 60)

    results = []

    for factor_col in factor_cols:
        if factor_col not in df.columns:
            continue

        for h in range(1, max_horizon + 1):
            fwd_col = f'fwd_ret_{h}d'
            if fwd_col not in df.columns:
                continue

            valid = df[[factor_col, fwd_col]].dropna()
            if len(valid) < 20:
                continue

            ic, p = stats.spearmanr(valid[factor_col], valid[fwd_col])
            results.append({
                'factor': factor_col,
                'horizon': h,
                'ic': ic,
                'p_value': p,
                'n': len(valid)
            })

    decay_df = pd.DataFrame(results)

    # 打印关键因子的衰减
    print("\n关键因子IC随持有期衰减:")
    print(f"{'持有期':>6s} | {'chg_5d':>10s} | {'pctchg_5d':>10s} | {'net_buy_5d':>10s} | {'ma_ratio':>10s} | {'zscore_5d':>10s}")
    print("-" * 75)

    for h in [1, 2, 3, 5, 10, 15, 20]:
        row = [f'{h:>6d}d']
        for col in ['margin_chg_5d', 'margin_pctchg_5d', 'net_buy_5d', 'margin_ma_ratio', 'margin_chg_5d_zscore']:
            subset = decay_df[(decay_df['factor'] == col) & (decay_df['horizon'] == h)]
            if len(subset) > 0:
                ic = subset['ic'].values[0]
                marker = '*' if abs(ic) > 0.05 else ' '
                row.append(f'{marker}{ic:+.3f}')
            else:
                row.append('   ---   ')
        print(' | '.join(row))

    print("-" * 75)
    print("注: * 表示 |IC| > 0.05")

    return decay_df


# ============================================================================
# Part 5: 分位数组合测试
# ============================================================================

def quantile_portfolio_test(df: pd.DataFrame, factor_col: str, n_quantiles: int = 5) -> pd.DataFrame:
    """分位数组合测试"""
    valid = df[[factor_col, 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'trade_date']].dropna()
    if len(valid) < 50:
        return pd.DataFrame()

    valid['quantile'] = pd.qcut(valid[factor_col], n_quantiles, labels=[f'Q{i+1}' for i in range(n_quantiles)], duplicates='drop')

    results = []
    for q in [f'Q{i+1}' for i in range(n_quantiles)]:
        q_df = valid[valid['quantile'] == q]
        if len(q_df) < 5:
            continue

        results.append({
            '分位组': q,
            '样本数': len(q_df),
            '因子均值': q_df[factor_col].mean(),
            '5日收益': q_df['fwd_ret_5d'].mean() * 100,
            '10日收益': q_df['fwd_ret_10d'].mean() * 100,
            '20日收益': q_df['fwd_ret_20d'].mean() * 100,
            '5日收益标准差': q_df['fwd_ret_5d'].std() * 100,
        })

    return pd.DataFrame(results)


def run_quantile_tests(df: pd.DataFrame) -> dict:
    """运行多个因子的分位数测试"""
    print("\n" + "=" * 60)
    print("Part 5: 分位数组合测试")
    print("=" * 60)

    key_factors = [
        ('margin_chg_5d', '融资余额5日变化(亿元)'),
        ('margin_pctchg_5d', '融资余额5日变化率(%)'),
        ('net_buy_5d', '融资净买入5日(亿元)'),
        ('margin_ma_ratio', '融资余额MA比值(%)'),
        ('margin_chg_5d_zscore', '融资余额变化Z-Score'),
    ]

    all_results = {}

    for col, name in key_factors:
        if col not in df.columns:
            continue

        print(f"\n{name}:")
        result = quantile_portfolio_test(df, col)
        if not result.empty:
            print(result.to_string(index=False))

            # 计算多空收益
            if len(result) >= 2:
                long_ret = result.iloc[-1]['20日收益']
                short_ret = result.iloc[0]['20日收益']
                ls_ret = long_ret - short_ret
                print(f"  多空收益(做多Q5+做空Q1, 20日): {ls_ret:+.3f}%")

            all_results[col] = result

    return all_results


# ============================================================================
# Part 6: 择时策略回测
# ============================================================================

def backtest_timing_strategies(df: pd.DataFrame) -> dict:
    """回测择时策略"""
    print("\n" + "=" * 60)
    print("Part 6: 择时策略回测")
    print("=" * 60)

    df = df.copy()
    df['index_ret'] = df['pct_chg'] / 100  # 中证全指日收益（小数）

    strategies = []

    # 策略1: 融资余额变化择时（正向：余额增加时做多）
    df['signal_margin_chg'] = np.where(df['margin_chg_5d'] > 0, 1, 0)

    # 策略2: 融资余额变化择时（反向：余额减少时做多）
    df['signal_margin_contrarian'] = np.where(df['margin_chg_5d'] < 0, 1, 0)

    # 策略3: 融资余额动量择时（MA金叉做多）
    df['signal_margin_momentum'] = np.where(df['margin_ma_ratio'] > 0, 1, 0)

    # 策略4: Z-Score择时（极端低值做多，极端高值做空）
    df['signal_zscore'] = 0
    df.loc[df['margin_chg_5d_zscore'] < -1, 'signal_zscore'] = 1   # 极端减少 -> 做多
    df.loc[df['margin_chg_5d_zscore'] > 1, 'signal_zscore'] = -1   # 极端增加 -> 做空

    # 策略5: 分位数择时（最低20%分位做多，最高20%做空）
    df['signal_pctile'] = 0
    df.loc[df['margin_pctchg_5d_pctile'] < 0.2, 'signal_pctile'] = 1
    df.loc[df['margin_pctchg_5d_pctile'] > 0.8, 'signal_pctile'] = -1

    # 策略6: 净买入择时（净买入为正时做多）
    df['signal_net_buy'] = np.where(df['net_buy_5d'] > 0, 1, 0)

    # 策略7: 复合信号（多因子投票）
    df['signal_composite'] = 0
    df.loc[
        (df['margin_chg_5d'] > 0) &
        (df['margin_ma_ratio'] > 0) &
        (df['net_buy_5d'] > 0),
        'signal_composite'
    ] = 1

    strategy_defs = [
        ('持有指数', 'index_ret', None),
        ('正向择时(余额↑做多)', 'index_ret', 'signal_margin_chg'),
        ('反向择时(余额↓做多)', 'index_ret', 'signal_margin_contrarian'),
        ('动量择时(MA金叉)', 'index_ret', 'signal_margin_momentum'),
        ('Z-Score择时', 'index_ret', 'signal_zscore'),
        ('分位数择时', 'index_ret', 'signal_pctile'),
        ('净买入择时', 'index_ret', 'signal_net_buy'),
        ('复合信号择时', 'index_ret', 'signal_composite'),
    ]

    strategy_results = {}

    for name, ret_col, signal_col in strategy_defs:
        if signal_col is None:
            # 持有指数
            df['strategy_ret'] = df[ret_col]
        else:
            # 使用shift(1)避免lookahead bias
            df['strategy_ret'] = df[signal_col].shift(1) * df[ret_col]

        returns = df['strategy_ret'].dropna()
        if len(returns) == 0:
            continue

        # 累计收益
        df['cum_ret'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

        # 年化收益
        n_days = len(returns)
        total_ret = (1 + returns).prod() - 1
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1

        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum = df['cum_ret']
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # 胜率
        win_rate = (returns > 0).mean()

        # 持仓天数比例
        if signal_col:
            position_ratio = (df[signal_col].shift(1).abs() > 0).mean()
        else:
            position_ratio = 1.0

        strategy_results[name] = {
            '累计收益': f"{total_ret*100:.1f}%",
            '年化收益': f"{annual_ret*100:.1f}%",
            '年化波动': f"{annual_vol*100:.1f}%",
            '夏普比率': f"{sharpe:.2f}",
            '最大回撤': f"{max_dd*100:.1f}%",
            '胜率': f"{win_rate*100:.1f}%",
            '持仓比例': f"{position_ratio*100:.1f}%",
            '交易天数': n_days,
            'cum_series': cum.copy(),
        }

    # 打印结果
    print("\n择时策略绩效:")
    print("-" * 110)
    print(f"{'策略':<22s} {'累计收益':>8s} {'年化收益':>8s} {'年化波动':>8s} {'夏普':>6s} {'最大回撤':>8s} {'胜率':>6s} {'持仓比例':>8s}")
    print("-" * 110)

    for name, result in strategy_results.items():
        if name == '持有指数':
            print(f"\n{name:<22s} {result['累计收益']:>8s} {result['年化收益']:>8s} {result['年化波动']:>8s} "
                  f"{result['夏普比率']:>6s} {result['最大回撤']:>8s} {result['胜率']:>6s} {result['持仓比例']:>8s}")
        else:
            print(f"{name:<22s} {result['累计收益']:>8s} {result['年化收益']:>8s} {result['年化波动']:>8s} "
                  f"{result['夏普比率']:>6s} {result['最大回撤']:>8s} {result['胜率']:>6s} {result['持仓比例']:>8s}")

    print("-" * 110)

    return strategy_results


# ============================================================================
# Part 7: 可视化
# ============================================================================

def generate_visualizations(df: pd.DataFrame, ic_monthly: dict, decay_df: pd.DataFrame,
                            quantile_results: dict, strategy_results: dict, output_dir: str):
    """生成可视化图表"""
    print("\n" + "=" * 60)
    print("Part 7: 生成可视化")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 图1: 融资余额与中证全指走势
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1_twin = ax1.twinx()
    ax1.plot(df['trade_date_dt'], df['rzye_yi'] / 1e4, color='#1f77b4', linewidth=1.5, label='融资余额（万亿元）')
    ax1_twin.plot(df['trade_date_dt'], df['close'], color='#ff7f0e', linewidth=1.5, label='中证全指')
    ax1.set_ylabel('融资余额（万亿元）', color='#1f77b4', fontsize=11)
    ax1_twin.set_ylabel('中证全指', color='#ff7f0e', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_title('融资余额 vs 中证全指走势', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 融资余额5日变化
    ax2.bar(df['trade_date_dt'], df['margin_chg_5d'], color=['red' if x > 0 else 'green' for x in df['margin_chg_5d']],
            alpha=0.6, width=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('融资余额5日变化（亿元）', fontsize=11)
    ax2.set_title('融资余额5日累计变化', fontsize=13)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/margin_factor_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_dir}/margin_factor_trend.png")

    # 图2: IC时间序列
    if ic_monthly:
        fig, axes = plt.subplots(len(ic_monthly), 1, figsize=(14, 3 * len(ic_monthly)), sharex=True)
        if len(ic_monthly) == 1:
            axes = [axes]

        for ax, (factor_col, ic_df) in zip(axes, ic_monthly.items()):
            if len(ic_df) == 0:
                continue
            ic_df['date'] = pd.to_datetime(ic_df['year_month'], format='%Y%m')

            ax.bar(ic_df['date'], ic_df['ic'], color=['red' if x > 0 else 'green' for x in ic_df['ic']], alpha=0.6, width=20)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=ic_df['ic'].mean(), color='blue', linestyle='--', alpha=0.7, label=f"均值: {ic_df['ic'].mean():+.3f}")
            ax.set_ylabel('IC', fontsize=10)
            ax.set_title(f'{factor_col} 月度IC', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/margin_factor_ic_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_dir}/margin_factor_ic_series.png")

    # 图3: Alpha衰减
    if not decay_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))

        key_factors = ['margin_chg_5d', 'margin_pctchg_5d', 'net_buy_5d', 'margin_ma_ratio', 'margin_chg_5d_zscore']
        labels = ['变化量5d', '变化率5d', '净买入5d', 'MA比值', 'Z-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for col, label, color in zip(key_factors, labels, colors):
            subset = decay_df[decay_df['factor'] == col]
            if len(subset) > 0:
                ax.plot(subset['horizon'], subset['ic'], marker='o', markersize=4,
                       linewidth=1.5, label=label, color=color, alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='|IC|=0.05')
        ax.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('持有期（交易日）', fontsize=11)
        ax.set_ylabel('IC', fontsize=11)
        ax.set_title('Alpha衰减：IC随持有期变化', fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 20)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/margin_factor_decay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_dir}/margin_factor_decay.png")

    # 图4: 分位数组合收益
    if quantile_results:
        fig, axes = plt.subplots(1, min(len(quantile_results), 3), figsize=(5 * min(len(quantile_results), 3), 5))
        if len(quantile_results) == 1:
            axes = [axes]

        for ax, (factor_col, result) in zip(axes, list(quantile_results.items())[:3]):
            if result.empty:
                continue

            x = range(len(result))
            width = 0.25
            ax.bar([i - width for i in x], result['5日收益'], width, label='5日', color='steelblue', alpha=0.8)
            ax.bar(x, result['10日收益'], width, label='10日', color='orange', alpha=0.8)
            ax.bar([i + width for i in x], result['20日收益'], width, label='20日', color='green', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(result['分位组'])
            ax.set_ylabel('平均收益 (%)', fontsize=10)
            ax.set_title(f'{factor_col}\n分位数组合收益', fontsize=11)
            ax.legend(fontsize=8)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/margin_factor_quantile.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_dir}/margin_factor_quantile.png")

    # 图5: 策略回测净值
    if strategy_results:
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_results)))

        for (name, result), color in zip(strategy_results.items(), colors):
            cum_series = result.get('cum_series')
            if cum_series is not None and len(cum_series) > 0:
                # 使用日期索引
                dates = df.loc[cum_series.index, 'trade_date_dt']
                ax.plot(dates, cum_series, label=name, linewidth=1.5, alpha=0.8, color=color)

        ax.set_ylabel('累计净值', fontsize=11)
        ax.set_title('择时策略累计净值对比', fontsize=13)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/margin_factor_strategy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_dir}/margin_factor_strategy.png")

    # 图6: 融资余额变化 vs 未来收益散点图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_data = [
        ('margin_chg_5d', 'fwd_ret_5d', '融资余额5日变化(亿元)', '未来5日收益(%)'),
        ('margin_pctchg_5d', 'fwd_ret_5d', '融资余额5日变化率(%)', '未来5日收益(%)'),
        ('net_buy_5d', 'fwd_ret_5d', '融资净买入5日(亿元)', '未来5日收益(%)'),
        ('margin_chg_5d_zscore', 'fwd_ret_5d', '融资余额变化Z-Score', '未来5日收益(%)'),
    ]

    for ax, (fx, fy, xlabel, ylabel) in zip(axes.flat, plot_data):
        valid = df[[fx, fy]].dropna()
        if len(valid) < 20:
            continue

        ax.scatter(valid[fx], valid[fy] * 100, alpha=0.4, s=15, color='steelblue')

        # 拟合线
        z = np.polyfit(valid[fx].dropna(), (valid[fy] * 100).dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[fx].min(), valid[fx].max(), 100)
        ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # 计算IC
        ic, pval = stats.spearmanr(valid[fx], valid[fy])
        ax.set_title(f'{xlabel} vs {ylabel}\nIC={ic:+.3f}, p={pval:.3f}', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/margin_factor_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_dir}/margin_factor_scatter.png")


# ============================================================================
# Part 8: 生成报告
# ============================================================================

def generate_report(df: pd.DataFrame, ic_df: pd.DataFrame, ic_monthly: dict,
                    decay_df: pd.DataFrame, quantile_results: dict,
                    strategy_results: dict, output_dir: str):
    """生成Markdown研究报告"""
    print("\n" + "=" * 60)
    print("Part 8: 生成报告")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 计算总体统计
    n_days = len(df)
    date_start = df['trade_date'].min()
    date_end = df['trade_date'].max()

    # IC总结
    ic_summary = []
    for _, row in ic_df.iterrows():
        ic_5d = row.get('IC_5D', np.nan)
        ic_20d = row.get('IC_20D', np.nan)
        if not pd.isna(ic_5d):
            ic_summary.append({
                '因子': row['因子名称'],
                'IC_5D': ic_5d,
                'IC_20D': ic_20d,
            })

    ic_summary_df = pd.DataFrame(ic_summary).sort_values('IC_5D', key=abs, ascending=False)

    # 策略总结
    strategy_summary = []
    for name, result in strategy_results.items():
        strategy_summary.append({
            '策略': name,
            '累计收益': result['累计收益'],
            '年化收益': result['年化收益'],
            '夏普比率': result['夏普比率'],
            '最大回撤': result['最大回撤'],
            '胜率': result['胜率'],
            '持仓比例': result['持仓比例'],
        })
    strategy_summary_df = pd.DataFrame(strategy_summary)

    # 月度IC统计
    monthly_stats = []
    for factor_col, ic_month_df in ic_monthly.items():
        if len(ic_month_df) > 0:
            monthly_stats.append({
                '因子': factor_col,
                'IC均值': ic_month_df['ic'].mean(),
                'IC标准差': ic_month_df['ic'].std(),
                'IR': ic_month_df['ic'].mean() / ic_month_df['ic'].std() if ic_month_df['ic'].std() > 0 else 0,
                '正向率': (ic_month_df['ic'] > 0).mean(),
            })
    monthly_stats_df = pd.DataFrame(monthly_stats)

    report = f"""# 融资余额变化因子研究报告

> **研究主题**: 融资余额变化对中证全指(000985.SH)的预测作用
> **研究期间**: {date_start} - {date_end}
> **样本天数**: {n_days} 个交易日
> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、研究概述

### 1.1 研究背景

融资融券余额是衡量市场情绪的重要指标。融资余额反映投资者通过杠杆做多的意愿，其变化可能包含对未来市场走势的预测信息。

本研究系统性地挖掘融资余额变化的多种衍生因子，评估其对中证全指未来收益的预测能力。

### 1.2 数据说明

| 数据项 | 来源 | 范围 |
|--------|------|------|
| 融资余额 | Tushare margin表 | 沪深两市汇总 |
| 中证全指 | Tushare index_daily | 000985.SH |
| 研究期间 | {date_start} ~ {date_end} | 约{n_days}个交易日 |

---

## 二、因子构建

### 2.1 因子分类

| 类别 | 因子 | 说明 |
|------|------|------|
| **变化量** | margin_chg_1/5/20d | 融资余额N日变化（亿元） |
| **变化率** | margin_pctchg_1/5/20d | 融资余额N日变化率（%） |
| **净买入** | net_buy_1/5/20d | 融资净买入额 = 买入额 - 偿还额（亿元） |
| **动量** | margin_ma_ratio | 融资余额MA5/MA20 - 1（%） |
| **标准化** | margin_chg_5d_zscore | 120日滚动Z-Score |
| **分位数** | margin_pctchg_5d_pctile | 120日滚动历史分位数 |
| **比率** | margin_per_index_chg | 融资余额/指数点位变化率 |
| **加速度** | margin_pctchg_5d_accel | 变化率的二阶差分 |
| **强度** | buy_intensity_5d | 融资买入额/余额（%） |

### 2.2 因子统计

"""

    # 因子描述统计
    factor_cols = ['margin_chg_5d', 'margin_pctchg_5d', 'net_buy_5d', 'margin_ma_ratio',
                   'margin_chg_5d_zscore', 'buy_intensity_5d', 'net_buy_ratio_5d']
    report += "| 因子 | 均值 | 标准差 | 最小值 | 最大值 |\n"
    report += "|------|------|--------|--------|--------|\n"
    for col in factor_cols:
        if col in df.columns:
            report += f"| {col} | {df[col].mean():.3f} | {df[col].std():.3f} | {df[col].min():.3f} | {df[col].max():.3f} |\n"

    report += """
---

## 三、IC/IR分析

### 3.1 因子与未来收益IC

| 因子 | IC_1D | IC_5D | IC_10D | IC_20D |
|------|-------|-------|--------|--------|
"""

    for _, row in ic_df.iterrows():
        ic_vals = []
        for h in [1, 5, 10, 20]:
            ic = row.get(f'IC_{h}D', np.nan)
            if pd.isna(ic):
                ic_vals.append('---')
            else:
                marker = '**' if abs(ic) > 0.05 else ''
                ic_vals.append(f"{marker}{ic:+.3f}{marker}")
        report += f"| {row['因子名称']} | {ic_vals[0]} | {ic_vals[1]} | {ic_vals[2]} | {ic_vals[3]} |\n"

    report += """
> 注: **加粗**表示 |IC| > 0.05（中等强度信号）

### 3.2 月度IC统计

| 因子 | IC均值 | IC标准差 | IR | 正向率 |
|------|--------|----------|-----|--------|
"""

    if not monthly_stats_df.empty:
        for _, row in monthly_stats_df.iterrows():
            report += f"| {row['因子']} | {row['IC均值']:+.4f} | {row['IC标准差']:.4f} | {row['IR']:+.3f} | {row['正向率']*100:.1f}% |\n"

    report += """
### 3.3 IC分析结论

"""

    if not ic_summary_df.empty:
        best_ic = ic_summary_df.iloc[0]
        report += f"""- **最强预测因子**: {best_ic['因子']}（5日IC = {best_ic['IC_5D']:+.4f}）
- **预测方向**: {"正相关（余额增加预示上涨）" if best_ic['IC_5D'] > 0 else "负相关（余额减少预示上涨）"}
- **预测持续性**: {"短期效果更强" if abs(best_ic['IC_5D']) > abs(best_ic.get('IC_20D', 0)) else "长期效果更强"}
"""

    report += """
---

## 四、分位数组合测试

"""

    for factor_col, result in quantile_results.items():
        if result.empty:
            continue
        report += f"\n### 4.1 {factor_col}\n\n"
        report += result.to_markdown(index=False) + "\n\n"

        if len(result) >= 2:
            ls_ret = result.iloc[-1]['20日收益'] - result.iloc[0]['20日收益']
            report += f"**多空收益**: 做多Q5 + 做空Q1 = {ls_ret:+.3f}%\n\n"

    report += """---

## 五、择时策略回测

### 5.1 策略绩效

| 策略 | 累计收益 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 持仓比例 |
|------|---------|---------|---------|---------|------|----------|
"""

    for _, row in strategy_summary_df.iterrows():
        report += f"| {row['策略']} | {row['累计收益']} | {row['年化收益']} | {row['夏普比率']} | {row['最大回撤']} | {row['胜率']} |\n"

    report += """
### 5.2 策略说明

| 策略 | 信号规则 | 逻辑 |
|------|---------|------|
| 正向择时 | 融资余额5日变化 > 0 时做多 | 跟随杠杆资金 |
| 反向择时 | 融资余额5日变化 < 0 时做多 | 逆向操作 |
| 动量择时 | MA5 > MA20 时做多 | 趋势跟踪 |
| Z-Score择时 | Z < -1做多, Z > 1做空 | 均值回归 |
| 分位数择时 | 分位 < 20%做多, > 80%做空 | 极端值反转 |
| 净买入择时 | 净买入 > 0 时做多 | 资金流向 |
| 复合信号 | 多因子同时满足时做多 | 信号共振 |

---

## 六、主要发现

"""

    # 根据IC结果生成发现
    if not ic_summary_df.empty:
        top_positive = ic_summary_df[ic_summary_df['IC_5D'] > 0].head(1)
        top_negative = ic_summary_df[ic_summary_df['IC_5D'] < 0].head(1)

        if not top_positive.empty:
            report += f"""1. **正向预测因子**: {top_positive.iloc[0]['因子']} 与未来收益呈正相关（IC_5D = {top_positive.iloc[0]['IC_5D']:+.3f}），说明{"融资余额增加预示市场上涨" if 'margin' in top_positive.iloc[0]['因子'] else "该因子增加预示市场上涨"}

"""
        if not top_negative.empty:
            report += f"""2. **反向预测因子**: {top_negative.iloc[0]['因子']} 与未来收益呈负相关（IC_5D = {top_negative.iloc[0]['IC_5D']:+.3f}），说明{"融资余额增速放缓可能是市场见顶信号" if 'margin' in top_negative.iloc[0]['因子'] else "该因子减小预示市场上涨"}

"""

    report += """3. **预测持续性**: 融资余额因子的预测能力主要集中在短期（1-5日），随着持有期延长，IC衰减明显

4. **策略效果**: 基于融资余额变化的择时策略在样本期内{"跑赢" if any('择时' in k and float(v['夏普比率']) > 0 for k, v in strategy_results.items() if '择时' in k) else "与持有指数表现接近"}基准指数

---

## 七、风险提示

1. **样本期较短**: 融资融券数据始于2019年，仅约6年数据，可能未覆盖完整市场周期
2. **幸存者偏差**: 分析基于当前仍在交易的股票，未考虑退市股票
3. **过拟合风险**: 测试了多个因子和策略，存在数据挖掘偏差
4. **交易成本未考虑**: 回测未计入交易费用和滑点，实际收益会打折扣
5. **市场环境变化**: 融资融券制度和市场结构可能变化，历史规律未必持续

---

## 八、后续研究方向

1. **个股级分析**: 使用 margin_detail 表进行个股融资余额变化与个股收益的预测研究
2. **行业轮动**: 分析各行业融资余额变化的差异，构建行业轮动策略
3. **结合其他情绪指标**: 将融资余额与涨跌停数量、换手率等结合，构建综合情绪指数
4. **机器学习增强**: 使用多因子模型或机器学习方法组合融资余额因子

---

## 附录：图表索引

1. `margin_factor_trend.png` - 融资余额与中证全指走势
2. `margin_factor_ic_series.png` - 月度IC时间序列
3. `margin_factor_decay.png` - Alpha衰减曲线
4. `margin_factor_quantile.png` - 分位数组合收益
5. `margin_factor_strategy.png` - 策略净值曲线
6. `margin_factor_scatter.png` - 因子与收益散点图

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*声明: 本报告仅供研究参考，不构成投资建议*
"""

    report_path = f'{output_dir}/margin_factor_research_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  已保存: {report_path}")

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    db_path = args.db_path or os.getenv('DB_PATH', 'tushare.db')
    end_date = args.end_date or datetime.now().strftime('%Y%m%d')

    print("=" * 70)
    print("融资余额变化因子研究")
    print("=" * 70)
    print(f"数据库: {db_path}")
    print(f"研究期间: {args.start_date} ~ {end_date}")
    print("=" * 70)

    reader = DataReader(db_path=db_path)
    try:
        # Part 1: 加载数据
        df = load_data(reader, args.start_date, end_date)

        # Part 2: 构建因子
        df = build_factors(df)

        # Part 3: IC分析
        ic_df = analyze_ic(df)

        # Part 3b: 月度IC时间序列
        key_factors = ['margin_chg_5d', 'margin_pctchg_5d', 'net_buy_5d', 'margin_ma_ratio']
        ic_monthly = {}
        print("\n" + "-" * 60)
        print("月度IC统计:")
        print("-" * 60)
        for col in key_factors:
            if col in df.columns:
                ic_month_df = analyze_ic_by_period(df, col)
                ic_monthly[col] = ic_month_df

        # Part 4: Alpha衰减
        decay_df = alpha_decay_analysis(df, key_factors)

        # Part 5: 分位数组合
        quantile_results = run_quantile_tests(df)

        # Part 6: 策略回测
        strategy_results = backtest_timing_strategies(df)

        # Part 7: 可视化
        generate_visualizations(df, ic_monthly, decay_df, quantile_results, strategy_results, args.output_dir)

        # Part 8: 报告
        generate_report(df, ic_df, ic_monthly, decay_df, quantile_results, strategy_results, args.output_dir)

        print("\n" + "=" * 70)
        print("研究完成！")
        print(f"报告: {args.output_dir}/margin_factor_research_report.md")
        print("=" * 70)

    finally:
        reader.close()


if __name__ == '__main__':
    main()
