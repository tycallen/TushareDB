#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势强度指标研究分析
研究内容:
1. 趋势指标计算: ADX、趋势线斜率、均线排列、Aroon指标
2. 趋势特征分析: 强趋势股票特征、趋势持续时间、趋势反转信号
3. 策略应用: 趋势跟踪策略、趋势过滤器、与动量结合
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def calculate_aroon(df, period=25):
    """
    计算Aroon指标
    Aroon Up = ((期间数 - 自最高价以来的天数) / 期间数) * 100
    Aroon Down = ((期间数 - 自最低价以来的天数) / 期间数) * 100
    """
    aroon_up = []
    aroon_down = []

    for i in range(len(df)):
        if i < period:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            high_slice = df['high_qfq'].iloc[i-period:i+1].values
            low_slice = df['low_qfq'].iloc[i-period:i+1].values

            days_since_high = period - np.argmax(high_slice)
            days_since_low = period - np.argmin(low_slice)

            aroon_up.append(((period - days_since_high) / period) * 100)
            aroon_down.append(((period - days_since_low) / period) * 100)

    df['aroon_up'] = aroon_up
    df['aroon_down'] = aroon_down
    df['aroon_osc'] = df['aroon_up'] - df['aroon_down']

    return df

def calculate_trend_slope(df, period=20):
    """
    计算趋势线斜率
    使用线性回归计算价格趋势斜率
    """
    slopes = []
    r_squared_list = []

    for i in range(len(df)):
        if i < period:
            slopes.append(np.nan)
            r_squared_list.append(np.nan)
        else:
            y = df['close_qfq'].iloc[i-period+1:i+1].values
            x = np.arange(period)

            # 避免除零
            if np.std(y) == 0:
                slopes.append(0)
                r_squared_list.append(1)
            else:
                # 线性回归
                slope, intercept = np.polyfit(x, y, 1)
                # 标准化斜率（以百分比形式）
                normalized_slope = (slope / y[0]) * 100 if y[0] != 0 else 0
                slopes.append(normalized_slope)

                # 计算R平方
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                r_squared_list.append(r_squared)

    df['trend_slope'] = slopes
    df['trend_r_squared'] = r_squared_list

    return df

def calculate_ma_alignment(df):
    """
    计算均线排列得分
    多头排列: MA5 > MA10 > MA20 > MA60 (正分)
    空头排列: MA5 < MA10 < MA20 < MA60 (负分)
    """
    alignment_scores = []

    for i in range(len(df)):
        row = df.iloc[i]
        ma5 = row.get('ma_qfq_5', np.nan)
        ma10 = row.get('ma_qfq_10', np.nan)
        ma20 = row.get('ma_qfq_20', np.nan)
        ma60 = row.get('ma_qfq_60', np.nan)

        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma60):
            alignment_scores.append(np.nan)
            continue

        score = 0
        # 多头排列检查
        if ma5 > ma10:
            score += 1
        elif ma5 < ma10:
            score -= 1

        if ma10 > ma20:
            score += 1
        elif ma10 < ma20:
            score -= 1

        if ma20 > ma60:
            score += 1
        elif ma20 < ma60:
            score -= 1

        alignment_scores.append(score)

    df['ma_alignment'] = alignment_scores

    return df

def calculate_composite_trend_score(df):
    """
    计算综合趋势得分
    结合ADX、Aroon、均线排列、趋势斜率
    """
    scores = []

    for i in range(len(df)):
        row = df.iloc[i]
        score = 0

        # ADX贡献 (0-100 -> -2到+2)
        adx = row.get('dmi_adx_qfq', np.nan)
        pdi = row.get('dmi_pdi_qfq', np.nan)
        mdi = row.get('dmi_mdi_qfq', np.nan)

        if not pd.isna(adx) and not pd.isna(pdi) and not pd.isna(mdi):
            # ADX强度
            if adx > 40:
                trend_strength = 2
            elif adx > 25:
                trend_strength = 1
            else:
                trend_strength = 0

            # 方向
            if pdi > mdi:
                score += trend_strength
            else:
                score -= trend_strength

        # Aroon贡献
        aroon_osc = row.get('aroon_osc', np.nan)
        if not pd.isna(aroon_osc):
            if aroon_osc > 50:
                score += 2
            elif aroon_osc > 0:
                score += 1
            elif aroon_osc < -50:
                score -= 2
            elif aroon_osc < 0:
                score -= 1

        # 均线排列贡献
        ma_align = row.get('ma_alignment', np.nan)
        if not pd.isna(ma_align):
            score += ma_align

        # 趋势斜率贡献
        slope = row.get('trend_slope', np.nan)
        r_sq = row.get('trend_r_squared', np.nan)
        if not pd.isna(slope) and not pd.isna(r_sq):
            if r_sq > 0.8:  # 趋势明确
                if slope > 2:
                    score += 2
                elif slope > 0.5:
                    score += 1
                elif slope < -2:
                    score -= 2
                elif slope < -0.5:
                    score -= 1

        scores.append(score)

    df['composite_trend_score'] = scores

    return df

def analyze_trend_indicators():
    """分析趋势指标"""
    print("=" * 60)
    print("1. 趋势指标分析")
    print("=" * 60)

    conn = get_connection()

    # 获取最近一年的数据用于分析
    end_date = '20260130'
    start_date = '20250101'

    # 获取活跃股票的数据
    query = f"""
    SELECT
        ts_code,
        trade_date,
        close_qfq,
        high_qfq,
        low_qfq,
        open_qfq,
        pct_chg,
        vol,
        dmi_adx_qfq,
        dmi_adxr_qfq,
        dmi_pdi_qfq,
        dmi_mdi_qfq,
        ma_qfq_5,
        ma_qfq_10,
        ma_qfq_20,
        ma_qfq_60,
        ma_qfq_90,
        ma_qfq_250,
        rsi_qfq_6,
        rsi_qfq_12,
        macd_qfq,
        macd_dif_qfq,
        macd_dea_qfq
    FROM stk_factor_pro
    WHERE trade_date >= '{start_date}'
    AND trade_date <= '{end_date}'
    AND close_qfq IS NOT NULL
    AND close_qfq > 0
    AND vol > 0
    ORDER BY ts_code, trade_date
    """

    print(f"正在加载数据（{start_date} - {end_date}）...")
    df = conn.execute(query).fetchdf()
    print(f"数据加载完成，共 {len(df):,} 条记录，{df['ts_code'].nunique()} 只股票")

    # 按股票分组计算额外指标
    print("\n正在计算趋势指标...")
    results = []
    stock_groups = df.groupby('ts_code')

    for ts_code, group in stock_groups:
        group = group.sort_values('trade_date').reset_index(drop=True)
        if len(group) < 60:  # 数据不足
            continue

        # 计算Aroon
        group = calculate_aroon(group)
        # 计算趋势斜率
        group = calculate_trend_slope(group)
        # 计算均线排列
        group = calculate_ma_alignment(group)
        # 计算综合得分
        group = calculate_composite_trend_score(group)

        results.append(group)

    df_all = pd.concat(results, ignore_index=True)
    print(f"指标计算完成，共 {len(df_all):,} 条记录")

    # ADX分布分析
    print("\n--- ADX分布统计 ---")
    latest_data = df_all[df_all['trade_date'] == df_all['trade_date'].max()].copy()

    adx_bins = [0, 20, 25, 40, 60, 100]
    adx_labels = ['<20(无趋势)', '20-25(弱趋势)', '25-40(强趋势)', '40-60(极强趋势)', '>60(超强趋势)']
    latest_data['adx_category'] = pd.cut(latest_data['dmi_adx_qfq'], bins=adx_bins, labels=adx_labels)

    adx_dist = latest_data['adx_category'].value_counts().sort_index()
    print("\nADX分布:")
    for cat, count in adx_dist.items():
        pct = count / len(latest_data) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")

    # Aroon分析
    print("\n--- Aroon指标分析 ---")
    aroon_summary = latest_data[['aroon_up', 'aroon_down', 'aroon_osc']].describe()
    print(aroon_summary)

    # 均线排列统计
    print("\n--- 均线排列统计 ---")
    ma_dist = latest_data['ma_alignment'].value_counts().sort_index()
    print("均线排列得分分布 (-3=空头, +3=多头):")
    for score, count in ma_dist.items():
        pct = count / len(latest_data) * 100
        status = "多头" if score > 0 else ("空头" if score < 0 else "中性")
        print(f"  {int(score):+d} ({status}): {count:,} ({pct:.1f}%)")

    # 趋势斜率分析
    print("\n--- 趋势斜率分析 ---")
    slope_summary = latest_data[['trend_slope', 'trend_r_squared']].describe()
    print(slope_summary)

    # 综合趋势得分分布
    print("\n--- 综合趋势得分分布 ---")
    score_dist = latest_data['composite_trend_score'].value_counts().sort_index()
    print("综合趋势得分 (负=空头, 正=多头):")
    for score, count in score_dist.items():
        pct = count / len(latest_data) * 100
        print(f"  {int(score):+d}: {count:,} ({pct:.1f}%)")

    conn.close()

    return df_all, latest_data

def analyze_strong_trend_characteristics(df_all, latest_data):
    """分析强趋势股票特征"""
    print("\n" + "=" * 60)
    print("2. 强趋势股票特征分析")
    print("=" * 60)

    # 定义强上升趋势
    strong_uptrend = latest_data[
        (latest_data['dmi_adx_qfq'] > 25) &
        (latest_data['dmi_pdi_qfq'] > latest_data['dmi_mdi_qfq']) &
        (latest_data['ma_alignment'] >= 2)
    ].copy()

    # 定义强下降趋势
    strong_downtrend = latest_data[
        (latest_data['dmi_adx_qfq'] > 25) &
        (latest_data['dmi_mdi_qfq'] > latest_data['dmi_pdi_qfq']) &
        (latest_data['ma_alignment'] <= -2)
    ].copy()

    print(f"\n强上升趋势股票: {len(strong_uptrend)} 只 ({len(strong_uptrend)/len(latest_data)*100:.1f}%)")
    print(f"强下降趋势股票: {len(strong_downtrend)} 只 ({len(strong_downtrend)/len(latest_data)*100:.1f}%)")

    # 强趋势股票特征
    print("\n--- 强上升趋势股票特征 ---")
    if len(strong_uptrend) > 0:
        print(f"平均ADX: {strong_uptrend['dmi_adx_qfq'].mean():.2f}")
        print(f"平均趋势斜率: {strong_uptrend['trend_slope'].mean():.2f}%")
        print(f"平均R平方: {strong_uptrend['trend_r_squared'].mean():.3f}")
        print(f"平均Aroon振荡器: {strong_uptrend['aroon_osc'].mean():.2f}")
        print(f"平均RSI(12): {strong_uptrend['rsi_qfq_12'].mean():.2f}")

    print("\n--- 强下降趋势股票特征 ---")
    if len(strong_downtrend) > 0:
        print(f"平均ADX: {strong_downtrend['dmi_adx_qfq'].mean():.2f}")
        print(f"平均趋势斜率: {strong_downtrend['trend_slope'].mean():.2f}%")
        print(f"平均R平方: {strong_downtrend['trend_r_squared'].mean():.3f}")
        print(f"平均Aroon振荡器: {strong_downtrend['aroon_osc'].mean():.2f}")
        print(f"平均RSI(12): {strong_downtrend['rsi_qfq_12'].mean():.2f}")

    return strong_uptrend, strong_downtrend

def analyze_trend_duration(df_all):
    """分析趋势持续时间"""
    print("\n" + "=" * 60)
    print("3. 趋势持续时间分析")
    print("=" * 60)

    # 选择部分股票进行详细分析
    sample_stocks = df_all['ts_code'].unique()[:500]  # 取500只股票样本

    trend_durations = []

    for ts_code in sample_stocks:
        stock_data = df_all[df_all['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)

        if len(stock_data) < 30:
            continue

        # 识别趋势期间
        in_uptrend = False
        in_downtrend = False
        trend_start = None

        for i in range(len(stock_data)):
            row = stock_data.iloc[i]
            adx = row['dmi_adx_qfq']
            pdi = row['dmi_pdi_qfq']
            mdi = row['dmi_mdi_qfq']

            if pd.isna(adx):
                continue

            is_trending = adx > 25
            is_uptrend = pdi > mdi

            if is_trending and is_uptrend and not in_uptrend:
                # 开始上升趋势
                in_uptrend = True
                in_downtrend = False
                trend_start = i
            elif is_trending and not is_uptrend and not in_downtrend:
                # 开始下降趋势
                in_downtrend = True
                if in_uptrend and trend_start is not None:
                    duration = i - trend_start
                    if duration >= 5:  # 至少5天
                        trend_durations.append({
                            'ts_code': ts_code,
                            'type': 'up',
                            'duration': duration
                        })
                in_uptrend = False
                trend_start = i
            elif not is_trending:
                # 趋势结束
                if in_uptrend and trend_start is not None:
                    duration = i - trend_start
                    if duration >= 5:
                        trend_durations.append({
                            'ts_code': ts_code,
                            'type': 'up',
                            'duration': duration
                        })
                elif in_downtrend and trend_start is not None:
                    duration = i - trend_start
                    if duration >= 5:
                        trend_durations.append({
                            'ts_code': ts_code,
                            'type': 'down',
                            'duration': duration
                        })
                in_uptrend = False
                in_downtrend = False
                trend_start = None

    if len(trend_durations) > 0:
        df_durations = pd.DataFrame(trend_durations)

        print(f"\n识别到 {len(df_durations)} 个趋势期间")

        up_durations = df_durations[df_durations['type'] == 'up']['duration']
        down_durations = df_durations[df_durations['type'] == 'down']['duration']

        print("\n上升趋势持续时间统计（交易日）:")
        if len(up_durations) > 0:
            print(f"  平均: {up_durations.mean():.1f} 天")
            print(f"  中位数: {up_durations.median():.1f} 天")
            print(f"  最长: {up_durations.max():.0f} 天")
            print(f"  25分位: {up_durations.quantile(0.25):.1f} 天")
            print(f"  75分位: {up_durations.quantile(0.75):.1f} 天")

        print("\n下降趋势持续时间统计（交易日）:")
        if len(down_durations) > 0:
            print(f"  平均: {down_durations.mean():.1f} 天")
            print(f"  中位数: {down_durations.median():.1f} 天")
            print(f"  最长: {down_durations.max():.0f} 天")
            print(f"  25分位: {down_durations.quantile(0.25):.1f} 天")
            print(f"  75分位: {down_durations.quantile(0.75):.1f} 天")

        return df_durations

    return None

def analyze_trend_reversal_signals(df_all):
    """分析趋势反转信号"""
    print("\n" + "=" * 60)
    print("4. 趋势反转信号分析")
    print("=" * 60)

    # 选择样本股票
    sample_stocks = df_all['ts_code'].unique()[:300]

    reversal_signals = []

    for ts_code in sample_stocks:
        stock_data = df_all[df_all['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)

        if len(stock_data) < 30:
            continue

        for i in range(5, len(stock_data) - 10):
            row = stock_data.iloc[i]
            prev_row = stock_data.iloc[i-1]

            adx = row['dmi_adx_qfq']
            pdi = row['dmi_pdi_qfq']
            mdi = row['dmi_mdi_qfq']
            prev_pdi = prev_row['dmi_pdi_qfq']
            prev_mdi = prev_row['dmi_mdi_qfq']

            if pd.isna(adx) or pd.isna(prev_pdi):
                continue

            # 检测PDI/MDI交叉
            bullish_cross = (pdi > mdi) and (prev_pdi <= prev_mdi)
            bearish_cross = (pdi < mdi) and (prev_pdi >= prev_mdi)

            # Aroon信号
            aroon_up = row.get('aroon_up', np.nan)
            aroon_down = row.get('aroon_down', np.nan)
            prev_aroon_up = prev_row.get('aroon_up', np.nan)
            prev_aroon_down = prev_row.get('aroon_down', np.nan)

            aroon_bullish = False
            aroon_bearish = False
            if not pd.isna(aroon_up) and not pd.isna(prev_aroon_up):
                aroon_bullish = (aroon_up > aroon_down) and (prev_aroon_up <= prev_aroon_down)
                aroon_bearish = (aroon_up < aroon_down) and (prev_aroon_up >= prev_aroon_down)

            # 计算反转后10天的收益
            if i + 10 < len(stock_data):
                future_return = (stock_data.iloc[i+10]['close_qfq'] - row['close_qfq']) / row['close_qfq'] * 100
            else:
                future_return = np.nan

            if bullish_cross or aroon_bullish:
                reversal_signals.append({
                    'ts_code': ts_code,
                    'date': row['trade_date'],
                    'type': 'bullish',
                    'signal_source': 'DMI' if bullish_cross else 'Aroon',
                    'adx': adx,
                    'future_return_10d': future_return
                })

            if bearish_cross or aroon_bearish:
                reversal_signals.append({
                    'ts_code': ts_code,
                    'date': row['trade_date'],
                    'type': 'bearish',
                    'signal_source': 'DMI' if bearish_cross else 'Aroon',
                    'adx': adx,
                    'future_return_10d': future_return
                })

    if len(reversal_signals) > 0:
        df_signals = pd.DataFrame(reversal_signals)

        print(f"\n识别到 {len(df_signals)} 个反转信号")

        # 按类型统计
        bullish_signals = df_signals[df_signals['type'] == 'bullish']
        bearish_signals = df_signals[df_signals['type'] == 'bearish']

        print(f"\n看涨反转信号: {len(bullish_signals)}")
        if len(bullish_signals) > 0:
            valid_returns = bullish_signals['future_return_10d'].dropna()
            if len(valid_returns) > 0:
                print(f"  10天后平均收益: {valid_returns.mean():.2f}%")
                print(f"  胜率(正收益): {(valid_returns > 0).mean() * 100:.1f}%")

                # 按ADX强度分组
                high_adx_signals = bullish_signals[bullish_signals['adx'] > 25]['future_return_10d'].dropna()
                low_adx_signals = bullish_signals[bullish_signals['adx'] <= 25]['future_return_10d'].dropna()

                if len(high_adx_signals) > 0:
                    print(f"  高ADX(>25)信号平均收益: {high_adx_signals.mean():.2f}%")
                if len(low_adx_signals) > 0:
                    print(f"  低ADX(<=25)信号平均收益: {low_adx_signals.mean():.2f}%")

        print(f"\n看跌反转信号: {len(bearish_signals)}")
        if len(bearish_signals) > 0:
            valid_returns = bearish_signals['future_return_10d'].dropna()
            if len(valid_returns) > 0:
                print(f"  10天后平均收益: {valid_returns.mean():.2f}%")
                print(f"  胜率(负收益): {(valid_returns < 0).mean() * 100:.1f}%")

        return df_signals

    return None

def backtest_trend_following_strategy(df_all):
    """回测趋势跟踪策略"""
    print("\n" + "=" * 60)
    print("5. 趋势跟踪策略回测")
    print("=" * 60)

    # 选择有足够数据的股票
    sample_stocks = df_all.groupby('ts_code').size()
    valid_stocks = sample_stocks[sample_stocks >= 200].index[:200].tolist()

    strategy_results = []

    for ts_code in valid_stocks:
        stock_data = df_all[df_all['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)

        position = 0  # 0=空仓, 1=持仓
        entry_price = 0
        trades = []

        for i in range(1, len(stock_data)):
            row = stock_data.iloc[i]
            prev_row = stock_data.iloc[i-1]

            adx = row['dmi_adx_qfq']
            pdi = row['dmi_pdi_qfq']
            mdi = row['dmi_mdi_qfq']
            close = row['close_qfq']
            ma_align = row.get('ma_alignment', 0)

            if pd.isna(adx) or pd.isna(close):
                continue

            # 入场条件: ADX>25, PDI>MDI, 均线多头排列(>=2)
            if position == 0:
                if adx > 25 and pdi > mdi and ma_align >= 2:
                    position = 1
                    entry_price = close
                    entry_date = row['trade_date']

            # 出场条件: PDI<MDI 或 ADX下降且<20
            elif position == 1:
                exit_signal = False

                if pdi < mdi:
                    exit_signal = True
                elif adx < 20:
                    exit_signal = True

                if exit_signal:
                    position = 0
                    exit_price = close
                    pct_return = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'ts_code': ts_code,
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_pct': pct_return
                    })

        strategy_results.extend(trades)

    if len(strategy_results) > 0:
        df_trades = pd.DataFrame(strategy_results)

        print(f"\n共完成 {len(df_trades)} 笔交易")
        print(f"涉及 {df_trades['ts_code'].nunique()} 只股票")

        returns = df_trades['return_pct']
        print(f"\n收益统计:")
        print(f"  平均收益: {returns.mean():.2f}%")
        print(f"  收益中位数: {returns.median():.2f}%")
        print(f"  收益标准差: {returns.std():.2f}%")
        print(f"  最大盈利: {returns.max():.2f}%")
        print(f"  最大亏损: {returns.min():.2f}%")

        win_rate = (returns > 0).mean() * 100
        print(f"\n胜率: {win_rate:.1f}%")

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
        print(f"平均盈利: {avg_win:.2f}%")
        print(f"平均亏损: {abs(avg_loss):.2f}%")
        print(f"盈亏比: {profit_factor:.2f}")

        # 期望收益
        expected_return = win_rate / 100 * avg_win - (1 - win_rate / 100) * avg_loss
        print(f"期望收益: {expected_return:.2f}%")

        return df_trades

    return None

def backtest_trend_filter_strategy(df_all):
    """回测趋势过滤器策略"""
    print("\n" + "=" * 60)
    print("6. 趋势过滤器策略回测")
    print("=" * 60)

    print("\n策略说明:")
    print("- 只在ADX>25（有明确趋势）时进行交易")
    print("- 在上升趋势中买入（PDI>MDI）")
    print("- 综合趋势得分>=3时入场")
    print("- 综合趋势得分<=0时出场")

    # 选择样本
    sample_stocks = df_all.groupby('ts_code').size()
    valid_stocks = sample_stocks[sample_stocks >= 200].index[:200].tolist()

    strategy_results = []

    for ts_code in valid_stocks:
        stock_data = df_all[df_all['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)

        position = 0
        entry_price = 0
        trades = []

        for i in range(1, len(stock_data)):
            row = stock_data.iloc[i]

            adx = row['dmi_adx_qfq']
            pdi = row['dmi_pdi_qfq']
            mdi = row['dmi_mdi_qfq']
            close = row['close_qfq']
            trend_score = row.get('composite_trend_score', 0)

            if pd.isna(adx) or pd.isna(close) or pd.isna(trend_score):
                continue

            # 趋势过滤器: 只在有趋势的市场交易
            has_trend = adx > 25

            if position == 0 and has_trend:
                # 入场: 综合趋势得分>=3
                if trend_score >= 3 and pdi > mdi:
                    position = 1
                    entry_price = close
                    entry_date = row['trade_date']

            elif position == 1:
                # 出场: 趋势得分<=0 或趋势消失
                if trend_score <= 0 or not has_trend:
                    position = 0
                    exit_price = close
                    pct_return = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'ts_code': ts_code,
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'return_pct': pct_return
                    })

        strategy_results.extend(trades)

    if len(strategy_results) > 0:
        df_trades = pd.DataFrame(strategy_results)

        print(f"\n共完成 {len(df_trades)} 笔交易")

        returns = df_trades['return_pct']
        print(f"\n收益统计:")
        print(f"  平均收益: {returns.mean():.2f}%")
        print(f"  收益中位数: {returns.median():.2f}%")
        print(f"  最大盈利: {returns.max():.2f}%")
        print(f"  最大亏损: {returns.min():.2f}%")

        win_rate = (returns > 0).mean() * 100
        print(f"\n胜率: {win_rate:.1f}%")

        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        print(f"盈亏比: {avg_win/avg_loss:.2f}" if avg_loss != 0 else "盈亏比: N/A")

        return df_trades

    return None

def backtest_trend_momentum_strategy(df_all):
    """回测趋势+动量结合策略"""
    print("\n" + "=" * 60)
    print("7. 趋势+动量结合策略回测")
    print("=" * 60)

    print("\n策略说明:")
    print("- 趋势确认: ADX>25, 均线多头排列")
    print("- 动量确认: RSI在40-70之间（避免超买超卖）")
    print("- MACD确认: MACD>0或金叉")
    print("- 出场: 趋势消失或RSI超过80")

    sample_stocks = df_all.groupby('ts_code').size()
    valid_stocks = sample_stocks[sample_stocks >= 200].index[:200].tolist()

    strategy_results = []

    for ts_code in valid_stocks:
        stock_data = df_all[df_all['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)

        position = 0
        entry_price = 0
        trades = []

        for i in range(1, len(stock_data)):
            row = stock_data.iloc[i]
            prev_row = stock_data.iloc[i-1]

            adx = row['dmi_adx_qfq']
            pdi = row['dmi_pdi_qfq']
            mdi = row['dmi_mdi_qfq']
            close = row['close_qfq']
            ma_align = row.get('ma_alignment', 0)
            rsi = row['rsi_qfq_12']
            macd = row['macd_qfq']
            prev_macd = prev_row['macd_qfq']

            if pd.isna(adx) or pd.isna(close) or pd.isna(rsi):
                continue

            # 趋势条件
            trend_ok = adx > 25 and pdi > mdi and ma_align >= 2

            # 动量条件
            momentum_ok = 40 < rsi < 70

            # MACD条件
            macd_ok = (not pd.isna(macd)) and (macd > 0 or (not pd.isna(prev_macd) and macd > prev_macd))

            if position == 0:
                if trend_ok and momentum_ok and macd_ok:
                    position = 1
                    entry_price = close
                    entry_date = row['trade_date']

            elif position == 1:
                # 出场条件
                exit_signal = False

                if pdi < mdi:
                    exit_signal = True
                elif adx < 20:
                    exit_signal = True
                elif rsi > 80:  # RSI超买
                    exit_signal = True

                if exit_signal:
                    position = 0
                    exit_price = close
                    pct_return = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'ts_code': ts_code,
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'return_pct': pct_return
                    })

        strategy_results.extend(trades)

    if len(strategy_results) > 0:
        df_trades = pd.DataFrame(strategy_results)

        print(f"\n共完成 {len(df_trades)} 笔交易")

        returns = df_trades['return_pct']
        print(f"\n收益统计:")
        print(f"  平均收益: {returns.mean():.2f}%")
        print(f"  收益中位数: {returns.median():.2f}%")
        print(f"  最大盈利: {returns.max():.2f}%")
        print(f"  最大亏损: {returns.min():.2f}%")

        win_rate = (returns > 0).mean() * 100
        print(f"\n胜率: {win_rate:.1f}%")

        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
        print(f"盈亏比: {profit_factor:.2f}")

        expected_return = win_rate / 100 * avg_win - (1 - win_rate / 100) * avg_loss
        print(f"期望收益: {expected_return:.2f}%")

        return df_trades

    return None

def create_visualizations(df_all, latest_data):
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("8. 生成可视化图表")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. ADX分布直方图
    ax1 = axes[0, 0]
    adx_data = latest_data['dmi_adx_qfq'].dropna()
    ax1.hist(adx_data, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(x=25, color='red', linestyle='--', label='ADX=25 (趋势阈值)')
    ax1.axvline(x=40, color='orange', linestyle='--', label='ADX=40 (强趋势)')
    ax1.set_xlabel('ADX值')
    ax1.set_ylabel('股票数量')
    ax1.set_title('ADX分布 - 趋势强度')
    ax1.legend()

    # 2. Aroon振荡器分布
    ax2 = axes[0, 1]
    aroon_data = latest_data['aroon_osc'].dropna()
    ax2.hist(aroon_data, bins=50, color='forestgreen', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='中性线')
    ax2.axvline(x=50, color='orange', linestyle='--', label='强多头')
    ax2.axvline(x=-50, color='purple', linestyle='--', label='强空头')
    ax2.set_xlabel('Aroon振荡器')
    ax2.set_ylabel('股票数量')
    ax2.set_title('Aroon振荡器分布')
    ax2.legend()

    # 3. 均线排列得分分布
    ax3 = axes[0, 2]
    ma_data = latest_data['ma_alignment'].dropna()
    ma_counts = ma_data.value_counts().sort_index()
    colors = ['darkred', 'red', 'salmon', 'gray', 'lightgreen', 'green', 'darkgreen']
    ax3.bar(ma_counts.index, ma_counts.values, color=colors[:len(ma_counts)], edgecolor='white')
    ax3.set_xlabel('均线排列得分')
    ax3.set_ylabel('股票数量')
    ax3.set_title('均线排列得分分布 (-3=空头, +3=多头)')
    ax3.set_xticks(range(-3, 4))

    # 4. 综合趋势得分分布
    ax4 = axes[1, 0]
    score_data = latest_data['composite_trend_score'].dropna()
    score_counts = score_data.value_counts().sort_index()
    n_colors = len(score_counts)
    cmap = plt.cm.RdYlGn
    colors = [cmap((i + abs(score_counts.index.min())) / (abs(score_counts.index.min()) + score_counts.index.max()))
              for i in score_counts.index]
    ax4.bar(score_counts.index, score_counts.values, color=colors, edgecolor='white')
    ax4.set_xlabel('综合趋势得分')
    ax4.set_ylabel('股票数量')
    ax4.set_title('综合趋势得分分布')

    # 5. 趋势斜率 vs R平方散点图
    ax5 = axes[1, 1]
    slope_data = latest_data[['trend_slope', 'trend_r_squared']].dropna()
    # 限制斜率范围避免极端值
    slope_data = slope_data[(slope_data['trend_slope'] > -10) & (slope_data['trend_slope'] < 10)]
    scatter = ax5.scatter(slope_data['trend_slope'], slope_data['trend_r_squared'],
                          c=slope_data['trend_slope'], cmap='RdYlGn', alpha=0.5, s=10)
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='R²=0.8')
    ax5.set_xlabel('趋势斜率 (%)')
    ax5.set_ylabel('R² (趋势确定性)')
    ax5.set_title('趋势斜率 vs 趋势确定性')
    ax5.legend()
    plt.colorbar(scatter, ax=ax5, label='斜率')

    # 6. ADX与趋势方向
    ax6 = axes[1, 2]
    direction_data = latest_data[['dmi_adx_qfq', 'dmi_pdi_qfq', 'dmi_mdi_qfq']].dropna()
    uptrend = direction_data[direction_data['dmi_pdi_qfq'] > direction_data['dmi_mdi_qfq']]['dmi_adx_qfq']
    downtrend = direction_data[direction_data['dmi_pdi_qfq'] <= direction_data['dmi_mdi_qfq']]['dmi_adx_qfq']

    ax6.hist([uptrend, downtrend], bins=30, label=['上升趋势', '下降趋势'],
             color=['green', 'red'], alpha=0.7, edgecolor='white')
    ax6.axvline(x=25, color='black', linestyle='--', label='ADX=25')
    ax6.set_xlabel('ADX值')
    ax6.set_ylabel('股票数量')
    ax6.set_title('不同趋势方向的ADX分布')
    ax6.legend()

    plt.tight_layout()

    # 保存图表
    chart_path = os.path.join(OUTPUT_DIR, 'trend_strength_analysis.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存到: {chart_path}")

    return chart_path

def generate_report(df_all, latest_data, df_durations, df_signals,
                   df_trend_trades, df_filter_trades, df_momentum_trades):
    """生成完整的研究报告"""
    print("\n" + "=" * 60)
    print("9. 生成研究报告")
    print("=" * 60)

    report = []
    report.append("# 趋势强度指标研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n数据范围: 2025-01-01 至 2026-01-30")
    report.append(f"\n分析股票数: {df_all['ts_code'].nunique()}")

    # 一、趋势指标概述
    report.append("\n\n## 一、趋势指标概述")

    report.append("\n### 1.1 ADX（平均趋向指标）")
    report.append("""
ADX是衡量趋势强度的核心指标，不区分趋势方向：
- ADX < 20: 无明显趋势，市场处于盘整
- ADX 20-25: 趋势开始形成
- ADX 25-40: 明确的趋势
- ADX 40-60: 强劲趋势
- ADX > 60: 极强趋势（较罕见）

配合PDI/MDI判断方向：
- PDI > MDI: 上升趋势
- PDI < MDI: 下降趋势
""")

    report.append("\n### 1.2 Aroon指标")
    report.append("""
Aroon指标用于识别趋势的开始和强度：
- Aroon Up: 衡量自最高价以来的时间
- Aroon Down: 衡量自最低价以来的时间
- Aroon振荡器 = Aroon Up - Aroon Down

解读：
- 振荡器 > 50: 强上升趋势
- 振荡器 < -50: 强下降趋势
- 振荡器接近0: 无明显趋势
""")

    report.append("\n### 1.3 均线排列")
    report.append("""
均线排列反映市场多空力量对比：
- 完美多头排列: MA5 > MA10 > MA20 > MA60 (得分+3)
- 完美空头排列: MA5 < MA10 < MA20 < MA60 (得分-3)
- 均线排列得分越高，多头趋势越强
""")

    report.append("\n### 1.4 趋势线斜率")
    report.append("""
使用线性回归计算价格趋势斜率：
- 斜率为正: 上升趋势
- 斜率为负: 下降趋势
- R²值: 趋势的确定性（越接近1，趋势越明确）
""")

    # 二、当前市场趋势状态
    report.append("\n\n## 二、当前市场趋势状态")

    # ADX分布
    adx_bins = [0, 20, 25, 40, 60, 100]
    adx_labels = ['无趋势(<20)', '弱趋势(20-25)', '强趋势(25-40)', '极强趋势(40-60)', '超强趋势(>60)']
    latest_data_copy = latest_data.copy()
    latest_data_copy['adx_cat'] = pd.cut(latest_data_copy['dmi_adx_qfq'], bins=adx_bins, labels=adx_labels)
    adx_dist = latest_data_copy['adx_cat'].value_counts()

    report.append("\n### 2.1 ADX分布")
    report.append("\n| 类别 | 数量 | 占比 |")
    report.append("|------|------|------|")
    for cat in adx_labels:
        count = adx_dist.get(cat, 0)
        pct = count / len(latest_data) * 100
        report.append(f"| {cat} | {count:,} | {pct:.1f}% |")

    # 均线排列
    ma_dist = latest_data['ma_alignment'].value_counts().sort_index()
    report.append("\n### 2.2 均线排列分布")
    report.append("\n| 得分 | 含义 | 数量 | 占比 |")
    report.append("|------|------|------|------|")
    for score in range(-3, 4):
        count = ma_dist.get(score, 0)
        pct = count / len(latest_data) * 100
        meaning = "多头" if score > 0 else ("空头" if score < 0 else "中性")
        report.append(f"| {score:+d} | {meaning} | {count:,} | {pct:.1f}% |")

    # 强趋势股票统计
    strong_up = latest_data[
        (latest_data['dmi_adx_qfq'] > 25) &
        (latest_data['dmi_pdi_qfq'] > latest_data['dmi_mdi_qfq'])
    ]
    strong_down = latest_data[
        (latest_data['dmi_adx_qfq'] > 25) &
        (latest_data['dmi_mdi_qfq'] > latest_data['dmi_pdi_qfq'])
    ]

    report.append("\n### 2.3 趋势方向统计")
    report.append(f"\n- 强上升趋势股票: {len(strong_up)} ({len(strong_up)/len(latest_data)*100:.1f}%)")
    report.append(f"- 强下降趋势股票: {len(strong_down)} ({len(strong_down)/len(latest_data)*100:.1f}%)")
    report.append(f"- 无明显趋势股票: {len(latest_data)-len(strong_up)-len(strong_down)} ({(len(latest_data)-len(strong_up)-len(strong_down))/len(latest_data)*100:.1f}%)")

    # 三、趋势特征分析
    report.append("\n\n## 三、趋势特征分析")

    report.append("\n### 3.1 强趋势股票特征")

    if len(strong_up) > 0:
        report.append("\n**强上升趋势股票特征:**")
        report.append(f"- 平均ADX: {strong_up['dmi_adx_qfq'].mean():.2f}")
        report.append(f"- 平均趋势斜率: {strong_up['trend_slope'].mean():.2f}%")
        report.append(f"- 平均RSI(12): {strong_up['rsi_qfq_12'].mean():.2f}")
        report.append(f"- 平均Aroon振荡器: {strong_up['aroon_osc'].mean():.2f}")

    if len(strong_down) > 0:
        report.append("\n**强下降趋势股票特征:**")
        report.append(f"- 平均ADX: {strong_down['dmi_adx_qfq'].mean():.2f}")
        report.append(f"- 平均趋势斜率: {strong_down['trend_slope'].mean():.2f}%")
        report.append(f"- 平均RSI(12): {strong_down['rsi_qfq_12'].mean():.2f}")
        report.append(f"- 平均Aroon振荡器: {strong_down['aroon_osc'].mean():.2f}")

    # 趋势持续时间
    report.append("\n### 3.2 趋势持续时间")
    if df_durations is not None and len(df_durations) > 0:
        up_dur = df_durations[df_durations['type'] == 'up']['duration']
        down_dur = df_durations[df_durations['type'] == 'down']['duration']

        report.append("\n**上升趋势持续时间（交易日）:**")
        if len(up_dur) > 0:
            report.append(f"- 平均: {up_dur.mean():.1f}天")
            report.append(f"- 中位数: {up_dur.median():.1f}天")
            report.append(f"- 25%-75%分位: {up_dur.quantile(0.25):.0f}-{up_dur.quantile(0.75):.0f}天")

        report.append("\n**下降趋势持续时间（交易日）:**")
        if len(down_dur) > 0:
            report.append(f"- 平均: {down_dur.mean():.1f}天")
            report.append(f"- 中位数: {down_dur.median():.1f}天")
            report.append(f"- 25%-75%分位: {down_dur.quantile(0.25):.0f}-{down_dur.quantile(0.75):.0f}天")

    # 趋势反转信号
    report.append("\n### 3.3 趋势反转信号效果")
    if df_signals is not None and len(df_signals) > 0:
        bullish = df_signals[df_signals['type'] == 'bullish']
        bearish = df_signals[df_signals['type'] == 'bearish']

        report.append("\n**看涨反转信号:**")
        if len(bullish) > 0:
            valid_ret = bullish['future_return_10d'].dropna()
            if len(valid_ret) > 0:
                report.append(f"- 信号数量: {len(bullish)}")
                report.append(f"- 10天后平均收益: {valid_ret.mean():.2f}%")
                report.append(f"- 胜率: {(valid_ret > 0).mean() * 100:.1f}%")

        report.append("\n**看跌反转信号:**")
        if len(bearish) > 0:
            valid_ret = bearish['future_return_10d'].dropna()
            if len(valid_ret) > 0:
                report.append(f"- 信号数量: {len(bearish)}")
                report.append(f"- 10天后平均收益: {valid_ret.mean():.2f}%")
                report.append(f"- 负收益率: {(valid_ret < 0).mean() * 100:.1f}%")

    # 四、策略回测结果
    report.append("\n\n## 四、策略回测结果")

    # 趋势跟踪策略
    report.append("\n### 4.1 趋势跟踪策略")
    report.append("""
**策略规则:**
- 入场: ADX>25 + PDI>MDI + 均线多头排列(>=2)
- 出场: PDI<MDI 或 ADX<20
""")
    if df_trend_trades is not None and len(df_trend_trades) > 0:
        returns = df_trend_trades['return_pct']
        win_rate = (returns > 0).mean() * 100
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1

        report.append(f"\n**回测结果:**")
        report.append(f"- 交易次数: {len(df_trend_trades)}")
        report.append(f"- 平均收益: {returns.mean():.2f}%")
        report.append(f"- 胜率: {win_rate:.1f}%")
        report.append(f"- 平均盈利: {avg_win:.2f}%")
        report.append(f"- 平均亏损: {avg_loss:.2f}%")
        report.append(f"- 盈亏比: {avg_win/avg_loss:.2f}" if avg_loss != 0 else "- 盈亏比: N/A")

    # 趋势过滤器策略
    report.append("\n### 4.2 趋势过滤器策略")
    report.append("""
**策略规则:**
- 入场: ADX>25 + PDI>MDI + 综合趋势得分>=3
- 出场: 综合趋势得分<=0 或 ADX<25
""")
    if df_filter_trades is not None and len(df_filter_trades) > 0:
        returns = df_filter_trades['return_pct']
        win_rate = (returns > 0).mean() * 100
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1

        report.append(f"\n**回测结果:**")
        report.append(f"- 交易次数: {len(df_filter_trades)}")
        report.append(f"- 平均收益: {returns.mean():.2f}%")
        report.append(f"- 胜率: {win_rate:.1f}%")
        report.append(f"- 盈亏比: {avg_win/avg_loss:.2f}" if avg_loss != 0 else "- 盈亏比: N/A")

    # 趋势+动量结合策略
    report.append("\n### 4.3 趋势+动量结合策略")
    report.append("""
**策略规则:**
- 入场: ADX>25 + PDI>MDI + 均线多头 + RSI在40-70 + MACD>0或上升
- 出场: PDI<MDI 或 ADX<20 或 RSI>80
""")
    if df_momentum_trades is not None and len(df_momentum_trades) > 0:
        returns = df_momentum_trades['return_pct']
        win_rate = (returns > 0).mean() * 100
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1

        report.append(f"\n**回测结果:**")
        report.append(f"- 交易次数: {len(df_momentum_trades)}")
        report.append(f"- 平均收益: {returns.mean():.2f}%")
        report.append(f"- 胜率: {win_rate:.1f}%")
        report.append(f"- 盈亏比: {avg_win/avg_loss:.2f}" if avg_loss != 0 else "- 盈亏比: N/A")

    # 五、策略对比
    report.append("\n\n## 五、策略对比")
    report.append("\n| 策略 | 交易次数 | 平均收益 | 胜率 | 盈亏比 |")
    report.append("|------|----------|----------|------|--------|")

    strategies = [
        ("趋势跟踪", df_trend_trades),
        ("趋势过滤器", df_filter_trades),
        ("趋势+动量", df_momentum_trades)
    ]

    for name, trades in strategies:
        if trades is not None and len(trades) > 0:
            returns = trades['return_pct']
            win_rate = (returns > 0).mean() * 100
            avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
            pf = avg_win/avg_loss if avg_loss != 0 else 0
            report.append(f"| {name} | {len(trades)} | {returns.mean():.2f}% | {win_rate:.1f}% | {pf:.2f} |")

    # 六、结论与建议
    report.append("\n\n## 六、结论与建议")

    report.append("\n### 6.1 关键发现")
    report.append("""
1. **ADX是有效的趋势强度指标**: ADX>25是判断趋势存在的有效阈值
2. **均线排列与趋势高度相关**: 完美多头/空头排列通常伴随强趋势
3. **趋势有惯性**: 一旦形成趋势，平均持续时间较长
4. **结合多指标效果更好**: 单一指标不如多指标组合可靠
""")

    report.append("\n### 6.2 实用建议")
    report.append("""
1. **趋势跟踪**:
   - 使用ADX>25作为趋势确认
   - 配合均线排列过滤假信号
   - 在趋势减弱时及时退出

2. **趋势过滤**:
   - 只在ADX>25的市场进行交易
   - 避免在盘整市场频繁操作
   - 趋势消失时转向观望

3. **风险管理**:
   - 趋势反转信号需要确认
   - 设置合理止损
   - 控制单笔仓位
""")

    report.append("\n### 6.3 注意事项")
    report.append("""
- ADX是滞后指标，趋势初期可能错过
- 极强趋势（ADX>60）可能预示反转临近
- 需结合基本面和市场环境综合判断
- 本研究基于历史数据，不代表未来表现
""")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'trend_strength_research_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n报告已保存到: {report_path}")

    return report_path

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("趋势强度指标研究分析")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 分析趋势指标
    df_all, latest_data = analyze_trend_indicators()

    # 2. 强趋势股票特征
    strong_up, strong_down = analyze_strong_trend_characteristics(df_all, latest_data)

    # 3. 趋势持续时间
    df_durations = analyze_trend_duration(df_all)

    # 4. 趋势反转信号
    df_signals = analyze_trend_reversal_signals(df_all)

    # 5. 趋势跟踪策略
    df_trend_trades = backtest_trend_following_strategy(df_all)

    # 6. 趋势过滤器策略
    df_filter_trades = backtest_trend_filter_strategy(df_all)

    # 7. 趋势+动量结合策略
    df_momentum_trades = backtest_trend_momentum_strategy(df_all)

    # 8. 创建可视化
    chart_path = create_visualizations(df_all, latest_data)

    # 9. 生成报告
    report_path = generate_report(df_all, latest_data, df_durations, df_signals,
                                  df_trend_trades, df_filter_trades, df_momentum_trades)

    print("\n" + "=" * 60)
    print("研究分析完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 报告: {report_path}")
    print(f"  - 图表: {chart_path}")

if __name__ == "__main__":
    main()
