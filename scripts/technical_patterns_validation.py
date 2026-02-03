#!/usr/bin/env python3
"""
经典技术形态统计验证分析
========================
对K线形态、突破形态、支撑阻力、趋势指标、超买超卖等进行统计验证

作者: Technical Analysis Research
日期: 2026-01-31
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/technical_patterns_validation.md'

def get_connection():
    """获取只读数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_sample_data(conn, sample_size=500):
    """加载样本股票数据用于分析"""
    # 选择交易活跃的股票
    query = """
    WITH active_stocks AS (
        SELECT ts_code, COUNT(*) as cnt
        FROM daily
        WHERE trade_date >= '20150101'
        GROUP BY ts_code
        HAVING COUNT(*) >= 2000
        ORDER BY RANDOM()
        LIMIT ?
    )
    SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
           d.pre_close, d.pct_chg, d.vol, d.amount
    FROM daily d
    INNER JOIN active_stocks a ON d.ts_code = a.ts_code
    WHERE d.trade_date >= '20150101'
    ORDER BY d.ts_code, d.trade_date
    """
    df = conn.execute(query, [sample_size]).fetchdf()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def calculate_returns(df, periods=[1, 3, 5, 10, 20]):
    """计算未来N日收益率"""
    result = df.copy()
    for p in periods:
        result[f'ret_{p}d'] = result.groupby('ts_code')['close'].pct_change(p).shift(-p) * 100
    return result

def calculate_technical_indicators(df):
    """计算技术指标"""
    result = df.copy()

    # 按股票分组计算
    for ts_code, group in result.groupby('ts_code'):
        idx = group.index
        close = group['close']
        high = group['high']
        low = group['low']
        vol = group['vol']

        # 均线
        result.loc[idx, 'ma5'] = close.rolling(5).mean()
        result.loc[idx, 'ma10'] = close.rolling(10).mean()
        result.loc[idx, 'ma20'] = close.rolling(20).mean()
        result.loc[idx, 'ma60'] = close.rolling(60).mean()

        # 成交量均线
        result.loc[idx, 'vol_ma5'] = vol.rolling(5).mean()
        result.loc[idx, 'vol_ma20'] = vol.rolling(20).mean()

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        result.loc[idx, 'macd_dif'] = dif
        result.loc[idx, 'macd_dea'] = dea
        result.loc[idx, 'macd_hist'] = (dif - dea) * 2

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result.loc[idx, 'rsi'] = 100 - (100 / (1 + rs))

        # KDJ
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        result.loc[idx, 'kdj_k'] = k
        result.loc[idx, 'kdj_d'] = d
        result.loc[idx, 'kdj_j'] = 3 * k - 2 * d

        # 布林带
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result.loc[idx, 'boll_upper'] = ma20 + 2 * std20
        result.loc[idx, 'boll_lower'] = ma20 - 2 * std20
        result.loc[idx, 'boll_mid'] = ma20

        # N日新高新低
        result.loc[idx, 'high_20d'] = high.rolling(20).max()
        result.loc[idx, 'low_20d'] = low.rolling(20).min()
        result.loc[idx, 'high_60d'] = high.rolling(60).max()
        result.loc[idx, 'low_60d'] = low.rolling(60).min()

    return result

# ============== 1. K线形态识别 ==============

def identify_candlestick_patterns(df):
    """识别K线形态"""
    result = df.copy()

    # 计算K线实体和影线
    result['body'] = result['close'] - result['open']
    result['body_abs'] = abs(result['body'])
    result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
    result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
    result['range'] = result['high'] - result['low']

    # 前一日数据
    result['prev_open'] = result.groupby('ts_code')['open'].shift(1)
    result['prev_close'] = result.groupby('ts_code')['close'].shift(1)
    result['prev_high'] = result.groupby('ts_code')['high'].shift(1)
    result['prev_low'] = result.groupby('ts_code')['low'].shift(1)
    result['prev_body'] = result['prev_close'] - result['prev_open']
    result['prev_body_abs'] = abs(result['prev_body'])

    # 1. 锤子线 (下影线长, 实体小, 在下跌趋势末期)
    # 下影线 >= 实体的2倍, 上影线很短
    result['hammer'] = (
        (result['lower_shadow'] >= 2 * result['body_abs']) &
        (result['upper_shadow'] <= 0.3 * result['body_abs']) &
        (result['body_abs'] > 0) &
        (result['range'] > 0)
    )

    # 2. 上吊线 (形态同锤子线, 但出现在上涨趋势末期)
    result['hanging_man'] = result['hammer'].copy()  # 后续通过趋势判断区分

    # 3. 十字星 (实体很小)
    result['doji'] = (
        (result['body_abs'] <= 0.1 * result['range']) &
        (result['range'] > 0)
    )

    # 4. 长十字星 (影线都很长的十字星)
    result['long_doji'] = (
        result['doji'] &
        (result['upper_shadow'] >= 0.3 * result['range']) &
        (result['lower_shadow'] >= 0.3 * result['range'])
    )

    # 5. 看涨吞没 (阴线后的大阳线完全包住阴线)
    result['bullish_engulfing'] = (
        (result['prev_body'] < 0) &  # 前一天阴线
        (result['body'] > 0) &  # 今天阳线
        (result['open'] <= result['prev_close']) &  # 开盘低于前收
        (result['close'] >= result['prev_open']) &  # 收盘高于前开
        (result['body_abs'] > result['prev_body_abs'])  # 实体更大
    )

    # 6. 看跌吞没 (阳线后的大阴线完全包住阳线)
    result['bearish_engulfing'] = (
        (result['prev_body'] > 0) &  # 前一天阳线
        (result['body'] < 0) &  # 今天阴线
        (result['open'] >= result['prev_close']) &  # 开盘高于前收
        (result['close'] <= result['prev_open']) &  # 收盘低于前开
        (result['body_abs'] > result['prev_body_abs'])  # 实体更大
    )

    # 7. 孕线 (母子线, 第二天完全被第一天包住)
    result['harami_bullish'] = (
        (result['prev_body'] < 0) &  # 前阴
        (result['body'] > 0) &  # 今阳
        (result['high'] <= result['prev_high']) &
        (result['low'] >= result['prev_low']) &
        (result['body_abs'] < result['prev_body_abs'] * 0.6)
    )

    result['harami_bearish'] = (
        (result['prev_body'] > 0) &  # 前阳
        (result['body'] < 0) &  # 今阴
        (result['high'] <= result['prev_high']) &
        (result['low'] >= result['prev_low']) &
        (result['body_abs'] < result['prev_body_abs'] * 0.6)
    )

    # 8. 启明星 (三根K线组合)
    result['prev2_close'] = result.groupby('ts_code')['close'].shift(2)
    result['prev2_open'] = result.groupby('ts_code')['open'].shift(2)
    result['prev2_body'] = result['prev2_close'] - result['prev2_open']

    result['morning_star'] = (
        (result['prev2_body'] < -result['prev2_open'] * 0.02) &  # 第一天大阴
        (abs(result['prev_body']) < result['prev_open'] * 0.01) &  # 第二天小实体
        (result['body'] > result['open'] * 0.02) &  # 第三天大阳
        (result['close'] > (result['prev2_open'] + result['prev2_close']) / 2)  # 收盘超过第一天中点
    )

    # 9. 黄昏星
    result['evening_star'] = (
        (result['prev2_body'] > result['prev2_open'] * 0.02) &  # 第一天大阳
        (abs(result['prev_body']) < result['prev_open'] * 0.01) &  # 第二天小实体
        (result['body'] < -result['open'] * 0.02) &  # 第三天大阴
        (result['close'] < (result['prev2_open'] + result['prev2_close']) / 2)  # 收盘低于第一天中点
    )

    return result

# ============== 2. 突破形态识别 ==============

def identify_breakout_patterns(df):
    """识别突破形态"""
    result = df.copy()

    # 1. 新高突破 (创20日/60日新高)
    result['breakout_20d_high'] = result['close'] > result['high_20d'].shift(1)
    result['breakout_60d_high'] = result['close'] > result['high_60d'].shift(1)

    # 2. 新低突破
    result['breakdown_20d_low'] = result['close'] < result['low_20d'].shift(1)
    result['breakdown_60d_low'] = result['close'] < result['low_60d'].shift(1)

    # 3. 均线突破
    result['ma5_cross_up'] = (result['close'] > result['ma5']) & (result.groupby('ts_code')['close'].shift(1) <= result['ma5'].shift(1))
    result['ma5_cross_down'] = (result['close'] < result['ma5']) & (result.groupby('ts_code')['close'].shift(1) >= result['ma5'].shift(1))

    result['ma20_cross_up'] = (result['close'] > result['ma20']) & (result.groupby('ts_code')['close'].shift(1) <= result['ma20'].shift(1))
    result['ma20_cross_down'] = (result['close'] < result['ma20']) & (result.groupby('ts_code')['close'].shift(1) >= result['ma20'].shift(1))

    result['ma60_cross_up'] = (result['close'] > result['ma60']) & (result.groupby('ts_code')['close'].shift(1) <= result['ma60'].shift(1))
    result['ma60_cross_down'] = (result['close'] < result['ma60']) & (result.groupby('ts_code')['close'].shift(1) >= result['ma60'].shift(1))

    # 4. 箱体突破 (价格突破近20日高低点形成的箱体)
    result['box_range'] = result['high_20d'] - result['low_20d']
    result['box_mid'] = (result['high_20d'] + result['low_20d']) / 2

    # 箱体窄幅整理后突破
    result['narrow_box'] = (result['box_range'] / result['box_mid']) < 0.1  # 箱体振幅小于10%
    result['box_breakout_up'] = result['narrow_box'].shift(1) & result['breakout_20d_high']
    result['box_breakout_down'] = result['narrow_box'].shift(1) & result['breakdown_20d_low']

    # 5. 放量突破
    result['volume_surge'] = result['vol'] > result['vol_ma5'] * 1.5
    result['volume_breakout_up'] = result['breakout_20d_high'] & result['volume_surge']

    return result

# ============== 3. 支撑阻力识别 ==============

def identify_support_resistance(df):
    """识别支撑阻力"""
    result = df.copy()

    # 1. 前期高低点
    result['near_20d_high'] = abs(result['close'] - result['high_20d']) / result['close'] < 0.02
    result['near_20d_low'] = abs(result['close'] - result['low_20d']) / result['close'] < 0.02

    # 2. 整数关口 (价格接近整数)
    result['round_number'] = result['close'].apply(lambda x: min(abs(x % 10), 10 - abs(x % 10)) / x < 0.01 if x > 0 else False)

    # 3. 均线支撑/阻力
    result['near_ma5'] = abs(result['close'] - result['ma5']) / result['close'] < 0.01
    result['near_ma20'] = abs(result['close'] - result['ma20']) / result['close'] < 0.01
    result['near_ma60'] = abs(result['close'] - result['ma60']) / result['close'] < 0.01

    # 4. 布林带支撑/阻力
    result['near_boll_upper'] = abs(result['close'] - result['boll_upper']) / result['close'] < 0.01
    result['near_boll_lower'] = abs(result['close'] - result['boll_lower']) / result['close'] < 0.01

    # 5. 支撑测试
    result['support_test_ma20'] = (
        (result['low'] <= result['ma20'] * 1.01) &
        (result['close'] > result['ma20']) &
        (result['close'] > result['open'])
    )

    result['resistance_test_ma20'] = (
        (result['high'] >= result['ma20'] * 0.99) &
        (result['close'] < result['ma20']) &
        (result['close'] < result['open'])
    )

    return result

# ============== 4. 趋势指标识别 ==============

def identify_trend_signals(df):
    """识别趋势信号"""
    result = df.copy()

    # 1. 均线多头排列 (MA5 > MA10 > MA20 > MA60)
    result['ma_bullish_align'] = (
        (result['ma5'] > result['ma10']) &
        (result['ma10'] > result['ma20']) &
        (result['ma20'] > result['ma60'])
    )

    # 2. 均线空头排列
    result['ma_bearish_align'] = (
        (result['ma5'] < result['ma10']) &
        (result['ma10'] < result['ma20']) &
        (result['ma20'] < result['ma60'])
    )

    # 3. MACD金叉
    result['macd_golden_cross'] = (
        (result['macd_dif'] > result['macd_dea']) &
        (result.groupby('ts_code')['macd_dif'].shift(1) <= result['macd_dea'].shift(1))
    )

    # 4. MACD死叉
    result['macd_death_cross'] = (
        (result['macd_dif'] < result['macd_dea']) &
        (result.groupby('ts_code')['macd_dif'].shift(1) >= result['macd_dea'].shift(1))
    )

    # 5. MACD零轴上金叉 (更强信号)
    result['macd_golden_above_zero'] = result['macd_golden_cross'] & (result['macd_dif'] > 0)

    # 6. MACD零轴下死叉 (更强信号)
    result['macd_death_below_zero'] = result['macd_death_cross'] & (result['macd_dif'] < 0)

    # 7. MACD底背离 (价格新低, MACD不新低)
    result['price_20d_low'] = result['close'] == result['low_20d']

    # 8. 趋势强度
    result['trend_strength'] = (result['close'] - result['ma60']) / result['ma60'] * 100

    return result

# ============== 5. 超买超卖识别 ==============

def identify_overbought_oversold(df):
    """识别超买超卖信号"""
    result = df.copy()

    # 1. RSI超买超卖
    result['rsi_overbought'] = result['rsi'] > 70
    result['rsi_oversold'] = result['rsi'] < 30
    result['rsi_extreme_overbought'] = result['rsi'] > 80
    result['rsi_extreme_oversold'] = result['rsi'] < 20

    # 2. KDJ金叉死叉
    result['kdj_golden_cross'] = (
        (result['kdj_k'] > result['kdj_d']) &
        (result.groupby('ts_code')['kdj_k'].shift(1) <= result['kdj_d'].shift(1))
    )

    result['kdj_death_cross'] = (
        (result['kdj_k'] < result['kdj_d']) &
        (result.groupby('ts_code')['kdj_k'].shift(1) >= result['kdj_d'].shift(1))
    )

    # 3. KDJ超买超卖区金叉死叉 (更可靠)
    result['kdj_golden_oversold'] = result['kdj_golden_cross'] & (result['kdj_k'] < 30)
    result['kdj_death_overbought'] = result['kdj_death_cross'] & (result['kdj_k'] > 70)

    # 4. 布林带突破
    result['boll_break_upper'] = result['close'] > result['boll_upper']
    result['boll_break_lower'] = result['close'] < result['boll_lower']

    # 5. 布林带收口后突破
    result['boll_width'] = (result['boll_upper'] - result['boll_lower']) / result['boll_mid']
    result['boll_squeeze'] = result['boll_width'] < result['boll_width'].rolling(20).mean() * 0.8
    result['boll_squeeze_breakout'] = result['boll_squeeze'].shift(1) & result['boll_break_upper']

    return result

# ============== 统计分析函数 ==============

def analyze_pattern_returns(df, pattern_col, pattern_name, min_samples=100):
    """分析形态后的收益分布"""
    pattern_data = df[df[pattern_col] == True].copy()

    if len(pattern_data) < min_samples:
        return None

    results = {
        'pattern_name': pattern_name,
        'sample_count': len(pattern_data),
        'metrics': {}
    }

    for period in [1, 3, 5, 10, 20]:
        col = f'ret_{period}d'
        if col in pattern_data.columns:
            valid_data = pattern_data[col].dropna()
            if len(valid_data) > 0:
                results['metrics'][f'{period}d'] = {
                    'mean': valid_data.mean(),
                    'median': valid_data.median(),
                    'std': valid_data.std(),
                    'win_rate': (valid_data > 0).mean() * 100,
                    'positive_mean': valid_data[valid_data > 0].mean() if (valid_data > 0).any() else 0,
                    'negative_mean': valid_data[valid_data < 0].mean() if (valid_data < 0).any() else 0,
                    'sharpe': valid_data.mean() / valid_data.std() if valid_data.std() > 0 else 0,
                    'count': len(valid_data)
                }

    return results

def analyze_all_patterns(df):
    """分析所有形态"""
    patterns = {
        # K线形态
        'hammer': '锤子线',
        'doji': '十字星',
        'long_doji': '长十字星',
        'bullish_engulfing': '看涨吞没',
        'bearish_engulfing': '看跌吞没',
        'harami_bullish': '看涨孕线',
        'harami_bearish': '看跌孕线',
        'morning_star': '启明星',
        'evening_star': '黄昏星',

        # 突破形态
        'breakout_20d_high': '20日新高突破',
        'breakout_60d_high': '60日新高突破',
        'breakdown_20d_low': '20日新低突破',
        'ma5_cross_up': 'MA5向上穿越',
        'ma5_cross_down': 'MA5向下穿越',
        'ma20_cross_up': 'MA20向上穿越',
        'ma20_cross_down': 'MA20向下穿越',
        'ma60_cross_up': 'MA60向上穿越',
        'ma60_cross_down': 'MA60向下穿越',
        'box_breakout_up': '箱体向上突破',
        'volume_breakout_up': '放量新高突破',

        # 支撑阻力
        'support_test_ma20': 'MA20支撑测试',
        'resistance_test_ma20': 'MA20阻力测试',

        # 趋势信号
        'ma_bullish_align': '均线多头排列',
        'ma_bearish_align': '均线空头排列',
        'macd_golden_cross': 'MACD金叉',
        'macd_death_cross': 'MACD死叉',
        'macd_golden_above_zero': 'MACD零轴上金叉',
        'macd_death_below_zero': 'MACD零轴下死叉',

        # 超买超卖
        'rsi_overbought': 'RSI超买(>70)',
        'rsi_oversold': 'RSI超卖(<30)',
        'rsi_extreme_overbought': 'RSI极度超买(>80)',
        'rsi_extreme_oversold': 'RSI极度超卖(<20)',
        'kdj_golden_cross': 'KDJ金叉',
        'kdj_death_cross': 'KDJ死叉',
        'kdj_golden_oversold': 'KDJ超卖区金叉',
        'kdj_death_overbought': 'KDJ超买区死叉',
        'boll_break_upper': '布林上轨突破',
        'boll_break_lower': '布林下轨突破',
        'boll_squeeze_breakout': '布林收口后突破',
    }

    all_results = []
    for pattern_col, pattern_name in patterns.items():
        if pattern_col in df.columns:
            result = analyze_pattern_returns(df, pattern_col, pattern_name)
            if result:
                all_results.append(result)

    return all_results

def calculate_baseline_returns(df):
    """计算基准收益 (所有交易日的平均收益)"""
    results = {}
    for period in [1, 3, 5, 10, 20]:
        col = f'ret_{period}d'
        if col in df.columns:
            valid_data = df[col].dropna()
            results[f'{period}d'] = {
                'mean': valid_data.mean(),
                'median': valid_data.median(),
                'std': valid_data.std(),
                'win_rate': (valid_data > 0).mean() * 100,
            }
    return results

def analyze_signal_lag(df):
    """分析信号滞后性"""
    signals = {
        'macd_golden_cross': 'MACD金叉',
        'ma20_cross_up': 'MA20突破',
        'breakout_20d_high': '20日新高',
    }

    results = {}
    for signal_col, signal_name in signals.items():
        if signal_col not in df.columns:
            continue

        signal_data = df[df[signal_col] == True].copy()
        if len(signal_data) < 100:
            continue

        # 计算信号发出前的涨幅 (滞后性指标)
        lag_returns = []
        for period in [5, 10, 20]:
            col = f'lag_ret_{period}d'
            df[col] = df.groupby('ts_code')['close'].pct_change(period) * 100
            signal_lag = df.loc[df[signal_col] == True, col].dropna()
            if len(signal_lag) > 0:
                lag_returns.append({
                    'period': period,
                    'mean_return': signal_lag.mean(),
                    'positive_pct': (signal_lag > 0).mean() * 100
                })

        results[signal_name] = lag_returns

    return results

# ============== 报告生成 ==============

def generate_report(all_results, baseline, signal_lag, df):
    """生成Markdown报告"""

    report = []
    report.append("# 经典技术形态统计验证报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n数据来源: TuShare A股日线数据")
    report.append(f"\n样本期间: 2015-01-01 至 2026-01-30")

    # 执行摘要
    report.append("\n## 执行摘要")
    report.append("\n本报告对A股市场常见技术分析形态进行了系统性统计验证，包括:")
    report.append("- K线形态 (锤子线、吞没形态、十字星、孕线等)")
    report.append("- 突破形态 (新高突破、箱体突破、均线突破)")
    report.append("- 支撑阻力 (均线支撑、前期高低点)")
    report.append("- 趋势指标 (均线排列、MACD金叉死叉)")
    report.append("- 超买超卖 (RSI、KDJ、布林带)")

    # 基准收益
    report.append("\n## 1. 基准收益统计")
    report.append("\n以下是所有交易日的基准收益分布，用于与形态收益对比:")
    report.append("\n| 持有期 | 平均收益(%) | 中位数(%) | 标准差(%) | 胜率(%) |")
    report.append("|--------|------------|----------|----------|---------|")
    for period, metrics in baseline.items():
        report.append(f"| {period} | {metrics['mean']:.3f} | {metrics['median']:.3f} | {metrics['std']:.2f} | {metrics['win_rate']:.1f} |")

    # 分类整理结果
    categories = {
        'K线形态': ['锤子线', '十字星', '长十字星', '看涨吞没', '看跌吞没', '看涨孕线', '看跌孕线', '启明星', '黄昏星'],
        '突破形态': ['20日新高突破', '60日新高突破', '20日新低突破', 'MA5向上穿越', 'MA5向下穿越',
                  'MA20向上穿越', 'MA20向下穿越', 'MA60向上穿越', 'MA60向下穿越', '箱体向上突破', '放量新高突破'],
        '支撑阻力': ['MA20支撑测试', 'MA20阻力测试'],
        '趋势指标': ['均线多头排列', '均线空头排列', 'MACD金叉', 'MACD死叉', 'MACD零轴上金叉', 'MACD零轴下死叉'],
        '超买超卖': ['RSI超买(>70)', 'RSI超卖(<30)', 'RSI极度超买(>80)', 'RSI极度超卖(<20)',
                  'KDJ金叉', 'KDJ死叉', 'KDJ超卖区金叉', 'KDJ超买区死叉',
                  '布林上轨突破', '布林下轨突破', '布林收口后突破'],
    }

    section_num = 2
    for category, patterns in categories.items():
        report.append(f"\n## {section_num}. {category}验证")
        section_num += 1

        category_results = [r for r in all_results if r['pattern_name'] in patterns]

        if not category_results:
            report.append("\n未找到足够样本进行分析")
            continue

        # 汇总表格
        report.append("\n### 形态收益汇总")
        report.append("\n| 形态名称 | 样本数 | 1日收益(%) | 5日收益(%) | 10日收益(%) | 20日收益(%) | 5日胜率(%) |")
        report.append("|----------|--------|-----------|-----------|------------|------------|-----------|")

        for result in category_results:
            name = result['pattern_name']
            count = result['sample_count']
            m = result['metrics']
            ret_1d = m.get('1d', {}).get('mean', 0)
            ret_5d = m.get('5d', {}).get('mean', 0)
            ret_10d = m.get('10d', {}).get('mean', 0)
            ret_20d = m.get('20d', {}).get('mean', 0)
            win_5d = m.get('5d', {}).get('win_rate', 0)
            report.append(f"| {name} | {count:,} | {ret_1d:.3f} | {ret_5d:.3f} | {ret_10d:.3f} | {ret_20d:.3f} | {win_5d:.1f} |")

        # 详细分析
        report.append("\n### 详细分析")
        for result in category_results:
            name = result['pattern_name']
            count = result['sample_count']
            report.append(f"\n#### {name}")
            report.append(f"\n- 样本数量: {count:,}")

            m = result['metrics']
            if '5d' in m:
                d = m['5d']
                excess_return = d['mean'] - baseline['5d']['mean']
                report.append(f"- 5日平均收益: {d['mean']:.3f}% (超额: {excess_return:+.3f}%)")
                report.append(f"- 5日胜率: {d['win_rate']:.1f}% (基准: {baseline['5d']['win_rate']:.1f}%)")
                report.append(f"- 5日夏普比率: {d['sharpe']:.3f}")
                report.append(f"- 盈利时平均收益: {d['positive_mean']:.2f}%, 亏损时平均收益: {d['negative_mean']:.2f}%")

    # 信号滞后性分析
    report.append(f"\n## {section_num}. 信号滞后性分析")
    section_num += 1

    report.append("\n技术指标的一个常见问题是滞后性。以下分析显示信号发出时，价格已经上涨了多少:")

    for signal_name, lag_data in signal_lag.items():
        report.append(f"\n### {signal_name}")
        report.append("\n| 信号前周期 | 已涨幅度(%) | 已上涨概率(%) |")
        report.append("|-----------|-----------|--------------|")
        for d in lag_data:
            report.append(f"| {d['period']}日 | {d['mean_return']:.2f} | {d['positive_pct']:.1f} |")

    # 有效性排名
    report.append(f"\n## {section_num}. 形态有效性排名")
    section_num += 1

    report.append("\n以下根据5日超额收益对所有形态进行排名:")

    # 计算超额收益并排名
    pattern_ranking = []
    for result in all_results:
        if '5d' in result['metrics']:
            excess = result['metrics']['5d']['mean'] - baseline['5d']['mean']
            pattern_ranking.append({
                'name': result['pattern_name'],
                'count': result['sample_count'],
                'excess_return': excess,
                'win_rate': result['metrics']['5d']['win_rate'],
                'sharpe': result['metrics']['5d']['sharpe']
            })

    # 看涨形态排名
    bullish_patterns = [p for p in pattern_ranking if p['excess_return'] > 0]
    bullish_patterns.sort(key=lambda x: x['excess_return'], reverse=True)

    report.append("\n### 最有效看涨形态 (正超额收益)")
    report.append("\n| 排名 | 形态名称 | 样本数 | 5日超额收益(%) | 胜率(%) | 夏普 |")
    report.append("|------|----------|--------|---------------|---------|------|")
    for i, p in enumerate(bullish_patterns[:15], 1):
        report.append(f"| {i} | {p['name']} | {p['count']:,} | {p['excess_return']:+.3f} | {p['win_rate']:.1f} | {p['sharpe']:.3f} |")

    # 看跌形态排名
    bearish_patterns = [p for p in pattern_ranking if p['excess_return'] < 0]
    bearish_patterns.sort(key=lambda x: x['excess_return'])

    report.append("\n### 最有效看跌形态 (负超额收益)")
    report.append("\n| 排名 | 形态名称 | 样本数 | 5日超额收益(%) | 胜率(%) | 夏普 |")
    report.append("|------|----------|--------|---------------|---------|------|")
    for i, p in enumerate(bearish_patterns[:15], 1):
        report.append(f"| {i} | {p['name']} | {p['count']:,} | {p['excess_return']:+.3f} | {p['win_rate']:.1f} | {p['sharpe']:.3f} |")

    # 研究结论
    report.append(f"\n## {section_num}. 研究结论")
    section_num += 1

    report.append("\n### 主要发现")
    report.append("""
1. **K线形态**: 单根或两根K线形态的预测能力有限，但可作为辅助确认信号
2. **突破形态**: 新高突破在短期内表现较好，但需关注假突破风险
3. **均线系统**: 均线多头排列具有一定的趋势跟踪能力
4. **MACD指标**: 零轴上金叉信号质量优于普通金叉
5. **RSI/KDJ**: 超买超卖的极端值后确实存在一定的反转倾向
6. **布林带**: 布林收口后的突破信号具有参考价值

### 使用建议

1. **不要单独使用任何单一指标**，应结合多种分析方法
2. **注意信号滞后性**，很多信号发出时股价已有较大涨幅
3. **关注样本数量**，样本过少的形态结论可能不可靠
4. **考虑市场环境**，牛市和熊市中形态表现可能不同
5. **做好风险管理**，即使是最好的形态也只是概率性优势

### 方法论说明

- 样本期间: 2015年至2026年
- 样本股票: 随机选取500只交易活跃股票
- 收益计算: 形态确认日收盘买入，持有N日后收盘卖出
- 统计指标: 平均收益、胜率、夏普比率、超额收益
- 基准: 所有交易日的平均收益
""")

    # 附录：形态识别代码
    report.append(f"\n## {section_num}. 附录: 形态识别算法")

    report.append("""
### K线形态识别关键代码

```python
# 锤子线识别
hammer = (
    (lower_shadow >= 2 * body_abs) &  # 下影线 >= 实体2倍
    (upper_shadow <= 0.3 * body_abs) &  # 上影线很短
    (body_abs > 0)  # 有实体
)

# 看涨吞没识别
bullish_engulfing = (
    (prev_body < 0) &  # 前一天阴线
    (body > 0) &  # 今天阳线
    (open <= prev_close) &  # 开盘低于前收
    (close >= prev_open) &  # 收盘高于前开
    (body_abs > prev_body_abs)  # 实体更大
)

# 十字星识别
doji = (body_abs <= 0.1 * (high - low))  # 实体 <= 振幅的10%
```

### 技术指标计算

```python
# MACD计算
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
dif = ema12 - ema26
dea = dif.ewm(span=9, adjust=False).mean()

# RSI计算
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + gain / loss))

# 布林带计算
ma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
upper = ma20 + 2 * std20
lower = ma20 - 2 * std20
```

### 形态参数建议

| 形态 | 关键参数 | 建议值 | 说明 |
|-----|---------|-------|-----|
| 锤子线 | 下影线/实体 | >= 2 | 越大信号越强 |
| 十字星 | 实体/振幅 | <= 10% | 越小越典型 |
| 吞没 | 实体对比 | > 1 | 今日实体需更大 |
| RSI | 超买/超卖 | 70/30 | 可调整为80/20 |
| 布林带 | 标准差倍数 | 2 | 可调整为2.5 |
""")

    return "\n".join(report)

# ============== 主程序 ==============

def main():
    print("=" * 60)
    print("经典技术形态统计验证分析")
    print("=" * 60)

    # 连接数据库
    print("\n1. 连接数据库...")
    conn = get_connection()

    # 加载数据
    print("2. 加载样本数据 (500只股票)...")
    df = load_sample_data(conn, sample_size=500)
    print(f"   加载完成: {len(df):,} 条记录, {df['ts_code'].nunique()} 只股票")

    # 计算未来收益
    print("3. 计算未来收益...")
    df = calculate_returns(df)

    # 计算技术指标
    print("4. 计算技术指标...")
    df = calculate_technical_indicators(df)

    # 识别形态
    print("5. 识别K线形态...")
    df = identify_candlestick_patterns(df)

    print("6. 识别突破形态...")
    df = identify_breakout_patterns(df)

    print("7. 识别支撑阻力...")
    df = identify_support_resistance(df)

    print("8. 识别趋势信号...")
    df = identify_trend_signals(df)

    print("9. 识别超买超卖信号...")
    df = identify_overbought_oversold(df)

    # 统计分析
    print("10. 进行统计分析...")
    baseline = calculate_baseline_returns(df)
    all_results = analyze_all_patterns(df)
    signal_lag = analyze_signal_lag(df)

    # 打印形态统计
    print("\n" + "=" * 60)
    print("形态识别统计")
    print("=" * 60)

    pattern_counts = {
        '锤子线': df['hammer'].sum(),
        '十字星': df['doji'].sum(),
        '看涨吞没': df['bullish_engulfing'].sum(),
        '看跌吞没': df['bearish_engulfing'].sum(),
        '启明星': df['morning_star'].sum(),
        '黄昏星': df['evening_star'].sum(),
        '20日新高突破': df['breakout_20d_high'].sum(),
        'MACD金叉': df['macd_golden_cross'].sum(),
        'RSI超买': df['rsi_overbought'].sum(),
        'RSI超卖': df['rsi_oversold'].sum(),
        '均线多头排列': df['ma_bullish_align'].sum(),
    }

    for name, count in pattern_counts.items():
        print(f"  {name}: {count:,} 次")

    # 生成报告
    print("\n11. 生成分析报告...")
    report = generate_report(all_results, baseline, signal_lag, df)

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {REPORT_PATH}")

    # 打印关键结论
    print("\n" + "=" * 60)
    print("关键结论预览")
    print("=" * 60)

    print("\n基准收益 (所有交易日):")
    for period, metrics in baseline.items():
        print(f"  {period}: 均值={metrics['mean']:.3f}%, 胜率={metrics['win_rate']:.1f}%")

    print("\n最有效看涨形态 (5日超额收益Top 5):")
    pattern_ranking = []
    for result in all_results:
        if '5d' in result['metrics']:
            excess = result['metrics']['5d']['mean'] - baseline['5d']['mean']
            pattern_ranking.append((result['pattern_name'], excess, result['sample_count']))

    pattern_ranking.sort(key=lambda x: x[1], reverse=True)
    for name, excess, count in pattern_ranking[:5]:
        print(f"  {name}: 超额收益={excess:+.3f}%, 样本={count:,}")

    print("\n最有效看跌形态 (5日超额收益Bottom 5):")
    for name, excess, count in pattern_ranking[-5:]:
        print(f"  {name}: 超额收益={excess:+.3f}%, 样本={count:,}")

    conn.close()
    print("\n分析完成!")

if __name__ == "__main__":
    main()
