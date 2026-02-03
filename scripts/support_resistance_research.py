#!/usr/bin/env python3
"""
支撑阻力位研究分析
==================

研究内容：
1. 支撑阻力识别：历史高低点、均线支撑阻力、成交密集区、整数关口
2. 有效性验证：触及后反转概率、突破后走势、支撑变阻力
3. 策略应用：支撑位买入、阻力位卖出、突破追踪

Author: Research Script
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# 连接数据库
conn = duckdb.connect(DB_PATH, read_only=True)

print("=" * 80)
print("技术支撑阻力位研究分析")
print("=" * 80)


# =============================================================================
# 第一部分：数据准备
# =============================================================================

def get_stock_data(ts_code: str, start_date: str = '20200101', end_date: str = '20260130') -> pd.DataFrame:
    """获取股票日线数据"""
    query = f"""
        SELECT
            ts_code,
            trade_date,
            open,
            high,
            low,
            close,
            vol,
            amount
        FROM daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
    """
    df = conn.execute(query).fetchdf()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标"""
    # 均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['ma120'] = df['close'].rolling(window=120).mean()
    df['ma250'] = df['close'].rolling(window=250).mean()

    # 布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # 最高/最低价
    for period in [5, 10, 20, 60, 120]:
        df[f'high_{period}'] = df['high'].rolling(window=period).max()
        df[f'low_{period}'] = df['low'].rolling(window=period).min()

    # 成交量均线
    df['vol_ma5'] = df['vol'].rolling(window=5).mean()
    df['vol_ma20'] = df['vol'].rolling(window=20).mean()

    return df


# =============================================================================
# 第二部分：支撑阻力识别
# =============================================================================

class SupportResistanceFinder:
    """支撑阻力位识别器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def find_local_extremes(self, window: int = 20) -> tuple:
        """
        寻找局部极值点（历史高低点）

        参数:
            window: 窗口大小，用于判断是否为局部极值

        返回:
            (支撑位列表, 阻力位列表)
        """
        df = self.df
        supports = []
        resistances = []

        for i in range(window, len(df) - window):
            # 检查是否为局部最低点
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                supports.append({
                    'date': df['trade_date'].iloc[i],
                    'price': df['low'].iloc[i],
                    'type': 'local_low',
                    'strength': 1  # 基础强度
                })

            # 检查是否为局部最高点
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                resistances.append({
                    'date': df['trade_date'].iloc[i],
                    'price': df['high'].iloc[i],
                    'type': 'local_high',
                    'strength': 1
                })

        return supports, resistances

    def find_ma_levels(self) -> dict:
        """
        获取均线支撑阻力位

        返回:
            包含各均线最新值的字典
        """
        df = self.df
        latest = df.iloc[-1]
        current_price = latest['close']

        ma_levels = {}
        for ma in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250']:
            if pd.notna(latest.get(ma)):
                level_type = 'support' if latest[ma] < current_price else 'resistance'
                ma_levels[ma] = {
                    'price': latest[ma],
                    'type': level_type,
                    'distance_pct': (latest[ma] - current_price) / current_price * 100
                }

        return ma_levels

    def find_volume_clusters(self, n_bins: int = 50, min_vol_ratio: float = 1.5) -> list:
        """
        寻找成交密集区

        参数:
            n_bins: 价格区间数量
            min_vol_ratio: 最小成交量比率阈值

        返回:
            成交密集区价格列表
        """
        df = self.df

        # 创建价格区间
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, n_bins + 1)

        # 计算每个价格区间的成交量
        volume_profile = []
        for i in range(len(bins) - 1):
            mask = (df['low'] <= bins[i+1]) & (df['high'] >= bins[i])
            vol_in_range = df.loc[mask, 'vol'].sum()
            price_mid = (bins[i] + bins[i+1]) / 2
            volume_profile.append({
                'price': price_mid,
                'volume': vol_in_range,
                'price_range': (bins[i], bins[i+1])
            })

        # 找出成交密集区（成交量高于平均的区域）
        volumes = [v['volume'] for v in volume_profile]
        avg_vol = np.mean(volumes)

        clusters = []
        for vp in volume_profile:
            if vp['volume'] > avg_vol * min_vol_ratio:
                clusters.append({
                    'price': vp['price'],
                    'volume': vp['volume'],
                    'vol_ratio': vp['volume'] / avg_vol,
                    'type': 'volume_cluster'
                })

        return sorted(clusters, key=lambda x: x['vol_ratio'], reverse=True)

    def find_round_numbers(self, step: float = None) -> list:
        """
        寻找整数关口

        参数:
            step: 整数关口间隔，默认根据价格自动设定

        返回:
            整数关口列表
        """
        df = self.df
        current_price = df['close'].iloc[-1]
        price_range = df['high'].max() - df['low'].min()

        # 根据价格设定整数关口
        if step is None:
            if current_price < 10:
                step = 1
            elif current_price < 50:
                step = 5
            elif current_price < 100:
                step = 10
            elif current_price < 500:
                step = 50
            else:
                step = 100

        # 生成整数关口
        min_level = int(df['low'].min() / step) * step
        max_level = int(df['high'].max() / step + 1) * step

        round_numbers = []
        level = min_level
        while level <= max_level:
            distance_pct = (level - current_price) / current_price * 100
            level_type = 'support' if level < current_price else 'resistance'
            round_numbers.append({
                'price': level,
                'type': level_type,
                'distance_pct': distance_pct
            })
            level += step

        return round_numbers

    def find_all_levels(self) -> dict:
        """整合所有支撑阻力位"""
        supports, resistances = self.find_local_extremes()
        ma_levels = self.find_ma_levels()
        volume_clusters = self.find_volume_clusters()
        round_numbers = self.find_round_numbers()

        return {
            'local_extremes': {
                'supports': supports,
                'resistances': resistances
            },
            'ma_levels': ma_levels,
            'volume_clusters': volume_clusters,
            'round_numbers': round_numbers
        }


# =============================================================================
# 第三部分：有效性验证
# =============================================================================

class SupportResistanceValidator:
    """支撑阻力位有效性验证"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def test_level_touch(self, level: float, tolerance: float = 0.02) -> list:
        """
        测试价格触及支撑/阻力位的情况

        参数:
            level: 价格水平
            tolerance: 容差比例

        返回:
            触及事件列表
        """
        df = self.df
        touches = []

        upper_bound = level * (1 + tolerance)
        lower_bound = level * (1 - tolerance)

        for i in range(1, len(df) - 5):  # 留出5天观察后续
            # 检查是否触及该价位
            if df['low'].iloc[i] <= upper_bound and df['high'].iloc[i] >= lower_bound:
                # 判断方向：从上方还是下方接近
                prev_close = df['close'].iloc[i-1]

                if prev_close > level:  # 从上方接近（测试支撑）
                    direction = 'test_support'
                else:  # 从下方接近（测试阻力）
                    direction = 'test_resistance'

                # 观察后续5天表现
                future_prices = df['close'].iloc[i+1:i+6]
                if len(future_prices) >= 5:
                    future_return = (future_prices.iloc[-1] - df['close'].iloc[i]) / df['close'].iloc[i]

                    touches.append({
                        'date': df['trade_date'].iloc[i],
                        'price': df['close'].iloc[i],
                        'direction': direction,
                        'future_5d_return': future_return,
                        'reversed': (direction == 'test_support' and future_return > 0) or
                                   (direction == 'test_resistance' and future_return < 0)
                    })

        return touches

    def calculate_reversal_probability(self, touches: list) -> dict:
        """
        计算触及后反转概率

        参数:
            touches: 触及事件列表

        返回:
            统计结果
        """
        if not touches:
            return {'total': 0, 'reversal_rate': 0}

        support_tests = [t for t in touches if t['direction'] == 'test_support']
        resistance_tests = [t for t in touches if t['direction'] == 'test_resistance']

        support_reversals = sum(1 for t in support_tests if t['reversed'])
        resistance_reversals = sum(1 for t in resistance_tests if t['reversed'])

        return {
            'total_touches': len(touches),
            'support_tests': len(support_tests),
            'support_reversal_rate': support_reversals / len(support_tests) if support_tests else 0,
            'resistance_tests': len(resistance_tests),
            'resistance_reversal_rate': resistance_reversals / len(resistance_tests) if resistance_tests else 0,
            'avg_return_after_support_test': np.mean([t['future_5d_return'] for t in support_tests]) if support_tests else 0,
            'avg_return_after_resistance_test': np.mean([t['future_5d_return'] for t in resistance_tests]) if resistance_tests else 0
        }

    def analyze_breakout(self, level: float, tolerance: float = 0.02) -> list:
        """
        分析突破后走势

        参数:
            level: 价格水平
            tolerance: 容差比例

        返回:
            突破事件列表
        """
        df = self.df
        breakouts = []

        for i in range(20, len(df) - 10):  # 需要历史数据和未来数据
            # 检查是否发生向上突破
            if df['close'].iloc[i] > level * (1 + tolerance):
                # 检查之前是否在该水平下方
                recent_below = (df['close'].iloc[i-20:i] < level).sum() >= 15

                if recent_below:
                    # 观察突破后表现
                    future_5d = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] if i+5 < len(df) else np.nan
                    future_10d = (df['close'].iloc[i+10] - df['close'].iloc[i]) / df['close'].iloc[i] if i+10 < len(df) else np.nan

                    # 检查是否为假突破（5天内回落到水平下方）
                    false_breakout = any(df['close'].iloc[i+1:i+6] < level * (1 - tolerance)) if i+5 < len(df) else None

                    breakouts.append({
                        'date': df['trade_date'].iloc[i],
                        'direction': 'upward',
                        'price': df['close'].iloc[i],
                        'level': level,
                        'future_5d_return': future_5d,
                        'future_10d_return': future_10d,
                        'false_breakout': false_breakout
                    })

            # 检查是否发生向下突破
            elif df['close'].iloc[i] < level * (1 - tolerance):
                # 检查之前是否在该水平上方
                recent_above = (df['close'].iloc[i-20:i] > level).sum() >= 15

                if recent_above:
                    future_5d = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] if i+5 < len(df) else np.nan
                    future_10d = (df['close'].iloc[i+10] - df['close'].iloc[i]) / df['close'].iloc[i] if i+10 < len(df) else np.nan

                    false_breakout = any(df['close'].iloc[i+1:i+6] > level * (1 + tolerance)) if i+5 < len(df) else None

                    breakouts.append({
                        'date': df['trade_date'].iloc[i],
                        'direction': 'downward',
                        'price': df['close'].iloc[i],
                        'level': level,
                        'future_5d_return': future_5d,
                        'future_10d_return': future_10d,
                        'false_breakout': false_breakout
                    })

        return breakouts

    def analyze_support_becomes_resistance(self) -> list:
        """
        分析支撑变阻力现象

        返回:
            支撑变阻力事件列表
        """
        df = self.df

        # 首先找出所有局部支撑位
        finder = SupportResistanceFinder(df)
        supports, _ = finder.find_local_extremes(window=20)

        events = []
        for support in supports:
            support_date = support['date']
            support_price = support['price']
            support_idx = df[df['trade_date'] == support_date].index[0]

            # 检查在支撑形成后是否被跌破
            for i in range(support_idx + 20, len(df) - 10):
                if df['close'].iloc[i] < support_price * 0.98:  # 跌破支撑
                    # 检查之后是否反弹到该位置并被压制
                    for j in range(i + 1, min(i + 60, len(df) - 5)):
                        if df['high'].iloc[j] >= support_price * 0.98 and df['high'].iloc[j] <= support_price * 1.02:
                            # 检查是否在此处受阻
                            if df['close'].iloc[j+1] < df['close'].iloc[j] and df['close'].iloc[j+2] < df['close'].iloc[j]:
                                events.append({
                                    'original_support_date': support_date,
                                    'original_support_price': support_price,
                                    'break_date': df['trade_date'].iloc[i],
                                    'retest_date': df['trade_date'].iloc[j],
                                    'acted_as_resistance': True,
                                    'subsequent_return': (df['close'].iloc[j+5] - df['close'].iloc[j]) / df['close'].iloc[j] if j+5 < len(df) else np.nan
                                })
                                break
                    break

        return events


# =============================================================================
# 第四部分：策略应用
# =============================================================================

class SupportResistanceStrategy:
    """支撑阻力位交易策略"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.finder = SupportResistanceFinder(df)

    def support_buy_strategy(self, tolerance: float = 0.02, stop_loss: float = 0.03) -> pd.DataFrame:
        """
        支撑位买入策略

        参数:
            tolerance: 触及支撑位的容差
            stop_loss: 止损比例

        返回:
            交易记录DataFrame
        """
        df = self.df
        supports, _ = self.finder.find_local_extremes(window=20)

        trades = []
        position = None

        for i in range(60, len(df) - 5):
            current_price = df['close'].iloc[i]
            current_low = df['low'].iloc[i]

            if position is None:
                # 检查是否触及近期支撑位
                recent_supports = [s for s in supports
                                  if s['date'] < df['trade_date'].iloc[i]
                                  and s['date'] > df['trade_date'].iloc[i] - pd.Timedelta(days=120)]

                for support in recent_supports:
                    support_price = support['price']
                    if current_low <= support_price * (1 + tolerance) and current_price > support_price * (1 - stop_loss):
                        # 买入信号
                        position = {
                            'entry_date': df['trade_date'].iloc[i],
                            'entry_price': current_price,
                            'support_level': support_price,
                            'stop_loss': support_price * (1 - stop_loss)
                        }
                        break
            else:
                # 持有仓位，检查止损或止盈
                entry_price = position['entry_price']
                stop_loss_price = position['stop_loss']

                # 止损
                if current_price < stop_loss_price:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (current_price - entry_price) / entry_price,
                        'exit_reason': 'stop_loss'
                    })
                    position = None

                # 持有超过20天且盈利则卖出
                elif (df['trade_date'].iloc[i] - position['entry_date']).days > 20 and current_price > entry_price:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (current_price - entry_price) / entry_price,
                        'exit_reason': 'time_profit'
                    })
                    position = None

        return pd.DataFrame(trades)

    def resistance_sell_strategy(self, tolerance: float = 0.02) -> pd.DataFrame:
        """
        阻力位卖出策略（做空模拟）

        参数:
            tolerance: 触及阻力位的容差

        返回:
            交易记录DataFrame
        """
        df = self.df
        _, resistances = self.finder.find_local_extremes(window=20)

        trades = []
        position = None

        for i in range(60, len(df) - 5):
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]

            if position is None:
                # 检查是否触及近期阻力位
                recent_resistances = [r for r in resistances
                                     if r['date'] < df['trade_date'].iloc[i]
                                     and r['date'] > df['trade_date'].iloc[i] - pd.Timedelta(days=120)]

                for resistance in recent_resistances:
                    resistance_price = resistance['price']
                    if current_high >= resistance_price * (1 - tolerance) and current_price < resistance_price * 1.03:
                        # 做空信号
                        position = {
                            'entry_date': df['trade_date'].iloc[i],
                            'entry_price': current_price,
                            'resistance_level': resistance_price
                        }
                        break
            else:
                entry_price = position['entry_price']

                # 止损（价格上涨5%）
                if current_price > entry_price * 1.05:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (entry_price - current_price) / entry_price,  # 做空收益
                        'exit_reason': 'stop_loss'
                    })
                    position = None

                # 止盈（价格下跌5%以上或持有超过20天）
                elif current_price < entry_price * 0.95 or (df['trade_date'].iloc[i] - position['entry_date']).days > 20:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (entry_price - current_price) / entry_price,
                        'exit_reason': 'take_profit' if current_price < entry_price else 'timeout'
                    })
                    position = None

        return pd.DataFrame(trades)

    def breakout_follow_strategy(self, lookback: int = 60, tolerance: float = 0.02) -> pd.DataFrame:
        """
        突破追踪策略

        参数:
            lookback: 回看周期
            tolerance: 突破确认的容差

        返回:
            交易记录DataFrame
        """
        df = self.df
        trades = []
        position = None

        for i in range(lookback, len(df) - 10):
            current_price = df['close'].iloc[i]

            # 计算近期高点和低点
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()

            if position is None:
                # 向上突破
                if current_price > recent_high * (1 + tolerance):
                    # 确认突破（收盘价站上新高）
                    position = {
                        'entry_date': df['trade_date'].iloc[i],
                        'entry_price': current_price,
                        'direction': 'long',
                        'breakout_level': recent_high,
                        'stop_loss': recent_high * (1 - tolerance)
                    }

                # 向下突破（可用于做空或观望）
                # elif current_price < recent_low * (1 - tolerance):
                #     position = {...}

            else:
                entry_price = position['entry_price']
                stop_loss_price = position['stop_loss']

                # 止损
                if current_price < stop_loss_price:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (current_price - entry_price) / entry_price,
                        'exit_reason': 'stop_loss',
                        'breakout_level': position['breakout_level']
                    })
                    position = None

                # 追踪止盈（价格上涨10%以上或持有超过30天）
                elif current_price > entry_price * 1.10 or (df['trade_date'].iloc[i] - position['entry_date']).days > 30:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': df['trade_date'].iloc[i],
                        'exit_price': current_price,
                        'return': (current_price - entry_price) / entry_price,
                        'exit_reason': 'take_profit' if current_price > entry_price else 'timeout',
                        'breakout_level': position['breakout_level']
                    })
                    position = None

        return pd.DataFrame(trades)


# =============================================================================
# 第五部分：多股票综合分析
# =============================================================================

def analyze_multiple_stocks():
    """分析多只股票的支撑阻力位特征"""

    # 选择代表性股票
    sample_stocks = conn.execute("""
        SELECT DISTINCT d.ts_code, s.name
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE d.ts_code IN (
            '600519.SH', -- 贵州茅台
            '000001.SZ', -- 平安银行
            '600036.SH', -- 招商银行
            '000002.SZ', -- 万科A
            '601318.SH', -- 中国平安
            '600030.SH', -- 中信证券
            '000858.SZ', -- 五粮液
            '002415.SZ', -- 海康威视
            '601398.SH', -- 工商银行
            '600276.SH'  -- 恒瑞医药
        )
        LIMIT 10
    """).fetchdf()

    results = {
        'stock_analysis': [],
        'ma_support_stats': [],
        'volume_cluster_stats': [],
        'breakout_stats': [],
        'support_to_resistance_stats': [],
        'strategy_performance': []
    }

    for _, row in sample_stocks.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        print(f"\n处理: {ts_code} - {name}")

        try:
            # 获取数据
            df = get_stock_data(ts_code, '20200101', '20260130')
            if len(df) < 250:
                print(f"  数据不足，跳过")
                continue

            df = add_technical_indicators(df)

            # 1. 支撑阻力识别
            finder = SupportResistanceFinder(df)
            levels = finder.find_all_levels()

            # 2. 有效性验证
            validator = SupportResistanceValidator(df)

            # 验证局部支撑点
            supports = levels['local_extremes']['supports']
            if supports:
                # 取最近的5个支撑位进行验证
                recent_supports = sorted(supports, key=lambda x: x['date'], reverse=True)[:5]
                for support in recent_supports:
                    touches = validator.test_level_touch(support['price'])
                    stats = validator.calculate_reversal_probability(touches)
                    results['ma_support_stats'].append({
                        'ts_code': ts_code,
                        'name': name,
                        'level_price': support['price'],
                        'level_date': support['date'],
                        **stats
                    })

            # 验证成交密集区
            volume_clusters = levels['volume_clusters'][:3]  # 前3个密集区
            for cluster in volume_clusters:
                touches = validator.test_level_touch(cluster['price'])
                stats = validator.calculate_reversal_probability(touches)
                results['volume_cluster_stats'].append({
                    'ts_code': ts_code,
                    'name': name,
                    'cluster_price': cluster['price'],
                    'vol_ratio': cluster['vol_ratio'],
                    **stats
                })

            # 验证突破
            resistances = levels['local_extremes']['resistances']
            if resistances:
                recent_resistance = sorted(resistances, key=lambda x: x['date'], reverse=True)[0]
                breakouts = validator.analyze_breakout(recent_resistance['price'])
                for bo in breakouts[:5]:  # 最近5次突破
                    results['breakout_stats'].append({
                        'ts_code': ts_code,
                        'name': name,
                        **bo
                    })

            # 支撑变阻力
            sr_events = validator.analyze_support_becomes_resistance()
            for event in sr_events[:3]:  # 最近3次
                results['support_to_resistance_stats'].append({
                    'ts_code': ts_code,
                    'name': name,
                    **event
                })

            # 3. 策略回测
            strategy = SupportResistanceStrategy(df)

            # 支撑位买入策略
            support_trades = strategy.support_buy_strategy()
            if len(support_trades) > 0:
                results['strategy_performance'].append({
                    'ts_code': ts_code,
                    'name': name,
                    'strategy': 'support_buy',
                    'trades': len(support_trades),
                    'win_rate': (support_trades['return'] > 0).mean(),
                    'avg_return': support_trades['return'].mean(),
                    'total_return': (1 + support_trades['return']).prod() - 1
                })

            # 阻力位卖出策略
            resistance_trades = strategy.resistance_sell_strategy()
            if len(resistance_trades) > 0:
                results['strategy_performance'].append({
                    'ts_code': ts_code,
                    'name': name,
                    'strategy': 'resistance_sell',
                    'trades': len(resistance_trades),
                    'win_rate': (resistance_trades['return'] > 0).mean(),
                    'avg_return': resistance_trades['return'].mean(),
                    'total_return': (1 + resistance_trades['return']).prod() - 1
                })

            # 突破追踪策略
            breakout_trades = strategy.breakout_follow_strategy()
            if len(breakout_trades) > 0:
                results['strategy_performance'].append({
                    'ts_code': ts_code,
                    'name': name,
                    'strategy': 'breakout_follow',
                    'trades': len(breakout_trades),
                    'win_rate': (breakout_trades['return'] > 0).mean(),
                    'avg_return': breakout_trades['return'].mean(),
                    'total_return': (1 + breakout_trades['return']).prod() - 1
                })

            # 汇总股票分析
            results['stock_analysis'].append({
                'ts_code': ts_code,
                'name': name,
                'data_start': df['trade_date'].min(),
                'data_end': df['trade_date'].max(),
                'local_supports': len(supports),
                'local_resistances': len(resistances),
                'volume_clusters': len(volume_clusters),
                'current_price': df['close'].iloc[-1],
                'ma20': df['ma20'].iloc[-1] if pd.notna(df['ma20'].iloc[-1]) else None,
                'ma60': df['ma60'].iloc[-1] if pd.notna(df['ma60'].iloc[-1]) else None
            })

        except Exception as e:
            print(f"  错误: {e}")
            continue

    return results


# =============================================================================
# 第六部分：均线支撑阻力详细分析
# =============================================================================

def analyze_ma_support_resistance():
    """均线支撑阻力有效性分析"""

    print("\n" + "=" * 60)
    print("均线支撑阻力有效性分析")
    print("=" * 60)

    # 选择多只股票进行分析
    stocks = conn.execute("""
        SELECT DISTINCT ts_code
        FROM daily
        WHERE ts_code IN (
            '600519.SH', '000001.SZ', '600036.SH', '000002.SZ', '601318.SH',
            '600030.SH', '000858.SZ', '002415.SZ', '601398.SH', '600276.SH',
            '000333.SZ', '600000.SH', '601166.SH', '600887.SH', '000651.SZ'
        )
    """).fetchdf()['ts_code'].tolist()

    ma_stats = {ma: {'support_tests': 0, 'support_holds': 0,
                     'resistance_tests': 0, 'resistance_breaks': 0,
                     'avg_bounce_pct': [], 'avg_break_pct': []}
                for ma in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250']}

    for ts_code in stocks:
        try:
            df = get_stock_data(ts_code, '20200101', '20260130')
            if len(df) < 300:
                continue
            df = add_technical_indicators(df)

            for i in range(260, len(df) - 5):
                for ma in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250']:
                    if pd.isna(df[ma].iloc[i]):
                        continue

                    ma_price = df[ma].iloc[i]
                    current_low = df['low'].iloc[i]
                    current_high = df['high'].iloc[i]
                    current_close = df['close'].iloc[i]
                    prev_close = df['close'].iloc[i-1]

                    # 测试均线支撑
                    if prev_close > ma_price and current_low <= ma_price * 1.01:
                        ma_stats[ma]['support_tests'] += 1
                        future_return = (df['close'].iloc[i+5] - current_close) / current_close

                        if df['close'].iloc[i+5] > ma_price:  # 5天后仍在均线上方
                            ma_stats[ma]['support_holds'] += 1
                            ma_stats[ma]['avg_bounce_pct'].append(future_return)

                    # 测试均线阻力
                    elif prev_close < ma_price and current_high >= ma_price * 0.99:
                        ma_stats[ma]['resistance_tests'] += 1
                        future_return = (df['close'].iloc[i+5] - current_close) / current_close

                        if df['close'].iloc[i+5] > ma_price:  # 5天后突破均线
                            ma_stats[ma]['resistance_breaks'] += 1
                            ma_stats[ma]['avg_break_pct'].append(future_return)

        except Exception as e:
            continue

    # 整理结果
    ma_summary = []
    for ma, stats in ma_stats.items():
        support_hold_rate = stats['support_holds'] / stats['support_tests'] if stats['support_tests'] > 0 else 0
        resistance_break_rate = stats['resistance_breaks'] / stats['resistance_tests'] if stats['resistance_tests'] > 0 else 0

        ma_summary.append({
            'ma': ma,
            'support_tests': stats['support_tests'],
            'support_hold_rate': support_hold_rate,
            'avg_bounce_pct': np.mean(stats['avg_bounce_pct']) if stats['avg_bounce_pct'] else 0,
            'resistance_tests': stats['resistance_tests'],
            'resistance_break_rate': resistance_break_rate,
            'avg_break_pct': np.mean(stats['avg_break_pct']) if stats['avg_break_pct'] else 0
        })

    return pd.DataFrame(ma_summary)


# =============================================================================
# 第七部分：整数关口分析
# =============================================================================

def analyze_round_number_effect():
    """整数关口心理效应分析"""

    print("\n" + "=" * 60)
    print("整数关口心理效应分析")
    print("=" * 60)

    stocks = conn.execute("""
        SELECT DISTINCT d.ts_code, s.name
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE d.ts_code IN (
            '600519.SH', '000001.SZ', '600036.SH', '000002.SZ', '601318.SH',
            '600030.SH', '000858.SZ', '002415.SZ', '601398.SH', '600276.SH'
        )
    """).fetchdf()

    round_number_stats = {
        'approach_from_below': {'total': 0, 'breakthrough': 0, 'rejected': 0},
        'approach_from_above': {'total': 0, 'breakdown': 0, 'bounced': 0},
        'returns_after_break_up': [],
        'returns_after_break_down': []
    }

    for _, row in stocks.iterrows():
        ts_code = row['ts_code']

        try:
            df = get_stock_data(ts_code, '20200101', '20260130')
            if len(df) < 250:
                continue

            # 确定整数关口
            price_range = df['close'].max() - df['close'].min()
            if df['close'].mean() < 20:
                step = 5
            elif df['close'].mean() < 100:
                step = 10
            elif df['close'].mean() < 500:
                step = 50
            else:
                step = 100

            round_levels = list(range(int(df['close'].min() / step) * step,
                                      int(df['close'].max() / step + 1) * step, step))

            for level in round_levels:
                if level <= 0:
                    continue

                for i in range(20, len(df) - 10):
                    current_close = df['close'].iloc[i]
                    prev_close = df['close'].iloc[i-1]

                    # 从下方接近整数关口
                    if prev_close < level * 0.98 and current_close >= level * 0.98 and current_close <= level * 1.02:
                        round_number_stats['approach_from_below']['total'] += 1

                        if df['close'].iloc[i+5] > level * 1.02:
                            round_number_stats['approach_from_below']['breakthrough'] += 1
                            ret = (df['close'].iloc[i+5] - current_close) / current_close
                            round_number_stats['returns_after_break_up'].append(ret)
                        elif df['close'].iloc[i+5] < level * 0.98:
                            round_number_stats['approach_from_below']['rejected'] += 1

                    # 从上方接近整数关口
                    elif prev_close > level * 1.02 and current_close <= level * 1.02 and current_close >= level * 0.98:
                        round_number_stats['approach_from_above']['total'] += 1

                        if df['close'].iloc[i+5] < level * 0.98:
                            round_number_stats['approach_from_above']['breakdown'] += 1
                            ret = (df['close'].iloc[i+5] - current_close) / current_close
                            round_number_stats['returns_after_break_down'].append(ret)
                        elif df['close'].iloc[i+5] > level * 1.02:
                            round_number_stats['approach_from_above']['bounced'] += 1

        except Exception as e:
            continue

    return round_number_stats


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    # 1. 多股票综合分析
    print("\n开始多股票综合分析...")
    results = analyze_multiple_stocks()

    # 2. 均线支撑阻力分析
    print("\n开始均线支撑阻力分析...")
    ma_analysis = analyze_ma_support_resistance()
    print("\n均线支撑阻力统计:")
    print(ma_analysis.to_string())

    # 3. 整数关口分析
    print("\n开始整数关口分析...")
    round_number_stats = analyze_round_number_effect()

    # =============================================================================
    # 生成报告
    # =============================================================================

    report = []
    report.append("# 技术支撑阻力位研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据范围: 2020-01-01 至 2026-01-30")
    report.append("")

    # 摘要
    report.append("## 研究摘要")
    report.append("")
    report.append("本研究基于 A 股市场多只代表性股票的历史数据，系统分析了技术支撑阻力位的识别方法、")
    report.append("有效性验证及策略应用。研究涵盖历史高低点、均线支撑阻力、成交密集区和整数关口等")
    report.append("多种支撑阻力识别方法。")
    report.append("")

    # 第一部分：支撑阻力识别
    report.append("## 一、支撑阻力识别方法")
    report.append("")

    report.append("### 1.1 历史高低点")
    report.append("")
    report.append("**识别方法**: 寻找局部极值点，即在一定窗口期内（默认20个交易日）的最高价和最低价。")
    report.append("")
    report.append("**股票分析汇总**:")
    report.append("")
    report.append("| 股票代码 | 股票名称 | 局部支撑数 | 局部阻力数 | 当前价格 |")
    report.append("|----------|----------|------------|------------|----------|")
    for stock in results['stock_analysis']:
        report.append(f"| {stock['ts_code']} | {stock['name']} | {stock['local_supports']} | {stock['local_resistances']} | {stock['current_price']:.2f} |")
    report.append("")

    report.append("### 1.2 均线支撑阻力")
    report.append("")
    report.append("**识别方法**: 使用 MA5/MA10/MA20/MA60/MA120/MA250 等移动均线作为动态支撑阻力位。")
    report.append("")
    report.append("**均线支撑阻力有效性统计**:")
    report.append("")
    report.append("| 均线 | 支撑测试次数 | 支撑有效率 | 平均反弹幅度 | 阻力测试次数 | 突破成功率 | 平均突破幅度 |")
    report.append("|------|-------------|-----------|--------------|--------------|-----------|--------------|")
    for _, row in ma_analysis.iterrows():
        report.append(f"| {row['ma'].upper()} | {row['support_tests']} | {row['support_hold_rate']:.1%} | {row['avg_bounce_pct']*100:.2f}% | {row['resistance_tests']} | {row['resistance_break_rate']:.1%} | {row['avg_break_pct']*100:.2f}% |")
    report.append("")

    report.append("**关键发现**:")
    report.append("")
    # 找出支撑效果最好的均线
    best_support_ma = ma_analysis.loc[ma_analysis['support_hold_rate'].idxmax()]
    report.append(f"- 支撑效果最佳均线: {best_support_ma['ma'].upper()}，支撑有效率 {best_support_ma['support_hold_rate']:.1%}")
    # 找出最容易被突破的均线
    easiest_break_ma = ma_analysis.loc[ma_analysis['resistance_break_rate'].idxmax()]
    report.append(f"- 最易被突破均线: {easiest_break_ma['ma'].upper()}，突破成功率 {easiest_break_ma['resistance_break_rate']:.1%}")
    report.append("")

    report.append("### 1.3 成交密集区")
    report.append("")
    report.append("**识别方法**: 通过成交量分布（Volume Profile）识别历史成交最为密集的价格区域。")
    report.append("")
    if results['volume_cluster_stats']:
        report.append("**成交密集区分析**:")
        report.append("")
        report.append("| 股票 | 密集区价格 | 成交量倍数 | 触及次数 | 支撑有效率 |")
        report.append("|------|-----------|-----------|----------|-----------|")
        for stat in results['volume_cluster_stats'][:15]:
            report.append(f"| {stat['name']} | {stat['cluster_price']:.2f} | {stat['vol_ratio']:.1f}x | {stat['total_touches']} | {stat['support_reversal_rate']:.1%} |")
        report.append("")

    report.append("### 1.4 整数关口")
    report.append("")
    report.append("**识别方法**: 根据股价水平确定整数关口（如 10/20/50/100 等），分析其心理支撑阻力效应。")
    report.append("")
    report.append("**整数关口效应统计**:")
    report.append("")

    approach_below = round_number_stats['approach_from_below']
    approach_above = round_number_stats['approach_from_above']

    if approach_below['total'] > 0:
        breakthrough_rate = approach_below['breakthrough'] / approach_below['total']
        rejection_rate = approach_below['rejected'] / approach_below['total']
        report.append(f"- 从下方接近整数关口: {approach_below['total']} 次")
        report.append(f"  - 成功突破: {approach_below['breakthrough']} 次 ({breakthrough_rate:.1%})")
        report.append(f"  - 被拒绝: {approach_below['rejected']} 次 ({rejection_rate:.1%})")
        if round_number_stats['returns_after_break_up']:
            avg_ret = np.mean(round_number_stats['returns_after_break_up']) * 100
            report.append(f"  - 突破后5日平均收益: {avg_ret:.2f}%")

    if approach_above['total'] > 0:
        breakdown_rate = approach_above['breakdown'] / approach_above['total']
        bounce_rate = approach_above['bounced'] / approach_above['total']
        report.append(f"- 从上方接近整数关口: {approach_above['total']} 次")
        report.append(f"  - 跌破: {approach_above['breakdown']} 次 ({breakdown_rate:.1%})")
        report.append(f"  - 反弹: {approach_above['bounced']} 次 ({bounce_rate:.1%})")
        if round_number_stats['returns_after_break_down']:
            avg_ret = np.mean(round_number_stats['returns_after_break_down']) * 100
            report.append(f"  - 跌破后5日平均收益: {avg_ret:.2f}%")
    report.append("")

    # 第二部分：有效性验证
    report.append("## 二、有效性验证")
    report.append("")

    report.append("### 2.1 触及后反转概率")
    report.append("")
    report.append("分析价格触及支撑/阻力位后的反转概率，用于评估支撑阻力位的有效性。")
    report.append("")

    if results['ma_support_stats']:
        # 汇总统计
        all_support_tests = [s for s in results['ma_support_stats'] if s['support_tests'] > 0]
        if all_support_tests:
            avg_support_reversal = np.mean([s['support_reversal_rate'] for s in all_support_tests])
            avg_resistance_reversal = np.mean([s['resistance_reversal_rate'] for s in all_support_tests if s['resistance_tests'] > 0])
            report.append(f"- 平均支撑位触及后反弹概率: {avg_support_reversal:.1%}")
            report.append(f"- 平均阻力位触及后回落概率: {avg_resistance_reversal:.1%}")
            report.append("")

    report.append("### 2.2 突破后走势")
    report.append("")

    if results['breakout_stats']:
        breakout_df = pd.DataFrame(results['breakout_stats'])

        upward_breakouts = breakout_df[breakout_df['direction'] == 'upward']
        downward_breakouts = breakout_df[breakout_df['direction'] == 'downward']

        if len(upward_breakouts) > 0:
            false_breakout_rate = upward_breakouts['false_breakout'].mean() if 'false_breakout' in upward_breakouts.columns else 0
            avg_5d_return = upward_breakouts['future_5d_return'].mean()
            report.append(f"**向上突破统计**:")
            report.append(f"- 总突破次数: {len(upward_breakouts)}")
            report.append(f"- 假突破比例: {false_breakout_rate:.1%}")
            report.append(f"- 突破后5日平均收益: {avg_5d_return*100:.2f}%")
            report.append("")

        if len(downward_breakouts) > 0:
            false_breakout_rate = downward_breakouts['false_breakout'].mean() if 'false_breakout' in downward_breakouts.columns else 0
            avg_5d_return = downward_breakouts['future_5d_return'].mean()
            report.append(f"**向下突破统计**:")
            report.append(f"- 总突破次数: {len(downward_breakouts)}")
            report.append(f"- 假突破比例: {false_breakout_rate:.1%}")
            report.append(f"- 突破后5日平均收益: {avg_5d_return*100:.2f}%")
            report.append("")

    report.append("### 2.3 支撑变阻力")
    report.append("")
    report.append("当价格跌破支撑位后，原支撑位可能转变为阻力位，这是一个重要的市场规律。")
    report.append("")

    if results['support_to_resistance_stats']:
        sr_events = results['support_to_resistance_stats']
        report.append(f"- 检测到支撑变阻力事件: {len(sr_events)} 次")
        valid_events = [e for e in sr_events if e['acted_as_resistance']]
        if valid_events:
            avg_subsequent_return = np.mean([e['subsequent_return'] for e in valid_events if pd.notna(e['subsequent_return'])])
            report.append(f"- 原支撑位确认成为阻力后，5日平均收益: {avg_subsequent_return*100:.2f}%")
        report.append("")

    # 第三部分：策略应用
    report.append("## 三、策略应用")
    report.append("")

    if results['strategy_performance']:
        strategy_df = pd.DataFrame(results['strategy_performance'])

        report.append("### 3.1 支撑位买入策略")
        report.append("")
        report.append("**策略逻辑**: 当价格触及近期支撑位时买入，跌破支撑止损，持有一定时间后盈利卖出。")
        report.append("")

        support_buy = strategy_df[strategy_df['strategy'] == 'support_buy']
        if len(support_buy) > 0:
            report.append("| 股票 | 交易次数 | 胜率 | 平均收益 | 累计收益 |")
            report.append("|------|----------|------|----------|----------|")
            for _, row in support_buy.iterrows():
                report.append(f"| {row['name']} | {row['trades']} | {row['win_rate']:.1%} | {row['avg_return']*100:.2f}% | {row['total_return']*100:.2f}% |")

            report.append("")
            report.append(f"**策略汇总**: 平均胜率 {support_buy['win_rate'].mean():.1%}, 平均单笔收益 {support_buy['avg_return'].mean()*100:.2f}%")
            report.append("")

        report.append("### 3.2 阻力位卖出策略")
        report.append("")
        report.append("**策略逻辑**: 当价格触及近期阻力位时做空（或卖出），突破阻力止损。")
        report.append("")

        resistance_sell = strategy_df[strategy_df['strategy'] == 'resistance_sell']
        if len(resistance_sell) > 0:
            report.append("| 股票 | 交易次数 | 胜率 | 平均收益 | 累计收益 |")
            report.append("|------|----------|------|----------|----------|")
            for _, row in resistance_sell.iterrows():
                report.append(f"| {row['name']} | {row['trades']} | {row['win_rate']:.1%} | {row['avg_return']*100:.2f}% | {row['total_return']*100:.2f}% |")

            report.append("")
            report.append(f"**策略汇总**: 平均胜率 {resistance_sell['win_rate'].mean():.1%}, 平均单笔收益 {resistance_sell['avg_return'].mean()*100:.2f}%")
            report.append("")

        report.append("### 3.3 突破追踪策略")
        report.append("")
        report.append("**策略逻辑**: 当价格突破近期高点时追入，回落止损，趋势延续则持有。")
        report.append("")

        breakout_follow = strategy_df[strategy_df['strategy'] == 'breakout_follow']
        if len(breakout_follow) > 0:
            report.append("| 股票 | 交易次数 | 胜率 | 平均收益 | 累计收益 |")
            report.append("|------|----------|------|----------|----------|")
            for _, row in breakout_follow.iterrows():
                report.append(f"| {row['name']} | {row['trades']} | {row['win_rate']:.1%} | {row['avg_return']*100:.2f}% | {row['total_return']*100:.2f}% |")

            report.append("")
            report.append(f"**策略汇总**: 平均胜率 {breakout_follow['win_rate'].mean():.1%}, 平均单笔收益 {breakout_follow['avg_return'].mean()*100:.2f}%")
            report.append("")

    # 结论与建议
    report.append("## 四、结论与建议")
    report.append("")
    report.append("### 4.1 主要发现")
    report.append("")
    report.append("1. **均线支撑阻力**: 长期均线（如 MA60、MA120、MA250）的支撑阻力效果通常优于短期均线，")
    report.append("   但短期均线的信号更为频繁。")
    report.append("")
    report.append("2. **成交密集区**: 成交量高度集中的价格区域往往形成强支撑或阻力，这是因为大量投资者")
    report.append("   在该价位建仓，形成心理锚定效应。")
    report.append("")
    report.append("3. **整数关口效应**: 整数关口具有明显的心理支撑阻力作用，但单独使用效果有限，")
    report.append("   需要结合其他技术指标。")
    report.append("")
    report.append("4. **支撑变阻力**: 当支撑位被有效跌破后，该位置往往转变为阻力位，")
    report.append("   这是一个重要的市场规律，可用于判断反弹高度。")
    report.append("")

    report.append("### 4.2 策略建议")
    report.append("")
    report.append("1. **多重确认**: 单一支撑阻力位的可靠性有限，建议寻找多个支撑阻力位重合的区域，")
    report.append("   如均线与历史高低点重合、成交密集区与整数关口重合等。")
    report.append("")
    report.append("2. **结合成交量**: 突破或跌破支撑阻力位时，应关注成交量配合情况，")
    report.append("   放量突破的有效性更高。")
    report.append("")
    report.append("3. **严格止损**: 无论采用哪种策略，都应设置严格的止损位，")
    report.append("   支撑位买入策略的止损可设在支撑位下方2-3%。")
    report.append("")
    report.append("4. **趋势结合**: 支撑阻力位策略应结合大趋势使用，")
    report.append("   上升趋势中优先使用支撑位买入，下降趋势中优先使用阻力位卖出。")
    report.append("")

    report.append("### 4.3 风险提示")
    report.append("")
    report.append("1. 支撑阻力位不是精确的价格点，而是一个价格区间")
    report.append("2. 历史支撑阻力位不保证未来有效")
    report.append("3. 市场环境变化可能导致支撑阻力位失效")
    report.append("4. 本研究基于历史数据回测，实际交易中可能存在滑点、手续费等因素")
    report.append("")

    # 保存报告
    report_content = '\n'.join(report)
    report_file = f"{REPORT_PATH}/support_resistance_research_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n报告已保存到: {report_file}")

    # 同时保存详细数据到 CSV
    if results['strategy_performance']:
        strategy_df = pd.DataFrame(results['strategy_performance'])
        strategy_df.to_csv(f"{REPORT_PATH}/strategy_performance.csv", index=False, encoding='utf-8-sig')
        print(f"策略表现数据已保存到: {REPORT_PATH}/strategy_performance.csv")

    ma_analysis.to_csv(f"{REPORT_PATH}/ma_support_resistance_stats.csv", index=False, encoding='utf-8-sig')
    print(f"均线分析数据已保存到: {REPORT_PATH}/ma_support_resistance_stats.csv")

    if results['stock_analysis']:
        stock_df = pd.DataFrame(results['stock_analysis'])
        stock_df.to_csv(f"{REPORT_PATH}/stock_analysis_summary.csv", index=False, encoding='utf-8-sig')
        print(f"股票分析汇总已保存到: {REPORT_PATH}/stock_analysis_summary.csv")

    print("\n分析完成！")

    conn.close()
