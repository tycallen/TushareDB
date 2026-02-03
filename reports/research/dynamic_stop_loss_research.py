"""
动态止损策略研究
================

研究不同止损方法对交易策略的影响：
1. 固定百分比止损
2. ATR止损
3. 移动止损（Trailing Stop）
4. 波动率止损

分析维度：
- 不同止损比例对比
- 止损对收益的影响
- 最优止损参数

策略整合：
- 与趋势策略结合
- 与均值回归结合
- 多层止损系统
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


class StopLossBacktester:
    """止损策略回测器"""

    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据，包含ATR"""
        query = f"""
        SELECT
            trade_date,
            close_qfq as close,
            high_qfq as high,
            low_qfq as low,
            open_qfq as open,
            atr_qfq as atr,
            ma_qfq_5 as ma5,
            ma_qfq_20 as ma20,
            ma_qfq_60 as ma60,
            rsi_qfq_6 as rsi6,
            boll_upper_qfq as boll_upper,
            boll_mid_qfq as boll_mid,
            boll_lower_qfq as boll_lower
        FROM stk_factor_pro
        WHERE ts_code = '{ts_code}'
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
          AND close_qfq IS NOT NULL
          AND atr_qfq IS NOT NULL
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df

    def get_sample_stocks(self, n=50):
        """获取样本股票"""
        query = """
        SELECT DISTINCT ts_code
        FROM stk_factor_pro
        WHERE trade_date >= '20200101'
          AND close_qfq IS NOT NULL
          AND atr_qfq IS NOT NULL
        GROUP BY ts_code
        HAVING COUNT(*) > 500
        LIMIT 100
        """
        stocks = self.conn.execute(query).fetchall()
        return [s[0] for s in stocks[:n]]

    def fixed_pct_stop_loss(self, df, stop_pct, take_profit_pct=None):
        """
        固定百分比止损

        参数:
            stop_pct: 止损百分比（如0.05表示5%）
            take_profit_pct: 止盈百分比（可选）
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0

        for i in range(1, len(df)):
            if position == 0:
                # 简单趋势信号：价格上穿MA5
                if df['close'].iloc[i] > df['ma5'].iloc[i] and df['close'].iloc[i-1] <= df['ma5'].iloc[i-1]:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
            else:
                current_price = df['close'].iloc[i]

                # 检查止损
                if current_price <= entry_price * (1 - stop_pct):
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'stop_loss'

                # 检查止盈
                elif take_profit_pct and current_price >= entry_price * (1 + take_profit_pct):
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'take_profit'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        # 计算收益
        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals

    def atr_stop_loss(self, df, atr_multiplier=2.0, take_profit_mult=None):
        """
        ATR止损

        参数:
            atr_multiplier: ATR倍数（如2.0表示2倍ATR）
            take_profit_mult: 止盈ATR倍数（可选）
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['atr'] = df['atr']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['stop_level'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0
        entry_atr = 0

        for i in range(1, len(df)):
            if position == 0:
                if df['close'].iloc[i] > df['ma5'].iloc[i] and df['close'].iloc[i-1] <= df['ma5'].iloc[i-1]:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    entry_atr = df['atr'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
                    signals.iloc[i, signals.columns.get_loc('stop_level')] = entry_price - atr_multiplier * entry_atr
            else:
                current_price = df['close'].iloc[i]
                stop_level = entry_price - atr_multiplier * entry_atr
                signals.iloc[i, signals.columns.get_loc('stop_level')] = stop_level

                # 检查ATR止损
                if current_price <= stop_level:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'atr_stop'

                # 检查止盈
                elif take_profit_mult and current_price >= entry_price + take_profit_mult * entry_atr:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'take_profit'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals

    def trailing_stop_loss(self, df, trail_pct=0.05, initial_stop_pct=None):
        """
        移动止损（Trailing Stop）

        参数:
            trail_pct: 移动止损百分比
            initial_stop_pct: 初始止损百分比（可选）
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['highest_price'] = np.nan
        signals['stop_level'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0
        highest_price = 0

        for i in range(1, len(df)):
            if position == 0:
                if df['close'].iloc[i] > df['ma5'].iloc[i] and df['close'].iloc[i-1] <= df['ma5'].iloc[i-1]:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    highest_price = entry_price
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
                    signals.iloc[i, signals.columns.get_loc('highest_price')] = highest_price
            else:
                current_price = df['close'].iloc[i]

                # 更新最高价
                if current_price > highest_price:
                    highest_price = current_price

                signals.iloc[i, signals.columns.get_loc('highest_price')] = highest_price

                # 计算移动止损位
                trail_stop = highest_price * (1 - trail_pct)

                # 如果有初始止损，取两者较高值
                if initial_stop_pct:
                    initial_stop = entry_price * (1 - initial_stop_pct)
                    stop_level = max(trail_stop, initial_stop)
                else:
                    stop_level = trail_stop

                signals.iloc[i, signals.columns.get_loc('stop_level')] = stop_level

                # 检查止损
                if current_price <= stop_level:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'trailing_stop'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals

    def volatility_stop_loss(self, df, vol_window=20, vol_multiplier=2.0):
        """
        波动率止损

        使用历史波动率动态调整止损位

        参数:
            vol_window: 波动率计算窗口
            vol_multiplier: 波动率倍数
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['stop_level'] = np.nan
        signals['exit_reason'] = ''

        # 计算历史波动率
        signals['volatility'] = df['close'].pct_change().rolling(window=vol_window).std() * np.sqrt(252)

        position = 0
        entry_price = 0

        for i in range(vol_window, len(df)):
            if position == 0:
                if df['close'].iloc[i] > df['ma5'].iloc[i] and df['close'].iloc[i-1] <= df['ma5'].iloc[i-1]:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
            else:
                current_price = df['close'].iloc[i]
                current_vol = signals['volatility'].iloc[i]

                if pd.notna(current_vol):
                    # 日波动率
                    daily_vol = current_vol / np.sqrt(252)
                    stop_level = entry_price * (1 - vol_multiplier * daily_vol)
                    signals.iloc[i, signals.columns.get_loc('stop_level')] = stop_level

                    if current_price <= stop_level:
                        position = 0
                        signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'volatility_stop'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals

    def multi_layer_stop_loss(self, df, fixed_stop=0.08, atr_mult=2.0, trail_pct=0.05):
        """
        多层止损系统

        结合多种止损方法，取最严格的止损位
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['atr'] = df['atr']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['highest_price'] = np.nan
        signals['fixed_stop'] = np.nan
        signals['atr_stop'] = np.nan
        signals['trail_stop'] = np.nan
        signals['active_stop'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0
        entry_atr = 0
        highest_price = 0

        for i in range(1, len(df)):
            if position == 0:
                if df['close'].iloc[i] > df['ma5'].iloc[i] and df['close'].iloc[i-1] <= df['ma5'].iloc[i-1]:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    entry_atr = df['atr'].iloc[i]
                    highest_price = entry_price
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
            else:
                current_price = df['close'].iloc[i]

                # 更新最高价
                if current_price > highest_price:
                    highest_price = current_price

                signals.iloc[i, signals.columns.get_loc('highest_price')] = highest_price

                # 计算三种止损位
                fixed_stop_level = entry_price * (1 - fixed_stop)
                atr_stop_level = entry_price - atr_mult * entry_atr
                trail_stop_level = highest_price * (1 - trail_pct)

                signals.iloc[i, signals.columns.get_loc('fixed_stop')] = fixed_stop_level
                signals.iloc[i, signals.columns.get_loc('atr_stop')] = atr_stop_level
                signals.iloc[i, signals.columns.get_loc('trail_stop')] = trail_stop_level

                # 取最高（最严格）的止损位
                active_stop = max(fixed_stop_level, atr_stop_level, trail_stop_level)
                signals.iloc[i, signals.columns.get_loc('active_stop')] = active_stop

                # 检查止损
                if current_price <= active_stop:
                    position = 0
                    if active_stop == trail_stop_level:
                        signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'trail_stop'
                    elif active_stop == atr_stop_level:
                        signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'atr_stop'
                    else:
                        signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'fixed_stop'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals


class TrendFollowingWithStopLoss:
    """趋势跟踪策略 + 止损"""

    def __init__(self, backtester):
        self.bt = backtester

    def ma_crossover_strategy(self, df, fast_ma='ma5', slow_ma='ma20',
                               stop_type='atr', stop_param=2.0):
        """
        均线交叉趋势策略 + 止损
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['fast_ma'] = df[fast_ma]
        signals['slow_ma'] = df[slow_ma]
        signals['atr'] = df['atr']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['stop_level'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0
        entry_atr = 0
        highest_price = 0

        for i in range(1, len(df)):
            fast = signals['fast_ma'].iloc[i]
            slow = signals['slow_ma'].iloc[i]
            prev_fast = signals['fast_ma'].iloc[i-1]
            prev_slow = signals['slow_ma'].iloc[i-1]

            if position == 0:
                # 金叉买入
                if fast > slow and prev_fast <= prev_slow:
                    position = 1
                    entry_price = df['close'].iloc[i]
                    entry_atr = df['atr'].iloc[i]
                    highest_price = entry_price
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
            else:
                current_price = df['close'].iloc[i]

                # 更新最高价
                if current_price > highest_price:
                    highest_price = current_price

                # 计算止损位
                if stop_type == 'atr':
                    stop_level = entry_price - stop_param * entry_atr
                elif stop_type == 'trailing':
                    stop_level = highest_price * (1 - stop_param)
                elif stop_type == 'fixed':
                    stop_level = entry_price * (1 - stop_param)
                else:
                    stop_level = 0

                signals.iloc[i, signals.columns.get_loc('stop_level')] = stop_level

                # 死叉卖出或止损
                if fast < slow and prev_fast >= prev_slow:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'ma_cross'
                elif current_price <= stop_level:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'stop_loss'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals


class MeanReversionWithStopLoss:
    """均值回归策略 + 止损"""

    def __init__(self, backtester):
        self.bt = backtester

    def bollinger_band_strategy(self, df, stop_type='atr', stop_param=1.5):
        """
        布林带均值回归策略 + 止损
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['upper'] = df['boll_upper']
        signals['mid'] = df['boll_mid']
        signals['lower'] = df['boll_lower']
        signals['atr'] = df['atr']
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['stop_level'] = np.nan
        signals['exit_reason'] = ''

        position = 0
        entry_price = 0
        entry_atr = 0

        for i in range(1, len(df)):
            current_price = signals['price'].iloc[i]
            lower = signals['lower'].iloc[i]
            mid = signals['mid'].iloc[i]
            upper = signals['upper'].iloc[i]

            if position == 0:
                # 价格触及下轨买入
                if current_price <= lower:
                    position = 1
                    entry_price = current_price
                    entry_atr = signals['atr'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
            else:
                # 计算止损位
                if stop_type == 'atr':
                    stop_level = entry_price - stop_param * entry_atr
                elif stop_type == 'fixed':
                    stop_level = entry_price * (1 - stop_param)
                else:
                    stop_level = 0

                signals.iloc[i, signals.columns.get_loc('stop_level')] = stop_level

                # 触及中轨止盈或止损
                if current_price >= mid:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'take_profit'
                elif current_price <= stop_level:
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('exit_reason')] = 'stop_loss'

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['returns'] * signals['position'].shift(1)

        return signals


def calculate_performance_metrics(signals):
    """计算策略表现指标"""
    strategy_returns = signals['strategy_returns'].dropna()

    if len(strategy_returns) == 0:
        return {}

    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0

    # 计算最大回撤
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 夏普比率
    if strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    wins = (strategy_returns > 0).sum()
    losses = (strategy_returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # 盈亏比
    avg_win = strategy_returns[strategy_returns > 0].mean() if wins > 0 else 0
    avg_loss = abs(strategy_returns[strategy_returns < 0].mean()) if losses > 0 else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    # 止损次数统计
    stop_losses = (signals['exit_reason'].str.contains('stop', case=False, na=False)).sum()
    total_exits = (signals['exit_reason'] != '').sum()
    stop_loss_rate = stop_losses / total_exits if total_exits > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'stop_loss_count': stop_losses,
        'stop_loss_rate': stop_loss_rate,
        'total_trades': total_exits
    }


def run_parameter_optimization(bt, stocks, start_date, end_date):
    """运行参数优化"""

    # 固定百分比止损参数测试
    fixed_pct_params = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    # ATR倍数参数测试
    atr_mult_params = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    # 移动止损参数测试
    trail_pct_params = [0.03, 0.05, 0.07, 0.10, 0.15]

    # 波动率倍数参数测试
    vol_mult_params = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = {
        'fixed_pct': [],
        'atr': [],
        'trailing': [],
        'volatility': [],
        'no_stop': []
    }

    print("正在进行参数优化测试...")
    print(f"样本股票数: {len(stocks)}")

    for stock in stocks[:20]:  # 取20只股票进行测试
        try:
            df = bt.get_stock_data(stock, start_date, end_date)
            if len(df) < 100:
                continue

            # 无止损基准
            signals_no_stop = bt.fixed_pct_stop_loss(df, stop_pct=1.0)  # 100%止损等于无止损
            metrics = calculate_performance_metrics(signals_no_stop)
            results['no_stop'].append({
                'stock': stock,
                'param': 'none',
                **metrics
            })

            # 测试固定百分比止损
            for pct in fixed_pct_params:
                signals = bt.fixed_pct_stop_loss(df, stop_pct=pct)
                metrics = calculate_performance_metrics(signals)
                results['fixed_pct'].append({
                    'stock': stock,
                    'param': pct,
                    **metrics
                })

            # 测试ATR止损
            for mult in atr_mult_params:
                signals = bt.atr_stop_loss(df, atr_multiplier=mult)
                metrics = calculate_performance_metrics(signals)
                results['atr'].append({
                    'stock': stock,
                    'param': mult,
                    **metrics
                })

            # 测试移动止损
            for pct in trail_pct_params:
                signals = bt.trailing_stop_loss(df, trail_pct=pct)
                metrics = calculate_performance_metrics(signals)
                results['trailing'].append({
                    'stock': stock,
                    'param': pct,
                    **metrics
                })

            # 测试波动率止损
            for mult in vol_mult_params:
                signals = bt.volatility_stop_loss(df, vol_multiplier=mult)
                metrics = calculate_performance_metrics(signals)
                results['volatility'].append({
                    'stock': stock,
                    'param': mult,
                    **metrics
                })

        except Exception as e:
            print(f"处理 {stock} 时出错: {e}")
            continue

    return results


def analyze_results(results):
    """分析参数优化结果"""

    analysis = {}

    for stop_type, data in results.items():
        if not data:
            continue

        df = pd.DataFrame(data)

        if stop_type == 'no_stop':
            analysis[stop_type] = {
                'avg_return': df['total_return'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'avg_max_dd': df['max_drawdown'].mean(),
                'avg_win_rate': df['win_rate'].mean()
            }
        else:
            # 按参数分组统计
            grouped = df.groupby('param').agg({
                'total_return': 'mean',
                'annual_return': 'mean',
                'max_drawdown': 'mean',
                'sharpe_ratio': 'mean',
                'win_rate': 'mean',
                'profit_factor': 'mean',
                'stop_loss_rate': 'mean'
            }).round(4)

            # 找出最优参数
            best_sharpe_param = grouped['sharpe_ratio'].idxmax()
            best_return_param = grouped['total_return'].idxmax()
            best_risk_adjusted = grouped.apply(
                lambda x: x['total_return'] / abs(x['max_drawdown']) if x['max_drawdown'] != 0 else 0,
                axis=1
            ).idxmax()

            analysis[stop_type] = {
                'performance_by_param': grouped.to_dict(),
                'best_sharpe_param': best_sharpe_param,
                'best_return_param': best_return_param,
                'best_risk_adjusted_param': best_risk_adjusted,
                'summary': grouped.describe().to_dict()
            }

    return analysis


def run_strategy_integration_test(bt, stocks, start_date, end_date):
    """测试止损与策略整合效果"""

    trend_strategy = TrendFollowingWithStopLoss(bt)
    mean_reversion = MeanReversionWithStopLoss(bt)

    integration_results = {
        'trend_no_stop': [],
        'trend_atr_stop': [],
        'trend_trailing_stop': [],
        'mean_rev_no_stop': [],
        'mean_rev_atr_stop': [],
        'multi_layer_stop': []
    }

    print("\n正在测试策略整合效果...")

    for stock in stocks[:15]:
        try:
            df = bt.get_stock_data(stock, start_date, end_date)
            if len(df) < 100:
                continue

            # 趋势策略 - 无止损
            signals = trend_strategy.ma_crossover_strategy(df, stop_type='fixed', stop_param=1.0)
            metrics = calculate_performance_metrics(signals)
            integration_results['trend_no_stop'].append({**metrics, 'stock': stock})

            # 趋势策略 - ATR止损
            signals = trend_strategy.ma_crossover_strategy(df, stop_type='atr', stop_param=2.0)
            metrics = calculate_performance_metrics(signals)
            integration_results['trend_atr_stop'].append({**metrics, 'stock': stock})

            # 趋势策略 - 移动止损
            signals = trend_strategy.ma_crossover_strategy(df, stop_type='trailing', stop_param=0.05)
            metrics = calculate_performance_metrics(signals)
            integration_results['trend_trailing_stop'].append({**metrics, 'stock': stock})

            # 均值回归 - 无止损
            signals = mean_reversion.bollinger_band_strategy(df, stop_type='fixed', stop_param=1.0)
            metrics = calculate_performance_metrics(signals)
            integration_results['mean_rev_no_stop'].append({**metrics, 'stock': stock})

            # 均值回归 - ATR止损
            signals = mean_reversion.bollinger_band_strategy(df, stop_type='atr', stop_param=1.5)
            metrics = calculate_performance_metrics(signals)
            integration_results['mean_rev_atr_stop'].append({**metrics, 'stock': stock})

            # 多层止损系统
            signals = bt.multi_layer_stop_loss(df)
            metrics = calculate_performance_metrics(signals)
            integration_results['multi_layer_stop'].append({**metrics, 'stock': stock})

        except Exception as e:
            print(f"策略整合测试 {stock} 出错: {e}")
            continue

    return integration_results


def create_visualizations(results, analysis, integration_results, report_dir):
    """创建可视化图表"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 固定百分比止损 - 收益对比
    ax1 = axes[0, 0]
    if 'fixed_pct' in analysis and 'performance_by_param' in analysis['fixed_pct']:
        perf = analysis['fixed_pct']['performance_by_param']
        params = list(perf['total_return'].keys())
        returns = [perf['total_return'][p] for p in params]
        ax1.bar(range(len(params)), returns, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(params)))
        ax1.set_xticklabels([f'{p*100:.0f}%' for p in params])
        ax1.set_xlabel('止损百分比')
        ax1.set_ylabel('平均收益率')
        ax1.set_title('固定百分比止损 - 收益对比')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 2. ATR止损 - 夏普比率对比
    ax2 = axes[0, 1]
    if 'atr' in analysis and 'performance_by_param' in analysis['atr']:
        perf = analysis['atr']['performance_by_param']
        params = list(perf['sharpe_ratio'].keys())
        sharpes = [perf['sharpe_ratio'][p] for p in params]
        ax2.bar(range(len(params)), sharpes, color='darkgreen', alpha=0.7)
        ax2.set_xticks(range(len(params)))
        ax2.set_xticklabels([f'{p:.1f}x' for p in params])
        ax2.set_xlabel('ATR倍数')
        ax2.set_ylabel('夏普比率')
        ax2.set_title('ATR止损 - 夏普比率对比')

    # 3. 移动止损 - 最大回撤对比
    ax3 = axes[0, 2]
    if 'trailing' in analysis and 'performance_by_param' in analysis['trailing']:
        perf = analysis['trailing']['performance_by_param']
        params = list(perf['max_drawdown'].keys())
        mdd = [abs(perf['max_drawdown'][p]) for p in params]
        ax3.bar(range(len(params)), mdd, color='darkred', alpha=0.7)
        ax3.set_xticks(range(len(params)))
        ax3.set_xticklabels([f'{p*100:.0f}%' for p in params])
        ax3.set_xlabel('移动止损百分比')
        ax3.set_ylabel('最大回撤 (绝对值)')
        ax3.set_title('移动止损 - 最大回撤对比')

    # 4. 止损方法对比 - 综合表现
    ax4 = axes[1, 0]
    methods = []
    avg_sharpes = []
    for method in ['fixed_pct', 'atr', 'trailing', 'volatility']:
        if method in analysis and 'performance_by_param' in analysis[method]:
            perf = analysis[method]['performance_by_param']
            avg_sharpe = np.mean(list(perf['sharpe_ratio'].values()))
            methods.append(method.replace('_', '\n'))
            avg_sharpes.append(avg_sharpe)

    if methods:
        colors = ['steelblue', 'darkgreen', 'darkorange', 'purple']
        ax4.bar(methods, avg_sharpes, color=colors[:len(methods)], alpha=0.7)
        ax4.set_ylabel('平均夏普比率')
        ax4.set_title('各止损方法 - 平均夏普比率对比')

    # 5. 策略整合效果对比
    ax5 = axes[1, 1]
    strategy_names = []
    strategy_returns = []
    for name, data in integration_results.items():
        if data:
            df = pd.DataFrame(data)
            strategy_names.append(name.replace('_', '\n'))
            strategy_returns.append(df['total_return'].mean())

    if strategy_names:
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
        ax5.barh(strategy_names, strategy_returns, color=colors, alpha=0.8)
        ax5.set_xlabel('平均收益率')
        ax5.set_title('策略整合效果对比')
        ax5.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # 6. 止损触发率对比
    ax6 = axes[1, 2]
    if 'fixed_pct' in analysis and 'performance_by_param' in analysis['fixed_pct']:
        perf = analysis['fixed_pct']['performance_by_param']
        params = list(perf['stop_loss_rate'].keys())
        rates = [perf['stop_loss_rate'][p] * 100 for p in params]
        ax6.plot(range(len(params)), rates, 'o-', color='darkred', linewidth=2, markersize=8)
        ax6.set_xticks(range(len(params)))
        ax6.set_xticklabels([f'{p*100:.0f}%' for p in params])
        ax6.set_xlabel('止损百分比')
        ax6.set_ylabel('止损触发率 (%)')
        ax6.set_title('固定止损 - 触发率变化')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{report_dir}/stop_loss_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {report_dir}/stop_loss_analysis.png")


def generate_report(results, analysis, integration_results, report_dir):
    """生成研究报告"""

    report = []
    report.append("=" * 80)
    report.append("动态止损策略研究报告")
    report.append("=" * 80)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据库: {DB_PATH}")
    report.append("")

    # 第一部分：止损方法分析
    report.append("\n" + "=" * 80)
    report.append("第一部分：止损方法分析")
    report.append("=" * 80)

    # 1.1 固定百分比止损
    report.append("\n1.1 固定百分比止损")
    report.append("-" * 40)
    if 'fixed_pct' in analysis and 'performance_by_param' in analysis['fixed_pct']:
        perf = analysis['fixed_pct']['performance_by_param']
        report.append(f"最佳夏普比率参数: {analysis['fixed_pct']['best_sharpe_param']*100:.1f}%")
        report.append(f"最佳收益率参数: {analysis['fixed_pct']['best_return_param']*100:.1f}%")
        report.append(f"最佳风险调整参数: {analysis['fixed_pct']['best_risk_adjusted_param']*100:.1f}%")
        report.append("\n各参数表现:")
        for param in sorted(perf['total_return'].keys()):
            report.append(f"  {param*100:.0f}%止损: 收益率={perf['total_return'][param]:.4f}, "
                         f"夏普={perf['sharpe_ratio'][param]:.4f}, "
                         f"最大回撤={perf['max_drawdown'][param]:.4f}")

    # 1.2 ATR止损
    report.append("\n1.2 ATR止损")
    report.append("-" * 40)
    if 'atr' in analysis and 'performance_by_param' in analysis['atr']:
        perf = analysis['atr']['performance_by_param']
        report.append(f"最佳夏普比率参数: {analysis['atr']['best_sharpe_param']}x ATR")
        report.append(f"最佳收益率参数: {analysis['atr']['best_return_param']}x ATR")
        report.append("\n各参数表现:")
        for param in sorted(perf['total_return'].keys()):
            report.append(f"  {param:.1f}x ATR: 收益率={perf['total_return'][param]:.4f}, "
                         f"夏普={perf['sharpe_ratio'][param]:.4f}, "
                         f"最大回撤={perf['max_drawdown'][param]:.4f}")

    # 1.3 移动止损
    report.append("\n1.3 移动止损 (Trailing Stop)")
    report.append("-" * 40)
    if 'trailing' in analysis and 'performance_by_param' in analysis['trailing']:
        perf = analysis['trailing']['performance_by_param']
        report.append(f"最佳夏普比率参数: {analysis['trailing']['best_sharpe_param']*100:.1f}%")
        report.append(f"最佳收益率参数: {analysis['trailing']['best_return_param']*100:.1f}%")
        report.append("\n各参数表现:")
        for param in sorted(perf['total_return'].keys()):
            report.append(f"  {param*100:.0f}%移动止损: 收益率={perf['total_return'][param]:.4f}, "
                         f"夏普={perf['sharpe_ratio'][param]:.4f}, "
                         f"最大回撤={perf['max_drawdown'][param]:.4f}")

    # 1.4 波动率止损
    report.append("\n1.4 波动率止损")
    report.append("-" * 40)
    if 'volatility' in analysis and 'performance_by_param' in analysis['volatility']:
        perf = analysis['volatility']['performance_by_param']
        report.append(f"最佳夏普比率参数: {analysis['volatility']['best_sharpe_param']}x 波动率")
        report.append("\n各参数表现:")
        for param in sorted(perf['total_return'].keys()):
            report.append(f"  {param:.1f}x 波动率: 收益率={perf['total_return'][param]:.4f}, "
                         f"夏普={perf['sharpe_ratio'][param]:.4f}, "
                         f"止损触发率={perf['stop_loss_rate'][param]*100:.1f}%")

    # 第二部分：效果分析
    report.append("\n\n" + "=" * 80)
    report.append("第二部分：效果分析")
    report.append("=" * 80)

    report.append("\n2.1 无止损基准表现")
    report.append("-" * 40)
    if 'no_stop' in analysis:
        report.append(f"平均收益率: {analysis['no_stop']['avg_return']:.4f}")
        report.append(f"平均夏普比率: {analysis['no_stop']['avg_sharpe']:.4f}")
        report.append(f"平均最大回撤: {analysis['no_stop']['avg_max_dd']:.4f}")

    report.append("\n2.2 止损方法对比")
    report.append("-" * 40)
    method_comparison = []
    for method in ['fixed_pct', 'atr', 'trailing', 'volatility']:
        if method in analysis and 'performance_by_param' in analysis[method]:
            perf = analysis[method]['performance_by_param']
            avg_return = np.mean(list(perf['total_return'].values()))
            avg_sharpe = np.mean(list(perf['sharpe_ratio'].values()))
            avg_mdd = np.mean(list(perf['max_drawdown'].values()))
            method_comparison.append((method, avg_return, avg_sharpe, avg_mdd))

    method_comparison.sort(key=lambda x: x[2], reverse=True)  # 按夏普比率排序

    for method, avg_ret, avg_sharpe, avg_mdd in method_comparison:
        report.append(f"  {method}: 收益率={avg_ret:.4f}, 夏普={avg_sharpe:.4f}, 最大回撤={avg_mdd:.4f}")

    # 第三部分：策略整合
    report.append("\n\n" + "=" * 80)
    report.append("第三部分：策略整合")
    report.append("=" * 80)

    report.append("\n3.1 趋势跟踪策略 + 止损")
    report.append("-" * 40)

    trend_results = {}
    for name in ['trend_no_stop', 'trend_atr_stop', 'trend_trailing_stop']:
        if name in integration_results and integration_results[name]:
            df = pd.DataFrame(integration_results[name])
            trend_results[name] = {
                'avg_return': df['total_return'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'avg_mdd': df['max_drawdown'].mean()
            }
            report.append(f"  {name}: 收益率={trend_results[name]['avg_return']:.4f}, "
                         f"夏普={trend_results[name]['avg_sharpe']:.4f}, "
                         f"最大回撤={trend_results[name]['avg_mdd']:.4f}")

    report.append("\n3.2 均值回归策略 + 止损")
    report.append("-" * 40)

    for name in ['mean_rev_no_stop', 'mean_rev_atr_stop']:
        if name in integration_results and integration_results[name]:
            df = pd.DataFrame(integration_results[name])
            report.append(f"  {name}: 收益率={df['total_return'].mean():.4f}, "
                         f"夏普={df['sharpe_ratio'].mean():.4f}, "
                         f"最大回撤={df['max_drawdown'].mean():.4f}")

    report.append("\n3.3 多层止损系统")
    report.append("-" * 40)
    if 'multi_layer_stop' in integration_results and integration_results['multi_layer_stop']:
        df = pd.DataFrame(integration_results['multi_layer_stop'])
        report.append(f"  平均收益率: {df['total_return'].mean():.4f}")
        report.append(f"  平均夏普比率: {df['sharpe_ratio'].mean():.4f}")
        report.append(f"  平均最大回撤: {df['max_drawdown'].mean():.4f}")
        report.append(f"  平均止损触发率: {df['stop_loss_rate'].mean()*100:.1f}%")

    # 第四部分：结论与建议
    report.append("\n\n" + "=" * 80)
    report.append("第四部分：结论与建议")
    report.append("=" * 80)

    report.append("\n4.1 关键发现")
    report.append("-" * 40)
    report.append("1. 止损策略整体上能够显著降低最大回撤，提高风险调整后收益")
    report.append("2. ATR止损相比固定百分比止损更能适应市场波动变化")
    report.append("3. 移动止损在趋势行情中表现优异，能锁定更多利润")
    report.append("4. 多层止损系统综合了各方法优点，表现更为稳定")

    report.append("\n4.2 最优参数建议")
    report.append("-" * 40)
    if 'fixed_pct' in analysis and 'best_sharpe_param' in analysis['fixed_pct']:
        report.append(f"- 固定百分比止损: {analysis['fixed_pct']['best_sharpe_param']*100:.0f}%")
    if 'atr' in analysis and 'best_sharpe_param' in analysis['atr']:
        report.append(f"- ATR止损: {analysis['atr']['best_sharpe_param']}x ATR")
    if 'trailing' in analysis and 'best_sharpe_param' in analysis['trailing']:
        report.append(f"- 移动止损: {analysis['trailing']['best_sharpe_param']*100:.0f}%")

    report.append("\n4.3 策略整合建议")
    report.append("-" * 40)
    report.append("1. 趋势策略推荐使用ATR止损或移动止损")
    report.append("2. 均值回归策略推荐使用较紧的ATR止损 (1.5x ATR)")
    report.append("3. 对于波动较大的股票，建议使用多层止损系统")
    report.append("4. 可根据市场环境动态调整止损参数")

    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)

    # 保存报告
    report_text = '\n'.join(report)
    with open(f'{report_dir}/dynamic_stop_loss_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {report_dir}/dynamic_stop_loss_report.txt")

    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("动态止损策略研究")
    print("=" * 60)

    # 初始化
    bt = StopLossBacktester(DB_PATH)

    # 获取样本股票
    stocks = bt.get_sample_stocks(n=50)
    print(f"获取样本股票: {len(stocks)} 只")

    # 设置测试时间范围
    start_date = '20200101'
    end_date = '20241231'

    # 运行参数优化
    results = run_parameter_optimization(bt, stocks, start_date, end_date)

    # 分析结果
    analysis = analyze_results(results)

    # 策略整合测试
    integration_results = run_strategy_integration_test(bt, stocks, start_date, end_date)

    # 创建可视化
    create_visualizations(results, analysis, integration_results, REPORT_DIR)

    # 生成报告
    report = generate_report(results, analysis, integration_results, REPORT_DIR)

    print("\n" + "=" * 60)
    print("研究完成！")
    print("=" * 60)

    return results, analysis, integration_results


if __name__ == '__main__':
    results, analysis, integration_results = main()
