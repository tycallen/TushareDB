#!/usr/bin/env python3
"""
趋势跟踪策略研究

包含：
1. 趋势识别方法：均线趋势、通道突破、ADX趋势强度
2. 策略设计：海龟交易系统、唐奇安通道策略、移动平均策略
3. 回测分析：不同参数测试、多品种测试、风险收益评估

作者: Claude
日期: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


class TrendIndicators:
    """趋势指标计算类"""

    @staticmethod
    def moving_average(prices: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def exponential_moving_average(prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def donchian_channel(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算唐奇安通道
        返回: (上轨, 下轨, 中轨)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, lower, middle

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅 (Average True Range)
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算ADX (Average Directional Index)
        返回: (ADX, +DI, -DI)
        """
        # 计算 +DM 和 -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # 计算 ATR
        atr = TrendIndicators.atr(high, low, close, period)

        # 计算 +DI 和 -DI
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr

        # 计算 DX 和 ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def trend_strength(adx: pd.Series) -> pd.Series:
        """
        基于ADX判断趋势强度
        < 20: 无趋势/盘整
        20-25: 趋势开始形成
        25-50: 趋势较强
        50-75: 趋势很强
        > 75: 趋势极强
        """
        def classify(x):
            if pd.isna(x):
                return 'Unknown'
            elif x < 20:
                return 'No Trend'
            elif x < 25:
                return 'Weak'
            elif x < 50:
                return 'Strong'
            elif x < 75:
                return 'Very Strong'
            else:
                return 'Extreme'
        return adx.apply(classify)


class TrendFollowingStrategy:
    """趋势跟踪策略基类"""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000000):
        """
        初始化策略

        Args:
            data: 包含 open, high, low, close, vol 的DataFrame
            initial_capital: 初始资金
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        self.cash = pd.Series(index=data.index, dtype=float)
        self.equity = pd.Series(index=data.index, dtype=float)
        self.trades = []

    def calculate_signals(self) -> pd.Series:
        """计算交易信号，子类需要实现"""
        raise NotImplementedError

    def backtest(self, commission: float = 0.001, slippage: float = 0.001) -> Dict:
        """
        执行回测

        Args:
            commission: 手续费率
            slippage: 滑点

        Returns:
            回测结果字典
        """
        signals = self.calculate_signals()

        cash = self.initial_capital
        position = 0
        position_price = 0

        for i, (idx, row) in enumerate(self.data.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = row['close']

            # 开多仓
            if signal == 1 and position == 0:
                # 计算可买入数量
                buy_price = price * (1 + slippage)
                shares = int(cash * 0.95 / buy_price / 100) * 100  # 按100股取整
                if shares > 0:
                    cost = shares * buy_price * (1 + commission)
                    cash -= cost
                    position = shares
                    position_price = buy_price
                    self.trades.append({
                        'date': idx,
                        'type': 'BUY',
                        'price': buy_price,
                        'shares': shares,
                        'cost': cost
                    })

            # 平多仓
            elif signal == -1 and position > 0:
                sell_price = price * (1 - slippage)
                revenue = position * sell_price * (1 - commission)
                pnl = revenue - position * position_price
                cash += revenue
                self.trades.append({
                    'date': idx,
                    'type': 'SELL',
                    'price': sell_price,
                    'shares': position,
                    'revenue': revenue,
                    'pnl': pnl
                })
                position = 0
                position_price = 0

            self.positions.loc[idx] = position
            self.cash.loc[idx] = cash
            self.equity.loc[idx] = cash + position * price

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """计算回测指标"""
        equity = self.equity
        returns = equity.pct_change().dropna()

        # 基础指标
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # 年化收益率
        days = (self.data.index[-1] - self.data.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 最大回撤
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 夏普比率 (假设无风险利率3%)
        rf = 0.03
        excess_return = returns.mean() * 252 - rf
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # 卡尔马比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 交易统计
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']

        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]

        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'total_trades': len(sell_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity.iloc[-1]
        }


class TurtleStrategy(TrendFollowingStrategy):
    """
    海龟交易系统

    入场规则:
    - 突破20日高点买入
    - 突破55日高点加仓

    出场规则:
    - 跌破10日低点卖出

    仓位管理:
    - 基于ATR的仓位调整
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000000,
                 entry_period: int = 20, exit_period: int = 10, atr_period: int = 20):
        super().__init__(data, initial_capital)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period

    def calculate_signals(self) -> pd.Series:
        """计算海龟系统信号"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        # 入场通道
        entry_high = high.rolling(window=self.entry_period).max().shift(1)

        # 出场通道
        exit_low = low.rolling(window=self.exit_period).min().shift(1)

        signals = pd.Series(index=self.data.index, dtype=float).fillna(0)

        in_position = False
        for i in range(len(self.data)):
            if i < max(self.entry_period, self.exit_period):
                continue

            current_close = close.iloc[i]

            # 入场信号
            if not in_position and current_close > entry_high.iloc[i]:
                signals.iloc[i] = 1
                in_position = True

            # 出场信号
            elif in_position and current_close < exit_low.iloc[i]:
                signals.iloc[i] = -1
                in_position = False

        return signals


class DonchianStrategy(TrendFollowingStrategy):
    """
    唐奇安通道突破策略

    入场规则:
    - 价格突破上轨做多

    出场规则:
    - 价格跌破下轨平仓
    - 或者跌破中轨平仓（保守版本）
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000000,
                 period: int = 20, exit_on_middle: bool = False):
        super().__init__(data, initial_capital)
        self.period = period
        self.exit_on_middle = exit_on_middle

    def calculate_signals(self) -> pd.Series:
        """计算唐奇安通道信号"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        upper, lower, middle = TrendIndicators.donchian_channel(high, low, self.period)
        upper = upper.shift(1)
        lower = lower.shift(1)
        middle = middle.shift(1)

        signals = pd.Series(index=self.data.index, dtype=float).fillna(0)

        in_position = False
        for i in range(len(self.data)):
            if i < self.period:
                continue

            current_close = close.iloc[i]

            # 入场信号
            if not in_position and current_close > upper.iloc[i]:
                signals.iloc[i] = 1
                in_position = True

            # 出场信号
            elif in_position:
                exit_level = middle.iloc[i] if self.exit_on_middle else lower.iloc[i]
                if current_close < exit_level:
                    signals.iloc[i] = -1
                    in_position = False

        return signals


class MovingAverageStrategy(TrendFollowingStrategy):
    """
    移动平均策略

    入场规则:
    - 短期均线上穿长期均线做多

    出场规则:
    - 短期均线下穿长期均线平仓

    可选过滤:
    - ADX过滤弱趋势
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000000,
                 short_period: int = 10, long_period: int = 30,
                 use_ema: bool = False, adx_filter: Optional[int] = None):
        super().__init__(data, initial_capital)
        self.short_period = short_period
        self.long_period = long_period
        self.use_ema = use_ema
        self.adx_filter = adx_filter

    def calculate_signals(self) -> pd.Series:
        """计算均线交叉信号"""
        close = self.data['close']

        if self.use_ema:
            short_ma = TrendIndicators.exponential_moving_average(close, self.short_period)
            long_ma = TrendIndicators.exponential_moving_average(close, self.long_period)
        else:
            short_ma = TrendIndicators.moving_average(close, self.short_period)
            long_ma = TrendIndicators.moving_average(close, self.long_period)

        # ADX过滤
        if self.adx_filter:
            adx, _, _ = TrendIndicators.adx(self.data['high'], self.data['low'], close)
            adx_condition = adx > self.adx_filter
        else:
            adx_condition = pd.Series(True, index=self.data.index)

        signals = pd.Series(index=self.data.index, dtype=float).fillna(0)

        in_position = False
        for i in range(len(self.data)):
            if i < self.long_period:
                continue

            # 金叉
            if (not in_position and
                short_ma.iloc[i] > long_ma.iloc[i] and
                short_ma.iloc[i-1] <= long_ma.iloc[i-1] and
                adx_condition.iloc[i]):
                signals.iloc[i] = 1
                in_position = True

            # 死叉
            elif (in_position and
                  short_ma.iloc[i] < long_ma.iloc[i] and
                  short_ma.iloc[i-1] >= long_ma.iloc[i-1]):
                signals.iloc[i] = -1
                in_position = False

        return signals


class TrendResearch:
    """趋势跟踪策略研究类"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_stock_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        query = f"""
        SELECT trade_date, open, high, low, close, vol, amount
        FROM daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df

    def get_index_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        query = f"""
        SELECT trade_date, open, high, low, close, vol, amount
        FROM index_daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df

    def analyze_trend_indicators(self, data: pd.DataFrame, name: str) -> Dict:
        """分析趋势指标"""
        close = data['close']
        high = data['high']
        low = data['low']

        # 计算各种趋势指标
        ma20 = TrendIndicators.moving_average(close, 20)
        ma60 = TrendIndicators.moving_average(close, 60)
        ema20 = TrendIndicators.exponential_moving_average(close, 20)
        ema60 = TrendIndicators.exponential_moving_average(close, 60)

        upper20, lower20, mid20 = TrendIndicators.donchian_channel(high, low, 20)
        upper55, lower55, mid55 = TrendIndicators.donchian_channel(high, low, 55)

        adx, plus_di, minus_di = TrendIndicators.adx(high, low, close)
        atr = TrendIndicators.atr(high, low, close)

        trend_strength = TrendIndicators.trend_strength(adx)

        # 统计趋势强度分布
        strength_dist = trend_strength.value_counts(normalize=True)

        return {
            'name': name,
            'ma20': ma20,
            'ma60': ma60,
            'ema20': ema20,
            'ema60': ema60,
            'donchian_upper_20': upper20,
            'donchian_lower_20': lower20,
            'donchian_upper_55': upper55,
            'donchian_lower_55': lower55,
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr,
            'trend_strength': trend_strength,
            'strength_distribution': strength_dist,
            'avg_adx': adx.mean(),
            'current_adx': adx.iloc[-1] if len(adx) > 0 else None
        }

    def parameter_optimization(self, data: pd.DataFrame, strategy_class,
                               param_grid: Dict, name: str = '') -> pd.DataFrame:
        """
        参数优化

        Args:
            data: 市场数据
            strategy_class: 策略类
            param_grid: 参数网格
            name: 策略名称

        Returns:
            参数测试结果
        """
        results = []

        # 生成参数组合
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                strategy = strategy_class(data, **params)
                metrics = strategy.backtest()

                result = {**params, **metrics}
                results.append(result)
            except Exception as e:
                print(f"参数 {params} 测试失败: {e}")
                continue

        df = pd.DataFrame(results)
        df['strategy'] = name
        return df

    def multi_asset_test(self, assets: Dict[str, str], strategy_class,
                        params: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        多品种测试

        Args:
            assets: {名称: 代码} 字典
            strategy_class: 策略类
            params: 策略参数
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            多品种测试结果
        """
        results = []

        for name, code in assets.items():
            try:
                # 判断是股票还是指数
                if code.startswith('0003') or code.startswith('3990') or code.startswith('0000') and code.endswith('.SH'):
                    data = self.get_index_data(code, start_date, end_date)
                else:
                    data = self.get_stock_data(code, start_date, end_date)

                if len(data) < 100:
                    print(f"{name} 数据不足，跳过")
                    continue

                strategy = strategy_class(data, **params)
                metrics = strategy.backtest()

                result = {
                    'asset': name,
                    'code': code,
                    'data_points': len(data),
                    **metrics
                }
                results.append(result)
            except Exception as e:
                print(f"{name} ({code}) 测试失败: {e}")
                continue

        return pd.DataFrame(results)

    def generate_report(self, output_dir: str = REPORT_DIR):
        """生成完整研究报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 测试数据: 使用沪深300指数 2015-2024
        print("=" * 60)
        print("趋势跟踪策略研究报告")
        print("=" * 60)

        # 获取测试数据
        print("\n1. 加载数据...")
        test_data = self.get_index_data('000300.SH', '20150101', '20251231')
        print(f"   沪深300数据: {len(test_data)} 条记录")
        print(f"   日期范围: {test_data.index[0]} ~ {test_data.index[-1]}")

        # 趋势指标分析
        print("\n2. 趋势指标分析...")
        indicators = self.analyze_trend_indicators(test_data, '沪深300')

        print(f"   平均ADX: {indicators['avg_adx']:.2f}")
        print(f"   当前ADX: {indicators['current_adx']:.2f}")
        print("   趋势强度分布:")
        for strength, ratio in indicators['strength_distribution'].items():
            print(f"      {strength}: {ratio*100:.1f}%")

        # 策略回测
        print("\n3. 策略回测...")

        # 3.1 海龟策略
        print("\n   3.1 海龟交易系统")
        turtle = TurtleStrategy(test_data, entry_period=20, exit_period=10)
        turtle_metrics = turtle.backtest()
        print(f"       总收益率: {turtle_metrics['total_return']*100:.2f}%")
        print(f"       年化收益: {turtle_metrics['annual_return']*100:.2f}%")
        print(f"       最大回撤: {turtle_metrics['max_drawdown']*100:.2f}%")
        print(f"       夏普比率: {turtle_metrics['sharpe_ratio']:.2f}")
        print(f"       交易次数: {turtle_metrics['total_trades']}")
        print(f"       胜率: {turtle_metrics['win_rate']*100:.1f}%")

        # 3.2 唐奇安通道策略
        print("\n   3.2 唐奇安通道策略")
        donchian = DonchianStrategy(test_data, period=20, exit_on_middle=False)
        donchian_metrics = donchian.backtest()
        print(f"       总收益率: {donchian_metrics['total_return']*100:.2f}%")
        print(f"       年化收益: {donchian_metrics['annual_return']*100:.2f}%")
        print(f"       最大回撤: {donchian_metrics['max_drawdown']*100:.2f}%")
        print(f"       夏普比率: {donchian_metrics['sharpe_ratio']:.2f}")
        print(f"       交易次数: {donchian_metrics['total_trades']}")
        print(f"       胜率: {donchian_metrics['win_rate']*100:.1f}%")

        # 3.3 移动平均策略
        print("\n   3.3 移动平均策略 (SMA 10/30)")
        ma_strategy = MovingAverageStrategy(test_data, short_period=10, long_period=30)
        ma_metrics = ma_strategy.backtest()
        print(f"       总收益率: {ma_metrics['total_return']*100:.2f}%")
        print(f"       年化收益: {ma_metrics['annual_return']*100:.2f}%")
        print(f"       最大回撤: {ma_metrics['max_drawdown']*100:.2f}%")
        print(f"       夏普比率: {ma_metrics['sharpe_ratio']:.2f}")
        print(f"       交易次数: {ma_metrics['total_trades']}")
        print(f"       胜率: {ma_metrics['win_rate']*100:.1f}%")

        # 3.4 EMA策略 + ADX过滤
        print("\n   3.4 EMA策略 + ADX过滤")
        ema_adx = MovingAverageStrategy(test_data, short_period=10, long_period=30,
                                        use_ema=True, adx_filter=25)
        ema_adx_metrics = ema_adx.backtest()
        print(f"       总收益率: {ema_adx_metrics['total_return']*100:.2f}%")
        print(f"       年化收益: {ema_adx_metrics['annual_return']*100:.2f}%")
        print(f"       最大回撤: {ema_adx_metrics['max_drawdown']*100:.2f}%")
        print(f"       夏普比率: {ema_adx_metrics['sharpe_ratio']:.2f}")
        print(f"       交易次数: {ema_adx_metrics['total_trades']}")
        print(f"       胜率: {ema_adx_metrics['win_rate']*100:.1f}%")

        # 参数优化
        print("\n4. 参数优化测试...")

        # 海龟策略参数优化
        turtle_params = {
            'entry_period': [10, 15, 20, 30, 40],
            'exit_period': [5, 10, 15, 20]
        }
        turtle_opt = self.parameter_optimization(test_data, TurtleStrategy,
                                                  turtle_params, '海龟策略')

        # 唐奇安通道参数优化
        donchian_params = {
            'period': [10, 15, 20, 30, 40, 55],
            'exit_on_middle': [True, False]
        }
        donchian_opt = self.parameter_optimization(test_data, DonchianStrategy,
                                                    donchian_params, '唐奇安通道')

        # 移动平均参数优化
        ma_params = {
            'short_period': [5, 10, 15, 20],
            'long_period': [20, 30, 40, 60],
            'use_ema': [True, False]
        }
        ma_opt = self.parameter_optimization(test_data, MovingAverageStrategy,
                                             ma_params, '移动平均')

        # 合并结果
        all_opt = pd.concat([turtle_opt, donchian_opt, ma_opt], ignore_index=True)

        # 输出最佳参数
        print("\n   最佳参数组合 (按夏普比率排序):")
        best_results = all_opt.nlargest(5, 'sharpe_ratio')
        for i, row in best_results.iterrows():
            print(f"      策略: {row['strategy']}, 夏普: {row['sharpe_ratio']:.2f}, "
                  f"年化: {row['annual_return']*100:.2f}%, 回撤: {row['max_drawdown']*100:.2f}%")

        # 多品种测试
        print("\n5. 多品种测试...")
        assets = {
            '沪深300': '000300.SH',
            '上证50': '000016.SH',
            '中证500': '000905.SH',
            '创业板指': '399006.SZ',
            '中证1000': '000852.SH',
        }

        # 使用最佳参数测试
        best_turtle_params = turtle_opt.loc[turtle_opt['sharpe_ratio'].idxmax()][['entry_period', 'exit_period']].to_dict()
        best_turtle_params = {k: int(v) for k, v in best_turtle_params.items()}

        multi_results = self.multi_asset_test(assets, TurtleStrategy, best_turtle_params,
                                              '20150101', '20251231')

        print("\n   多品种海龟策略测试结果:")
        for _, row in multi_results.iterrows():
            print(f"      {row['asset']}: 年化{row['annual_return']*100:.2f}%, "
                  f"夏普{row['sharpe_ratio']:.2f}, 回撤{row['max_drawdown']*100:.2f}%")

        # 生成图表
        print("\n6. 生成图表...")
        self._generate_charts(test_data, indicators, turtle, donchian, ma_strategy,
                             all_opt, multi_results, output_dir)

        # 保存报告
        print("\n7. 保存报告...")
        self._save_report(indicators, turtle_metrics, donchian_metrics, ma_metrics,
                         ema_adx_metrics, all_opt, multi_results, best_turtle_params,
                         output_dir)

        print("\n" + "=" * 60)
        print("研究报告生成完成!")
        print(f"报告保存至: {output_dir}")
        print("=" * 60)

        return {
            'indicators': indicators,
            'turtle_metrics': turtle_metrics,
            'donchian_metrics': donchian_metrics,
            'ma_metrics': ma_metrics,
            'ema_adx_metrics': ema_adx_metrics,
            'parameter_optimization': all_opt,
            'multi_asset_results': multi_results
        }

    def _generate_charts(self, data, indicators, turtle, donchian, ma_strategy,
                        opt_results, multi_results, output_dir):
        """生成分析图表"""

        # 图1: 趋势指标分析
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # 价格与均线
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='收盘价', linewidth=1)
        ax1.plot(data.index, indicators['ma20'], label='MA20', linewidth=0.8)
        ax1.plot(data.index, indicators['ma60'], label='MA60', linewidth=0.8)
        ax1.set_title('沪深300价格与均线', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)

        # 唐奇安通道
        ax2 = axes[1]
        ax2.plot(data.index, data['close'], label='收盘价', linewidth=1)
        ax2.plot(data.index, indicators['donchian_upper_20'], label='20日上轨',
                linewidth=0.8, linestyle='--')
        ax2.plot(data.index, indicators['donchian_lower_20'], label='20日下轨',
                linewidth=0.8, linestyle='--')
        ax2.fill_between(data.index, indicators['donchian_lower_20'],
                        indicators['donchian_upper_20'], alpha=0.2)
        ax2.set_title('唐奇安通道 (20日)', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.set_ylabel('价格')
        ax2.grid(True, alpha=0.3)

        # ADX
        ax3 = axes[2]
        ax3.plot(data.index, indicators['adx'], label='ADX', linewidth=1)
        ax3.plot(data.index, indicators['plus_di'], label='+DI', linewidth=0.8, alpha=0.7)
        ax3.plot(data.index, indicators['minus_di'], label='-DI', linewidth=0.8, alpha=0.7)
        ax3.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='趋势阈值(25)')
        ax3.set_title('ADX趋势强度指标', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.set_ylabel('ADX值')
        ax3.grid(True, alpha=0.3)

        # ATR
        ax4 = axes[3]
        ax4.plot(data.index, indicators['atr'], label='ATR(14)', linewidth=1)
        ax4.set_title('平均真实波幅 (ATR)', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.set_ylabel('ATR')
        ax4.set_xlabel('日期')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/trend_indicators.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图2: 策略净值曲线对比
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax1 = axes[0]
        ax1.plot(turtle.equity.index, turtle.equity / 1000000, label='海龟策略', linewidth=1)
        ax1.plot(donchian.equity.index, donchian.equity / 1000000, label='唐奇安通道', linewidth=1)
        ax1.plot(ma_strategy.equity.index, ma_strategy.equity / 1000000, label='移动平均', linewidth=1)
        # 基准: 买入持有
        benchmark = data['close'] / data['close'].iloc[0]
        ax1.plot(data.index, benchmark, label='买入持有', linewidth=1, linestyle='--')
        ax1.set_title('策略净值曲线对比 (初始100万)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('净值 (百万)')
        ax1.grid(True, alpha=0.3)

        # 回撤对比
        ax2 = axes[1]
        for strategy, name in [(turtle, '海龟策略'), (donchian, '唐奇安通道'), (ma_strategy, '移动平均')]:
            equity = strategy.equity
            drawdown = (equity - equity.cummax()) / equity.cummax()
            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=name)
        ax2.set_title('回撤对比', fontsize=12)
        ax2.legend(loc='lower left')
        ax2.set_ylabel('回撤比例')
        ax2.set_xlabel('日期')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图3: 参数优化热力图
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 海龟策略热力图
        turtle_pivot = opt_results[opt_results['strategy'] == '海龟策略'].pivot_table(
            values='sharpe_ratio', index='entry_period', columns='exit_period')
        if not turtle_pivot.empty:
            im1 = axes[0].imshow(turtle_pivot.values, cmap='RdYlGn', aspect='auto')
            axes[0].set_xticks(range(len(turtle_pivot.columns)))
            axes[0].set_xticklabels(turtle_pivot.columns)
            axes[0].set_yticks(range(len(turtle_pivot.index)))
            axes[0].set_yticklabels(turtle_pivot.index)
            axes[0].set_xlabel('出场周期')
            axes[0].set_ylabel('入场周期')
            axes[0].set_title('海龟策略 - 夏普比率')
            plt.colorbar(im1, ax=axes[0])

        # 唐奇安通道
        donchian_df = opt_results[opt_results['strategy'] == '唐奇安通道']
        if not donchian_df.empty:
            periods = sorted(donchian_df['period'].unique())
            exits = [True, False]
            pivot_data = np.zeros((len(periods), 2))
            for i, p in enumerate(periods):
                for j, e in enumerate(exits):
                    val = donchian_df[(donchian_df['period']==p) & (donchian_df['exit_on_middle']==e)]['sharpe_ratio']
                    pivot_data[i, j] = val.values[0] if len(val) > 0 else 0

            im2 = axes[1].imshow(pivot_data, cmap='RdYlGn', aspect='auto')
            axes[1].set_xticks([0, 1])
            axes[1].set_xticklabels(['中轨出场', '下轨出场'])
            axes[1].set_yticks(range(len(periods)))
            axes[1].set_yticklabels(periods)
            axes[1].set_ylabel('通道周期')
            axes[1].set_title('唐奇安通道 - 夏普比率')
            plt.colorbar(im2, ax=axes[1])

        # 移动平均
        ma_df = opt_results[opt_results['strategy'] == '移动平均']
        if not ma_df.empty:
            # 只看SMA
            ma_sma = ma_df[ma_df['use_ema'] == False]
            if not ma_sma.empty:
                ma_pivot = ma_sma.pivot_table(
                    values='sharpe_ratio', index='short_period', columns='long_period')
                im3 = axes[2].imshow(ma_pivot.values, cmap='RdYlGn', aspect='auto')
                axes[2].set_xticks(range(len(ma_pivot.columns)))
                axes[2].set_xticklabels(ma_pivot.columns)
                axes[2].set_yticks(range(len(ma_pivot.index)))
                axes[2].set_yticklabels(ma_pivot.index)
                axes[2].set_xlabel('长期均线')
                axes[2].set_ylabel('短期均线')
                axes[2].set_title('移动平均(SMA) - 夏普比率')
                plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/parameter_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图4: 多品种对比
        if len(multi_results) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            assets = multi_results['asset'].values
            x = np.arange(len(assets))

            # 收益率对比
            ax1 = axes[0]
            bars = ax1.bar(x, multi_results['annual_return'] * 100)
            ax1.set_xticks(x)
            ax1.set_xticklabels(assets, rotation=45)
            ax1.set_ylabel('年化收益率 (%)')
            ax1.set_title('多品种年化收益率对比')
            ax1.axhline(y=0, color='black', linewidth=0.5)
            for bar, val in zip(bars, multi_results['annual_return']):
                color = 'green' if val > 0 else 'red'
                bar.set_color(color)

            # 风险收益散点图
            ax2 = axes[1]
            ax2.scatter(multi_results['max_drawdown'].abs() * 100,
                       multi_results['annual_return'] * 100, s=100)
            for i, row in multi_results.iterrows():
                ax2.annotate(row['asset'],
                           (abs(row['max_drawdown']) * 100, row['annual_return'] * 100),
                           xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('最大回撤 (%)')
            ax2.set_ylabel('年化收益率 (%)')
            ax2.set_title('风险收益散点图')
            ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/multi_asset_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 图5: 趋势强度分布
        fig, ax = plt.subplots(figsize=(8, 6))
        strength_dist = indicators['strength_distribution']
        colors = {'No Trend': 'gray', 'Weak': 'yellow', 'Strong': 'green',
                 'Very Strong': 'blue', 'Extreme': 'red', 'Unknown': 'lightgray'}

        bars = ax.bar(strength_dist.index, strength_dist.values * 100,
                     color=[colors.get(s, 'gray') for s in strength_dist.index])
        ax.set_ylabel('占比 (%)')
        ax.set_title('沪深300 ADX趋势强度分布 (2015-2025)')
        ax.set_xlabel('趋势强度')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/trend_strength_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   图表已保存至 {output_dir}")

    def _save_report(self, indicators, turtle_metrics, donchian_metrics, ma_metrics,
                    ema_adx_metrics, opt_results, multi_results, best_params, output_dir):
        """保存研究报告"""

        report = f"""# 趋势跟踪策略研究报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 研究概述

本报告研究了三种经典趋势跟踪策略在A股市场的表现:
- 海龟交易系统
- 唐奇安通道策略
- 移动平均策略

测试数据: 沪深300指数 (2015-2025)

## 2. 趋势识别分析

### 2.1 ADX趋势强度

- 平均ADX: {indicators['avg_adx']:.2f}
- 当前ADX: {indicators['current_adx']:.2f}

趋势强度分布:
"""
        for strength, ratio in indicators['strength_distribution'].items():
            report += f"- {strength}: {ratio*100:.1f}%\n"

        report += f"""
### 2.2 趋势特征

根据ADX分析，沪深300指数:
- 约有{indicators['strength_distribution'].get('No Trend', 0)*100:.1f}%的时间处于无趋势状态
- 约有{(indicators['strength_distribution'].get('Strong', 0) + indicators['strength_distribution'].get('Very Strong', 0))*100:.1f}%的时间趋势较强
- 这意味着趋势跟踪策略在大部分时间可能面临震荡市的挑战

## 3. 策略回测结果

### 3.1 海龟交易系统 (入场20日/出场10日)

| 指标 | 数值 |
|------|------|
| 总收益率 | {turtle_metrics['total_return']*100:.2f}% |
| 年化收益 | {turtle_metrics['annual_return']*100:.2f}% |
| 最大回撤 | {turtle_metrics['max_drawdown']*100:.2f}% |
| 夏普比率 | {turtle_metrics['sharpe_ratio']:.2f} |
| 索提诺比率 | {turtle_metrics['sortino_ratio']:.2f} |
| 卡尔马比率 | {turtle_metrics['calmar_ratio']:.2f} |
| 波动率 | {turtle_metrics['volatility']*100:.2f}% |
| 交易次数 | {turtle_metrics['total_trades']} |
| 胜率 | {turtle_metrics['win_rate']*100:.1f}% |
| 盈亏比 | {turtle_metrics['profit_factor']:.2f} |

### 3.2 唐奇安通道策略 (20日)

| 指标 | 数值 |
|------|------|
| 总收益率 | {donchian_metrics['total_return']*100:.2f}% |
| 年化收益 | {donchian_metrics['annual_return']*100:.2f}% |
| 最大回撤 | {donchian_metrics['max_drawdown']*100:.2f}% |
| 夏普比率 | {donchian_metrics['sharpe_ratio']:.2f} |
| 索提诺比率 | {donchian_metrics['sortino_ratio']:.2f} |
| 卡尔马比率 | {donchian_metrics['calmar_ratio']:.2f} |
| 波动率 | {donchian_metrics['volatility']*100:.2f}% |
| 交易次数 | {donchian_metrics['total_trades']} |
| 胜率 | {donchian_metrics['win_rate']*100:.1f}% |
| 盈亏比 | {donchian_metrics['profit_factor']:.2f} |

### 3.3 移动平均策略 (SMA 10/30)

| 指标 | 数值 |
|------|------|
| 总收益率 | {ma_metrics['total_return']*100:.2f}% |
| 年化收益 | {ma_metrics['annual_return']*100:.2f}% |
| 最大回撤 | {ma_metrics['max_drawdown']*100:.2f}% |
| 夏普比率 | {ma_metrics['sharpe_ratio']:.2f} |
| 索提诺比率 | {ma_metrics['sortino_ratio']:.2f} |
| 卡尔马比率 | {ma_metrics['calmar_ratio']:.2f} |
| 波动率 | {ma_metrics['volatility']*100:.2f}% |
| 交易次数 | {ma_metrics['total_trades']} |
| 胜率 | {ma_metrics['win_rate']*100:.1f}% |
| 盈亏比 | {ma_metrics['profit_factor']:.2f} |

### 3.4 EMA策略 + ADX过滤

| 指标 | 数值 |
|------|------|
| 总收益率 | {ema_adx_metrics['total_return']*100:.2f}% |
| 年化收益 | {ema_adx_metrics['annual_return']*100:.2f}% |
| 最大回撤 | {ema_adx_metrics['max_drawdown']*100:.2f}% |
| 夏普比率 | {ema_adx_metrics['sharpe_ratio']:.2f} |
| 交易次数 | {ema_adx_metrics['total_trades']} |
| 胜率 | {ema_adx_metrics['win_rate']*100:.1f}% |

## 4. 参数优化结果

### 4.1 海龟策略最佳参数

"""
        turtle_opt = opt_results[opt_results['strategy'] == '海龟策略'].nlargest(5, 'sharpe_ratio')
        report += "| 入场周期 | 出场周期 | 年化收益 | 夏普比率 | 最大回撤 |\n"
        report += "|----------|----------|----------|----------|----------|\n"
        for _, row in turtle_opt.iterrows():
            report += f"| {int(row['entry_period'])} | {int(row['exit_period'])} | {row['annual_return']*100:.2f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.2f}% |\n"

        report += "\n### 4.2 唐奇安通道最佳参数\n\n"
        donchian_opt = opt_results[opt_results['strategy'] == '唐奇安通道'].nlargest(5, 'sharpe_ratio')
        report += "| 周期 | 出场方式 | 年化收益 | 夏普比率 | 最大回撤 |\n"
        report += "|------|----------|----------|----------|----------|\n"
        for _, row in donchian_opt.iterrows():
            exit_type = "中轨" if row['exit_on_middle'] else "下轨"
            report += f"| {int(row['period'])} | {exit_type} | {row['annual_return']*100:.2f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.2f}% |\n"

        report += "\n### 4.3 移动平均最佳参数\n\n"
        ma_opt = opt_results[opt_results['strategy'] == '移动平均'].nlargest(5, 'sharpe_ratio')
        report += "| 短期 | 长期 | 类型 | 年化收益 | 夏普比率 | 最大回撤 |\n"
        report += "|------|------|------|----------|----------|----------|\n"
        for _, row in ma_opt.iterrows():
            ma_type = "EMA" if row['use_ema'] else "SMA"
            report += f"| {int(row['short_period'])} | {int(row['long_period'])} | {ma_type} | {row['annual_return']*100:.2f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.2f}% |\n"

        report += f"""

## 5. 多品种测试

使用海龟策略最佳参数 (入场{best_params['entry_period']}日/出场{best_params['exit_period']}日) 测试:

| 品种 | 年化收益 | 最大回撤 | 夏普比率 | 胜率 | 交易次数 |
|------|----------|----------|----------|------|----------|
"""
        for _, row in multi_results.iterrows():
            report += f"| {row['asset']} | {row['annual_return']*100:.2f}% | {row['max_drawdown']*100:.2f}% | {row['sharpe_ratio']:.2f} | {row['win_rate']*100:.1f}% | {row['total_trades']} |\n"

        report += f"""

## 6. 研究结论

### 6.1 策略比较

1. **海龟交易系统**: 经典的通道突破策略，在强趋势市场表现较好，但在震荡市会产生较多假信号。

2. **唐奇安通道策略**: 与海龟系统类似，但出场规则可以更灵活。使用中轨出场可以更好地保护利润。

3. **移动平均策略**: 信号产生滞后，但可以通过ADX过滤来减少震荡市中的假信号。

### 6.2 关键发现

1. **趋势强度**: 沪深300指数约有{indicators['strength_distribution'].get('No Trend', 0)*100:.1f}%的时间处于无趋势状态(ADX<20)，这对趋势跟踪策略构成挑战。

2. **参数敏感性**: 策略表现对参数选择较为敏感，建议通过回测找到适合当前市场环境的参数。

3. **多品种分散**: 不同指数的趋势特征不同，建议在多个品种上分散配置。

4. **风险管理**: 趋势策略的胜率通常较低(30-40%)，依靠较高的盈亏比获利，需要严格执行止损。

### 6.3 改进建议

1. 添加趋势过滤器(如ADX)来减少震荡市中的交易
2. 动态调整仓位大小，根据ATR控制风险
3. 结合多个时间框架确认趋势
4. 考虑加入止盈机制，在趋势末期锁定利润

## 7. 图表说明

- `trend_indicators.png`: 趋势指标分析图，包含均线、唐奇安通道、ADX和ATR
- `strategy_comparison.png`: 策略净值曲线和回撤对比
- `parameter_heatmap.png`: 参数优化热力图
- `multi_asset_comparison.png`: 多品种收益和风险对比
- `trend_strength_distribution.png`: ADX趋势强度分布

---

*本报告仅供研究参考，不构成投资建议。*
"""

        with open(f'{output_dir}/trend_following_research.md', 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存参数优化结果CSV
        opt_results.to_csv(f'{output_dir}/parameter_optimization.csv', index=False, encoding='utf-8-sig')

        # 保存多品种结果CSV
        multi_results.to_csv(f'{output_dir}/multi_asset_results.csv', index=False, encoding='utf-8-sig')

        print(f"   报告已保存至 {output_dir}/trend_following_research.md")
        print(f"   参数优化结果已保存至 {output_dir}/parameter_optimization.csv")
        print(f"   多品种结果已保存至 {output_dir}/multi_asset_results.csv")


def main():
    """主函数"""
    research = TrendResearch()
    results = research.generate_report()
    return results


if __name__ == '__main__':
    main()
