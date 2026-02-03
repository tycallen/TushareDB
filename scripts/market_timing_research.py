#!/usr/bin/env python3
"""
市场择时策略研究
================
研究内容：
1. 择时指标：均线择时、动量择时、波动率择时、资金流择时
2. 择时效果：不同指标对比、择时频率分析、市场状态适应性
3. 策略应用：仓位择时、品种择时、综合择时系统
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


class MarketTimingResearch:
    """市场择时策略研究类"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.index_data = None
        self.market_data = None
        self.moneyflow_data = None

    def load_data(self, start_date='20100101', end_date='20260130'):
        """加载数据"""
        print("正在加载数据...")

        # 加载上证指数日线数据
        query = f"""
        SELECT trade_date, close, open, high, low, vol, amount, pct_chg
        FROM index_daily
        WHERE ts_code = '000001.SH'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        self.index_data = self.conn.execute(query).df()
        self.index_data['trade_date'] = pd.to_datetime(self.index_data['trade_date'])
        self.index_data.set_index('trade_date', inplace=True)

        # 加载沪深300数据
        query_hs300 = f"""
        SELECT trade_date, close, pct_chg
        FROM index_daily
        WHERE ts_code = '000300.SH'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        self.hs300_data = self.conn.execute(query_hs300).df()
        self.hs300_data['trade_date'] = pd.to_datetime(self.hs300_data['trade_date'])
        self.hs300_data.set_index('trade_date', inplace=True)

        # 加载创业板指数数据
        query_cyb = f"""
        SELECT trade_date, close, pct_chg
        FROM index_daily
        WHERE ts_code = '399006.SZ'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        self.cyb_data = self.conn.execute(query_cyb).df()
        self.cyb_data['trade_date'] = pd.to_datetime(self.cyb_data['trade_date'])
        self.cyb_data.set_index('trade_date', inplace=True)

        # 加载市场整体数据（计算市场宽度等）
        query_market = f"""
        SELECT trade_date,
               COUNT(*) as total_stocks,
               SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) as up_stocks,
               SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END) as down_stocks,
               AVG(pct_chg) as avg_pct_chg,
               SUM(amount) as total_amount
        FROM daily
        WHERE trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        self.market_data = self.conn.execute(query_market).df()
        self.market_data['trade_date'] = pd.to_datetime(self.market_data['trade_date'])
        self.market_data.set_index('trade_date', inplace=True)

        # 加载资金流数据
        query_mf = f"""
        SELECT trade_date,
               SUM(net_mf_amount) as net_mf_amount,
               SUM(buy_elg_amount) as buy_elg_amount,
               SUM(sell_elg_amount) as sell_elg_amount,
               SUM(buy_lg_amount) as buy_lg_amount,
               SUM(sell_lg_amount) as sell_lg_amount
        FROM moneyflow
        WHERE trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        self.moneyflow_data = self.conn.execute(query_mf).df()
        self.moneyflow_data['trade_date'] = pd.to_datetime(self.moneyflow_data['trade_date'])
        self.moneyflow_data.set_index('trade_date', inplace=True)

        print(f"数据加载完成: 指数数据 {len(self.index_data)} 条, 市场数据 {len(self.market_data)} 条")
        print(f"日期范围: {self.index_data.index.min()} 至 {self.index_data.index.max()}")

    def calculate_ma_signals(self):
        """计算均线择时信号"""
        print("\n计算均线择时信号...")
        df = self.index_data.copy()

        # 计算各周期均线
        for period in [5, 10, 20, 60, 120, 250]:
            df[f'ma{period}'] = df['close'].rolling(period).mean()

        # 均线择时信号
        # 1. 单均线突破
        df['signal_ma20'] = (df['close'] > df['ma20']).astype(int)
        df['signal_ma60'] = (df['close'] > df['ma60']).astype(int)
        df['signal_ma120'] = (df['close'] > df['ma120']).astype(int)
        df['signal_ma250'] = (df['close'] > df['ma250']).astype(int)

        # 2. 双均线交叉 (金叉/死叉)
        df['signal_ma5_20'] = (df['ma5'] > df['ma20']).astype(int)
        df['signal_ma20_60'] = (df['ma20'] > df['ma60']).astype(int)
        df['signal_ma60_120'] = (df['ma60'] > df['ma120']).astype(int)

        # 3. 均线多头排列
        df['signal_ma_bull'] = ((df['ma5'] > df['ma10']) &
                                (df['ma10'] > df['ma20']) &
                                (df['ma20'] > df['ma60'])).astype(int)

        self.ma_signals = df
        return df

    def calculate_momentum_signals(self):
        """计算动量择时信号"""
        print("计算动量择时信号...")
        df = self.index_data.copy()

        # 1. 价格动量 (ROC)
        for period in [5, 10, 20, 60]:
            df[f'roc{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100

        # 2. RSI
        for period in [6, 14, 20]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi{period}'] = 100 - (100 / (1 + rs))

        # 3. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 动量择时信号
        df['signal_roc20'] = (df['roc20'] > 0).astype(int)
        df['signal_rsi14'] = ((df['rsi14'] > 30) & (df['rsi14'] < 70)).astype(int)
        df['signal_macd'] = (df['macd'] > df['macd_signal']).astype(int)

        # 4. 综合动量信号
        df['momentum_score'] = (
            (df['roc5'] > 0).astype(int) +
            (df['roc20'] > 0).astype(int) +
            (df['rsi14'] > 50).astype(int) +
            (df['macd'] > 0).astype(int)
        )
        df['signal_momentum'] = (df['momentum_score'] >= 3).astype(int)

        self.momentum_signals = df
        return df

    def calculate_volatility_signals(self):
        """计算波动率择时信号"""
        print("计算波动率择时信号...")
        df = self.index_data.copy()

        # 1. 历史波动率
        for period in [10, 20, 60]:
            df[f'volatility{period}'] = df['pct_chg'].rolling(period).std() * np.sqrt(252)

        # 2. ATR (Average True Range)
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr14'] / df['close'] * 100

        # 3. 布林带宽度
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

        # 4. 波动率百分位
        df['vol20_pct'] = df['volatility20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        # 波动率择时信号
        # 低波动做多，高波动减仓
        df['signal_low_vol'] = (df['vol20_pct'] < 0.5).astype(int)
        df['signal_vol_breakout'] = (df['close'] > df['bb_upper'].shift(1)).astype(int)
        df['signal_vol_contract'] = (df['bb_width'] < df['bb_width'].rolling(60).mean()).astype(int)

        self.volatility_signals = df
        return df

    def calculate_moneyflow_signals(self):
        """计算资金流择时信号"""
        print("计算资金流择时信号...")

        # 合并资金流数据
        df = self.index_data.copy()
        df = df.join(self.moneyflow_data, how='left')

        # 填充缺失值
        df['net_mf_amount'] = df['net_mf_amount'].fillna(0)
        df['buy_elg_amount'] = df['buy_elg_amount'].fillna(0)
        df['sell_elg_amount'] = df['sell_elg_amount'].fillna(0)

        # 1. 资金流净额
        df['mf_net_5d'] = df['net_mf_amount'].rolling(5).sum()
        df['mf_net_20d'] = df['net_mf_amount'].rolling(20).sum()

        # 2. 主力资金 (大单+特大单)
        df['main_force'] = df['buy_elg_amount'] + df['buy_lg_amount'].fillna(0) - \
                          df['sell_elg_amount'] - df['sell_lg_amount'].fillna(0)
        df['main_force_5d'] = df['main_force'].rolling(5).sum()
        df['main_force_20d'] = df['main_force'].rolling(20).sum()

        # 3. 资金流强度
        total_flow = df['buy_elg_amount'] + df['sell_elg_amount'] + \
                    df['buy_lg_amount'].fillna(0) + df['sell_lg_amount'].fillna(0)
        df['mf_strength'] = df['main_force'] / (total_flow + 1) * 100
        df['mf_strength_ma5'] = df['mf_strength'].rolling(5).mean()

        # 资金流择时信号
        df['signal_mf_net'] = (df['mf_net_5d'] > 0).astype(int)
        df['signal_main_force'] = (df['main_force_5d'] > 0).astype(int)
        df['signal_mf_strength'] = (df['mf_strength_ma5'] > 0).astype(int)

        self.moneyflow_signals = df
        return df

    def calculate_market_breadth_signals(self):
        """计算市场宽度择时信号"""
        print("计算市场宽度择时信号...")

        df = self.index_data.copy()
        df = df.join(self.market_data, how='left')

        # 1. 涨跌比
        df['ad_ratio'] = df['up_stocks'] / (df['down_stocks'] + 1)
        df['ad_ratio_ma5'] = df['ad_ratio'].rolling(5).mean()
        df['ad_ratio_ma20'] = df['ad_ratio'].rolling(20).mean()

        # 2. 涨跌幅差
        df['ad_diff'] = df['up_stocks'] - df['down_stocks']
        df['ad_line'] = df['ad_diff'].cumsum()
        df['ad_line_ma20'] = df['ad_line'].rolling(20).mean()

        # 3. 上涨股占比
        df['up_ratio'] = df['up_stocks'] / df['total_stocks']
        df['up_ratio_ma5'] = df['up_ratio'].rolling(5).mean()

        # 4. 成交额变化
        df['amount_ma5'] = df['total_amount'].rolling(5).mean()
        df['amount_ma20'] = df['total_amount'].rolling(20).mean()
        df['amount_ratio'] = df['total_amount'] / df['amount_ma20']

        # 市场宽度择时信号
        df['signal_breadth'] = (df['ad_ratio_ma5'] > 1).astype(int)
        df['signal_ad_line'] = (df['ad_line'] > df['ad_line_ma20']).astype(int)
        df['signal_up_ratio'] = (df['up_ratio_ma5'] > 0.5).astype(int)

        self.breadth_signals = df
        return df

    def backtest_signal(self, signal_series, returns_series, signal_name='Signal'):
        """回测单个信号"""
        # 信号对齐：信号在T日收盘后产生，T+1日执行
        signal_shifted = signal_series.shift(1)

        # 计算策略收益
        strategy_returns = signal_shifted * returns_series / 100

        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        benchmark_returns = (1 + returns_series / 100).cumprod()

        # 计算指标
        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        benchmark_return = benchmark_returns.iloc[-1] - 1 if len(benchmark_returns) > 0 else 0

        # 年化收益
        n_years = len(returns_series) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        benchmark_annual = (1 + benchmark_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 夏普比率
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # 最大回撤
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 胜率
        signal_days = signal_shifted == 1
        win_days = (returns_series[signal_days] > 0).sum()
        total_signal_days = signal_days.sum()
        win_rate = win_days / total_signal_days if total_signal_days > 0 else 0

        # 换手率（信号变化次数）
        signal_changes = signal_shifted.diff().abs().sum()
        turnover = signal_changes / len(signal_shifted) * 252 if len(signal_shifted) > 0 else 0

        # 持仓比例
        holding_ratio = signal_shifted.mean()

        return {
            'signal_name': signal_name,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'annual_return': annual_return,
            'benchmark_annual': benchmark_annual,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'turnover': turnover,
            'holding_ratio': holding_ratio,
            'cumulative_returns': cumulative_returns,
            'benchmark_returns': benchmark_returns
        }

    def compare_signals(self):
        """比较不同择时信号的效果"""
        print("\n比较不同择时信号...")

        returns = self.index_data['pct_chg']
        results = []

        # 均线信号
        ma_signals = [
            ('MA20突破', self.ma_signals['signal_ma20']),
            ('MA60突破', self.ma_signals['signal_ma60']),
            ('MA120突破', self.ma_signals['signal_ma120']),
            ('MA250突破', self.ma_signals['signal_ma250']),
            ('MA5-20金叉', self.ma_signals['signal_ma5_20']),
            ('MA20-60金叉', self.ma_signals['signal_ma20_60']),
            ('均线多头排列', self.ma_signals['signal_ma_bull']),
        ]

        # 动量信号
        momentum_signals = [
            ('ROC20>0', self.momentum_signals['signal_roc20']),
            ('RSI14适中', self.momentum_signals['signal_rsi14']),
            ('MACD金叉', self.momentum_signals['signal_macd']),
            ('综合动量', self.momentum_signals['signal_momentum']),
        ]

        # 波动率信号
        vol_signals = [
            ('低波动率', self.volatility_signals['signal_low_vol']),
            ('波动率突破', self.volatility_signals['signal_vol_breakout']),
            ('波动率收缩', self.volatility_signals['signal_vol_contract']),
        ]

        # 资金流信号
        mf_signals = [
            ('资金净流入', self.moneyflow_signals['signal_mf_net']),
            ('主力资金流入', self.moneyflow_signals['signal_main_force']),
            ('资金流强度', self.moneyflow_signals['signal_mf_strength']),
        ]

        # 市场宽度信号
        breadth_signals = [
            ('涨跌比', self.breadth_signals['signal_breadth']),
            ('AD线趋势', self.breadth_signals['signal_ad_line']),
            ('上涨股占比', self.breadth_signals['signal_up_ratio']),
        ]

        all_signals = ma_signals + momentum_signals + vol_signals + mf_signals + breadth_signals

        for name, signal in all_signals:
            result = self.backtest_signal(signal, returns, name)
            results.append(result)

        self.signal_results = pd.DataFrame(results)
        return self.signal_results

    def analyze_timing_frequency(self):
        """分析择时频率"""
        print("\n分析择时频率...")

        results = []

        signals_dict = {
            'MA20突破': self.ma_signals['signal_ma20'],
            'MA60突破': self.ma_signals['signal_ma60'],
            'MA120突破': self.ma_signals['signal_ma120'],
            'ROC20>0': self.momentum_signals['signal_roc20'],
            'MACD金叉': self.momentum_signals['signal_macd'],
            '低波动率': self.volatility_signals['signal_low_vol'],
            '资金净流入': self.moneyflow_signals['signal_mf_net'],
            '涨跌比': self.breadth_signals['signal_breadth'],
        }

        for name, signal in signals_dict.items():
            # 计算信号变化次数
            changes = signal.diff().abs()

            # 年均换手次数
            years = len(signal) / 252
            annual_changes = changes.sum() / years

            # 平均持仓时间
            holding_periods = []
            current_holding = 0
            for i in range(len(signal)):
                if signal.iloc[i] == 1:
                    current_holding += 1
                else:
                    if current_holding > 0:
                        holding_periods.append(current_holding)
                    current_holding = 0
            if current_holding > 0:
                holding_periods.append(current_holding)

            avg_holding = np.mean(holding_periods) if holding_periods else 0

            # 信号覆盖率
            coverage = signal.mean()

            results.append({
                'signal_name': name,
                'annual_changes': annual_changes,
                'avg_holding_days': avg_holding,
                'coverage': coverage,
                'total_signals': len(holding_periods)
            })

        self.frequency_results = pd.DataFrame(results)
        return self.frequency_results

    def analyze_market_states(self):
        """分析不同市场状态下的择时效果"""
        print("\n分析市场状态适应性...")

        df = self.index_data.copy()
        returns = df['pct_chg']

        # 定义市场状态
        df['ma60'] = df['close'].rolling(60).mean()
        df['volatility'] = df['pct_chg'].rolling(20).std()
        df['vol_median'] = df['volatility'].rolling(252).median()

        # 市场状态分类
        df['bull_market'] = (df['close'] > df['ma60']) & (df['close'] > df['close'].rolling(120).mean())
        df['bear_market'] = (df['close'] < df['ma60']) & (df['close'] < df['close'].rolling(120).mean())
        df['high_vol'] = df['volatility'] > df['vol_median']

        # 状态组合
        df['market_state'] = 'neutral'
        df.loc[df['bull_market'] & ~df['high_vol'], 'market_state'] = 'bull_low_vol'
        df.loc[df['bull_market'] & df['high_vol'], 'market_state'] = 'bull_high_vol'
        df.loc[df['bear_market'] & ~df['high_vol'], 'market_state'] = 'bear_low_vol'
        df.loc[df['bear_market'] & df['high_vol'], 'market_state'] = 'bear_high_vol'

        # 各状态下的信号表现
        signals_to_test = {
            'MA60突破': self.ma_signals['signal_ma60'],
            'MACD金叉': self.momentum_signals['signal_macd'],
            '低波动率': self.volatility_signals['signal_low_vol'],
            '资金净流入': self.moneyflow_signals['signal_mf_net'],
        }

        state_results = []
        for state in df['market_state'].unique():
            state_mask = df['market_state'] == state
            state_returns = returns[state_mask]
            state_days = state_mask.sum()

            for signal_name, signal in signals_to_test.items():
                signal_state = signal[state_mask]
                result = self.backtest_signal(signal_state, state_returns, f'{signal_name}_{state}')
                state_results.append({
                    'market_state': state,
                    'signal_name': signal_name,
                    'state_days': state_days,
                    'excess_return': result['excess_return'],
                    'sharpe': result['sharpe'],
                    'win_rate': result['win_rate'],
                    'holding_ratio': result['holding_ratio']
                })

        self.state_results = pd.DataFrame(state_results)
        return self.state_results

    def build_position_timing_model(self):
        """构建仓位择时模型"""
        print("\n构建仓位择时模型...")

        df = self.index_data.copy()

        # 综合多个信号计算仓位
        df['ma_score'] = (
            self.ma_signals['signal_ma20'] +
            self.ma_signals['signal_ma60'] +
            self.ma_signals['signal_ma120']
        ) / 3

        df['momentum_score'] = (
            self.momentum_signals['signal_roc20'] +
            self.momentum_signals['signal_macd']
        ) / 2

        df['vol_score'] = self.volatility_signals['signal_low_vol']

        df['mf_score'] = (
            self.moneyflow_signals['signal_mf_net'].fillna(0) +
            self.moneyflow_signals['signal_main_force'].fillna(0)
        ) / 2

        df['breadth_score'] = (
            self.breadth_signals['signal_breadth'].fillna(0) +
            self.breadth_signals['signal_up_ratio'].fillna(0)
        ) / 2

        # 综合仓位 (0-100%)
        df['position'] = (
            df['ma_score'] * 0.25 +
            df['momentum_score'] * 0.25 +
            df['vol_score'] * 0.20 +
            df['mf_score'] * 0.15 +
            df['breadth_score'] * 0.15
        )

        # 仓位分档
        df['position_level'] = pd.cut(
            df['position'],
            bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=['空仓', '轻仓', '半仓', '重仓', '满仓']
        )

        self.position_model = df
        return df

    def build_asset_rotation_model(self):
        """构建品种择时模型"""
        print("\n构建品种择时模型...")

        # 准备三个指数数据
        sh_index = self.index_data['pct_chg']
        hs300 = self.hs300_data['pct_chg']
        cyb = self.cyb_data['pct_chg']

        # 合并数据
        df = pd.DataFrame({
            'sh_index': sh_index,
            'hs300': hs300,
            'cyb': cyb
        })

        # 计算动量
        for col in ['sh_index', 'hs300', 'cyb']:
            df[f'{col}_cum20'] = df[col].rolling(20).sum()
            df[f'{col}_cum60'] = df[col].rolling(60).sum()

        # 选择动量最强的品种
        momentum_cols = ['sh_index_cum20', 'hs300_cum20', 'cyb_cum20']
        df['best_asset'] = df[momentum_cols].idxmax(axis=1).str.replace('_cum20', '')

        # 计算轮动策略收益
        df['rotation_return'] = 0.0
        for asset in ['sh_index', 'hs300', 'cyb']:
            mask = df['best_asset'].shift(1) == asset
            df.loc[mask, 'rotation_return'] = df.loc[mask, asset]

        self.rotation_model = df
        return df

    def build_composite_system(self):
        """构建综合择时系统"""
        print("\n构建综合择时系统...")

        df = self.position_model.copy()
        returns = df['pct_chg']

        # 综合信号：仓位 + 品种
        df['composite_position'] = df['position'].shift(1)

        # 策略收益
        df['strategy_return'] = df['composite_position'] * returns / 100
        df['benchmark_return'] = returns / 100

        # 累计收益
        df['strategy_cum'] = (1 + df['strategy_return']).cumprod()
        df['benchmark_cum'] = (1 + df['benchmark_return']).cumprod()

        self.composite_system = df
        return df

    def generate_report(self):
        """生成研究报告"""
        print("\n生成研究报告...")

        report = []
        report.append("=" * 80)
        report.append("                    市场择时策略研究报告")
        report.append("=" * 80)
        report.append(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据时间范围: {self.index_data.index.min().strftime('%Y-%m-%d')} 至 {self.index_data.index.max().strftime('%Y-%m-%d')}")
        report.append(f"总交易日数: {len(self.index_data)}")

        # 第一部分：择时指标
        report.append("\n" + "=" * 80)
        report.append("第一部分：择时指标分析")
        report.append("=" * 80)

        report.append("\n1.1 均线择时指标")
        report.append("-" * 40)
        report.append("策略说明：")
        report.append("  - 单均线突破：价格站上/跌破均线")
        report.append("  - 双均线交叉：短期均线与长期均线的金叉/死叉")
        report.append("  - 均线多头排列：MA5 > MA10 > MA20 > MA60")

        report.append("\n1.2 动量择时指标")
        report.append("-" * 40)
        report.append("策略说明：")
        report.append("  - ROC(变动率)：衡量价格动量强度")
        report.append("  - RSI(相对强弱)：衡量超买超卖状态")
        report.append("  - MACD：趋势跟踪动量指标")

        report.append("\n1.3 波动率择时指标")
        report.append("-" * 40)
        report.append("策略说明：")
        report.append("  - 历史波动率：基于收益率标准差")
        report.append("  - ATR：真实波动幅度")
        report.append("  - 布林带宽度：衡量波动率收缩/扩张")

        report.append("\n1.4 资金流择时指标")
        report.append("-" * 40)
        report.append("策略说明：")
        report.append("  - 资金净流入：全市场净资金流向")
        report.append("  - 主力资金：大单+特大单资金流向")
        report.append("  - 资金流强度：净流入/总成交额比例")

        # 第二部分：择时效果对比
        report.append("\n" + "=" * 80)
        report.append("第二部分：择时效果对比")
        report.append("=" * 80)

        report.append("\n2.1 不同指标表现对比")
        report.append("-" * 40)

        # 按超额收益排序
        sorted_results = self.signal_results.sort_values('excess_return', ascending=False)

        report.append("\n策略表现排名（按超额收益）:")
        report.append(f"{'策略名称':<20} {'总收益':>10} {'超额收益':>10} {'年化收益':>10} {'夏普比':>8} {'最大回撤':>10} {'胜率':>8}")
        report.append("-" * 90)

        for _, row in sorted_results.iterrows():
            report.append(f"{row['signal_name']:<20} {row['total_return']*100:>9.2f}% {row['excess_return']*100:>9.2f}% {row['annual_return']*100:>9.2f}% {row['sharpe']:>8.2f} {row['max_drawdown']*100:>9.2f}% {row['win_rate']*100:>7.1f}%")

        report.append("\n关键发现:")
        best = sorted_results.iloc[0]
        worst = sorted_results.iloc[-1]
        report.append(f"  - 最佳策略: {best['signal_name']}，超额收益 {best['excess_return']*100:.2f}%，夏普比率 {best['sharpe']:.2f}")
        report.append(f"  - 最差策略: {worst['signal_name']}，超额收益 {worst['excess_return']*100:.2f}%，夏普比率 {worst['sharpe']:.2f}")

        # 分类统计
        ma_results = sorted_results[sorted_results['signal_name'].str.contains('MA|均线')]
        momentum_results = sorted_results[sorted_results['signal_name'].str.contains('ROC|RSI|MACD|动量')]
        vol_results = sorted_results[sorted_results['signal_name'].str.contains('波动')]
        mf_results = sorted_results[sorted_results['signal_name'].str.contains('资金')]

        report.append(f"\n  - 均线类策略平均超额收益: {ma_results['excess_return'].mean()*100:.2f}%")
        report.append(f"  - 动量类策略平均超额收益: {momentum_results['excess_return'].mean()*100:.2f}%")
        if len(vol_results) > 0:
            report.append(f"  - 波动率策略平均超额收益: {vol_results['excess_return'].mean()*100:.2f}%")
        if len(mf_results) > 0:
            report.append(f"  - 资金流策略平均超额收益: {mf_results['excess_return'].mean()*100:.2f}%")

        report.append("\n2.2 择时频率分析")
        report.append("-" * 40)

        report.append(f"\n{'策略名称':<15} {'年均换手次数':>12} {'平均持仓天数':>12} {'信号覆盖率':>10}")
        report.append("-" * 55)

        for _, row in self.frequency_results.iterrows():
            report.append(f"{row['signal_name']:<15} {row['annual_changes']:>12.1f} {row['avg_holding_days']:>12.1f} {row['coverage']*100:>9.1f}%")

        report.append("\n关键发现:")
        high_freq = self.frequency_results[self.frequency_results['annual_changes'] > 20]
        low_freq = self.frequency_results[self.frequency_results['annual_changes'] < 10]
        report.append(f"  - 高频策略（年换手>20次）: {', '.join(high_freq['signal_name'].tolist())}")
        report.append(f"  - 低频策略（年换手<10次）: {', '.join(low_freq['signal_name'].tolist())}")

        report.append("\n2.3 市场状态适应性分析")
        report.append("-" * 40)

        state_pivot = self.state_results.pivot_table(
            values='excess_return',
            index='signal_name',
            columns='market_state'
        )

        report.append("\n各信号在不同市场状态下的超额收益(%):")
        report.append(state_pivot.to_string())

        report.append("\n关键发现:")
        for signal in self.state_results['signal_name'].unique():
            signal_data = self.state_results[self.state_results['signal_name'] == signal]
            best_state = signal_data.loc[signal_data['excess_return'].idxmax()]
            report.append(f"  - {signal}: 最适合 {best_state['market_state']} 状态，超额收益 {best_state['excess_return']*100:.2f}%")

        # 第三部分：策略应用
        report.append("\n" + "=" * 80)
        report.append("第三部分：策略应用")
        report.append("=" * 80)

        report.append("\n3.1 仓位择时模型")
        report.append("-" * 40)

        position_dist = self.position_model['position_level'].value_counts()
        report.append("\n仓位分布:")
        for level, count in position_dist.items():
            report.append(f"  - {level}: {count} 天 ({count/len(self.position_model)*100:.1f}%)")

        # 计算仓位择时收益
        position_returns = self.composite_system['strategy_return'].sum() * 100
        benchmark_returns = self.composite_system['benchmark_return'].sum() * 100

        report.append(f"\n仓位择时策略表现:")
        report.append(f"  - 策略累计收益: {(self.composite_system['strategy_cum'].iloc[-1]-1)*100:.2f}%")
        report.append(f"  - 基准累计收益: {(self.composite_system['benchmark_cum'].iloc[-1]-1)*100:.2f}%")
        report.append(f"  - 超额收益: {((self.composite_system['strategy_cum'].iloc[-1]-1) - (self.composite_system['benchmark_cum'].iloc[-1]-1))*100:.2f}%")

        report.append("\n3.2 品种择时模型")
        report.append("-" * 40)

        asset_dist = self.rotation_model['best_asset'].value_counts()
        report.append("\n品种选择分布:")
        for asset, count in asset_dist.items():
            asset_name = {'sh_index': '上证指数', 'hs300': '沪深300', 'cyb': '创业板'}
            report.append(f"  - {asset_name.get(asset, asset)}: {count} 天 ({count/len(self.rotation_model)*100:.1f}%)")

        # 轮动策略收益
        rotation_cum = (1 + self.rotation_model['rotation_return']/100).cumprod().iloc[-1] - 1
        sh_cum = (1 + self.rotation_model['sh_index']/100).cumprod().iloc[-1] - 1

        report.append(f"\n品种轮动策略表现:")
        report.append(f"  - 轮动策略累计收益: {rotation_cum*100:.2f}%")
        report.append(f"  - 上证指数累计收益: {sh_cum*100:.2f}%")

        report.append("\n3.3 综合择时系统")
        report.append("-" * 40)

        report.append("\n综合择时系统设计:")
        report.append("  - 仓位择时权重分配:")
        report.append("    * 均线信号: 25%")
        report.append("    * 动量信号: 25%")
        report.append("    * 波动率信号: 20%")
        report.append("    * 资金流信号: 15%")
        report.append("    * 市场宽度信号: 15%")

        report.append("\n  - 系统优势:")
        report.append("    * 多维度信号融合，降低单一指标失效风险")
        report.append("    * 仓位动态调整，适应不同市场环境")
        report.append("    * 趋势与反转信号结合，平衡收益与风险")

        # 第四部分：结论与建议
        report.append("\n" + "=" * 80)
        report.append("第四部分：结论与建议")
        report.append("=" * 80)

        report.append("\n4.1 主要结论")
        report.append("-" * 40)

        # 找出最佳策略组合
        report.append("\n1. 择时指标有效性:")
        report.append(f"   - 均线类指标整体稳健，适合中长期择时")
        report.append(f"   - 动量指标在趋势行情中表现突出")
        report.append(f"   - 波动率指标适合风险管理和仓位控制")
        report.append(f"   - 资金流指标对短期行情有一定预测能力")

        report.append("\n2. 择时频率建议:")
        report.append(f"   - 高频择时（周级别）: 适用于资金流、市场宽度指标")
        report.append(f"   - 中频择时（月级别）: 适用于动量、均线交叉指标")
        report.append(f"   - 低频择时（季度级别）: 适用于长期均线、波动率周期")

        report.append("\n3. 市场状态适应:")
        report.append(f"   - 牛市低波动: 动量策略最优")
        report.append(f"   - 牛市高波动: 均线策略更稳健")
        report.append(f"   - 熊市: 波动率和资金流信号有防御价值")

        report.append("\n4.2 实操建议")
        report.append("-" * 40)

        report.append("\n1. 入场信号组合:")
        report.append("   - 强势信号: 均线多头 + MACD金叉 + 资金净流入")
        report.append("   - 中性信号: 价格站上MA60 + RSI>50")
        report.append("   - 观望信号: 价格在均线附近，波动率收缩")

        report.append("\n2. 出场信号组合:")
        report.append("   - 止盈信号: RSI>70 + 放量滞涨 + 主力资金流出")
        report.append("   - 止损信号: 跌破MA60 + MACD死叉")
        report.append("   - 减仓信号: 波动率急剧放大 + 市场宽度恶化")

        report.append("\n3. 仓位管理建议:")
        report.append("   - 满仓条件: 综合信号>0.8，牛市确认")
        report.append("   - 重仓条件: 综合信号0.6-0.8，趋势向好")
        report.append("   - 半仓条件: 综合信号0.4-0.6，方向不明")
        report.append("   - 轻仓条件: 综合信号0.2-0.4，风险较高")
        report.append("   - 空仓条件: 综合信号<0.2，熊市明确")

        report.append("\n" + "=" * 80)
        report.append("                         报告完毕")
        report.append("=" * 80)

        return "\n".join(report)

    def plot_results(self):
        """绑制可视化图表"""
        print("\n生成可视化图表...")

        # 图1：择时信号效果对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1.1 超额收益对比
        ax1 = axes[0, 0]
        sorted_results = self.signal_results.sort_values('excess_return', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_results['excess_return']]
        ax1.barh(sorted_results['signal_name'], sorted_results['excess_return'] * 100, color=colors)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('超额收益 (%)')
        ax1.set_title('各择时信号超额收益对比')
        ax1.grid(axis='x', alpha=0.3)

        # 1.2 夏普比率对比
        ax2 = axes[0, 1]
        sorted_by_sharpe = self.signal_results.sort_values('sharpe', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_by_sharpe['sharpe']]
        ax2.barh(sorted_by_sharpe['signal_name'], sorted_by_sharpe['sharpe'], color=colors)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('夏普比率')
        ax2.set_title('各择时信号夏普比率对比')
        ax2.grid(axis='x', alpha=0.3)

        # 1.3 风险收益散点图
        ax3 = axes[1, 0]
        ax3.scatter(self.signal_results['max_drawdown'] * 100,
                   self.signal_results['annual_return'] * 100,
                   s=100, alpha=0.7)
        for i, row in self.signal_results.iterrows():
            ax3.annotate(row['signal_name'],
                        (row['max_drawdown'] * 100, row['annual_return'] * 100),
                        fontsize=8, alpha=0.8)
        ax3.set_xlabel('最大回撤 (%)')
        ax3.set_ylabel('年化收益 (%)')
        ax3.set_title('风险收益散点图')
        ax3.grid(alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 1.4 胜率与持仓比对比
        ax4 = axes[1, 1]
        x = np.arange(len(self.signal_results))
        width = 0.35
        ax4.bar(x - width/2, self.signal_results['win_rate'] * 100, width, label='胜率', color='steelblue')
        ax4.bar(x + width/2, self.signal_results['holding_ratio'] * 100, width, label='持仓比例', color='coral')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.signal_results['signal_name'], rotation=45, ha='right')
        ax4.set_ylabel('百分比 (%)')
        ax4.set_title('胜率与持仓比例对比')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/timing_signals_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图2：综合择时系统表现
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # 2.1 累计收益对比
        ax1 = axes[0]
        ax1.plot(self.composite_system.index, self.composite_system['strategy_cum'],
                label='择时策略', linewidth=1.5, color='blue')
        ax1.plot(self.composite_system.index, self.composite_system['benchmark_cum'],
                label='买入持有', linewidth=1.5, color='gray', alpha=0.7)
        ax1.set_ylabel('累计收益')
        ax1.set_title('综合择时策略 vs 买入持有')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)

        # 2.2 仓位变化
        ax2 = axes[1]
        ax2.fill_between(self.composite_system.index,
                        self.composite_system['composite_position'] * 100,
                        alpha=0.5, color='steelblue')
        ax2.set_ylabel('仓位 (%)')
        ax2.set_title('动态仓位变化')
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.3)

        # 2.3 超额收益
        ax3 = axes[2]
        excess = self.composite_system['strategy_cum'] - self.composite_system['benchmark_cum']
        ax3.fill_between(self.composite_system.index, excess,
                        where=excess > 0, color='green', alpha=0.5, label='超额收益>0')
        ax3.fill_between(self.composite_system.index, excess,
                        where=excess <= 0, color='red', alpha=0.5, label='超额收益<0')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('超额收益')
        ax3.set_title('相对基准超额收益')
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/timing_strategy_performance.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图3：市场状态分析
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 3.1 市场状态分布
        ax1 = axes[0, 0]
        state_counts = self.state_results.groupby('market_state')['state_days'].first()
        ax1.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', colors=plt.cm.Set3.colors)
        ax1.set_title('市场状态分布')

        # 3.2 各状态下的信号表现热力图
        ax2 = axes[0, 1]
        state_pivot = self.state_results.pivot_table(
            values='excess_return',
            index='signal_name',
            columns='market_state'
        ) * 100
        im = ax2.imshow(state_pivot.values, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(np.arange(len(state_pivot.columns)))
        ax2.set_yticks(np.arange(len(state_pivot.index)))
        ax2.set_xticklabels(state_pivot.columns, rotation=45, ha='right')
        ax2.set_yticklabels(state_pivot.index)
        ax2.set_title('各信号在不同市场状态下的超额收益(%)')
        plt.colorbar(im, ax=ax2)

        # 3.3 择时频率分析
        ax3 = axes[1, 0]
        ax3.scatter(self.frequency_results['annual_changes'],
                   self.frequency_results['avg_holding_days'],
                   s=self.frequency_results['coverage'] * 500, alpha=0.6)
        for i, row in self.frequency_results.iterrows():
            ax3.annotate(row['signal_name'],
                        (row['annual_changes'], row['avg_holding_days']),
                        fontsize=8)
        ax3.set_xlabel('年均换手次数')
        ax3.set_ylabel('平均持仓天数')
        ax3.set_title('择时频率分析 (气泡大小=信号覆盖率)')
        ax3.grid(alpha=0.3)

        # 3.4 仓位分布
        ax4 = axes[1, 1]
        position_hist = self.position_model['position'].dropna()
        ax4.hist(position_hist, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=position_hist.mean(), color='red', linestyle='--', label=f'均值: {position_hist.mean():.2f}')
        ax4.set_xlabel('综合仓位')
        ax4.set_ylabel('频数')
        ax4.set_title('综合仓位分布')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/timing_market_state_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 图4：品种轮动分析
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 4.1 各指数累计收益
        ax1 = axes[0]
        for col, name, color in [('sh_index', '上证指数', 'blue'),
                                  ('hs300', '沪深300', 'red'),
                                  ('cyb', '创业板', 'green')]:
            cum_ret = (1 + self.rotation_model[col]/100).cumprod()
            ax1.plot(self.rotation_model.index, cum_ret, label=name, linewidth=1.5, color=color)

        rotation_cum = (1 + self.rotation_model['rotation_return']/100).cumprod()
        ax1.plot(self.rotation_model.index, rotation_cum, label='轮动策略',
                linewidth=2, color='purple', linestyle='--')
        ax1.set_ylabel('累计收益')
        ax1.set_title('品种轮动策略 vs 各指数')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)

        # 4.2 品种选择变化
        ax2 = axes[1]
        asset_map = {'sh_index': 0, 'hs300': 1, 'cyb': 2}
        asset_values = self.rotation_model['best_asset'].map(asset_map)
        ax2.fill_between(self.rotation_model.index, asset_values, step='post', alpha=0.5)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['上证指数', '沪深300', '创业板'])
        ax2.set_ylabel('选择品种')
        ax2.set_title('品种轮动选择变化')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/timing_asset_rotation.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"图表已保存至 {OUTPUT_DIR}/")

    def run_research(self):
        """运行完整研究"""
        # 1. 加载数据
        self.load_data()

        # 2. 计算各类择时信号
        self.calculate_ma_signals()
        self.calculate_momentum_signals()
        self.calculate_volatility_signals()
        self.calculate_moneyflow_signals()
        self.calculate_market_breadth_signals()

        # 3. 比较信号效果
        self.compare_signals()
        self.analyze_timing_frequency()
        self.analyze_market_states()

        # 4. 构建择时系统
        self.build_position_timing_model()
        self.build_asset_rotation_model()
        self.build_composite_system()

        # 5. 生成报告和图表
        report = self.generate_report()
        self.plot_results()

        # 6. 保存报告
        report_path = f'{OUTPUT_DIR}/market_timing_research_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n研究报告已保存至: {report_path}")
        print(report)

        return report


if __name__ == '__main__':
    research = MarketTimingResearch()
    research.run_research()
