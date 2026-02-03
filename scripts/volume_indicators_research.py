#!/usr/bin/env python3
"""
成交量技术指标研究
Volume Technical Indicators Research

包含：
1. 量能指标计算：OBV, VR, PVT, MFI
2. 量价关系分析：量增价涨/量缩价跌、量价背离、量能突破
3. 策略回测：OBV策略、量能确认策略、量价配合策略
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research/'


class VolumeIndicators:
    """成交量技术指标计算类"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_stock_data(self, ts_code, start_date='20200101', end_date='20260130'):
        """获取股票日线数据"""
        query = f"""
        SELECT
            ts_code,
            trade_date,
            open,
            high,
            low,
            close,
            pre_close,
            change,
            pct_chg,
            vol,
            amount
        FROM daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def get_multiple_stocks(self, start_date='20230101', end_date='20260130', limit=100):
        """获取多只股票数据用于统计分析"""
        # 获取活跃股票列表
        query = f"""
        SELECT DISTINCT ts_code
        FROM daily
        WHERE trade_date >= '{start_date}'
        GROUP BY ts_code
        HAVING COUNT(*) > 200
        ORDER BY SUM(amount) DESC
        LIMIT {limit}
        """
        stocks = self.conn.execute(query).fetchdf()['ts_code'].tolist()
        return stocks

    @staticmethod
    def calc_obv(df):
        """
        计算OBV (On-Balance Volume 能量潮)

        OBV计算规则：
        - 当日收盘价 > 前日收盘价时，OBV = 前日OBV + 当日成交量
        - 当日收盘价 < 前日收盘价时，OBV = 前日OBV - 当日成交量
        - 当日收盘价 = 前日收盘价时，OBV = 前日OBV
        """
        df = df.copy()
        df['price_change'] = df['close'].diff()
        df['obv_change'] = np.where(df['price_change'] > 0, df['vol'],
                                     np.where(df['price_change'] < 0, -df['vol'], 0))
        df['OBV'] = df['obv_change'].cumsum()

        # 计算OBV均线
        df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()
        df['OBV_MA10'] = df['OBV'].rolling(window=10).mean()
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        return df

    @staticmethod
    def calc_vr(df, period=26):
        """
        计算VR (Volume Ratio 成交量比率)

        VR = (N日内上涨日成交量总和 + 1/2 * N日内平盘日成交量总和) /
             (N日内下跌日成交量总和 + 1/2 * N日内平盘日成交量总和) * 100

        参数说明：
        - period: 计算周期，默认26日

        VR指标判断：
        - VR < 40: 低价区，安全区，可以买入
        - 40 < VR < 70: 低价安全区，可以买入
        - 70 < VR < 150: 获利卖出区，可以持有
        - 150 < VR < 250: 警戒区，应该减仓
        - VR > 250: 高价区，危险区，应该卖出
        """
        df = df.copy()
        df['price_change'] = df['close'].diff()

        # 上涨日成交量
        df['up_vol'] = np.where(df['price_change'] > 0, df['vol'], 0)
        # 下跌日成交量
        df['down_vol'] = np.where(df['price_change'] < 0, df['vol'], 0)
        # 平盘日成交量
        df['flat_vol'] = np.where(df['price_change'] == 0, df['vol'], 0)

        # 计算VR
        sum_up = df['up_vol'].rolling(window=period).sum()
        sum_down = df['down_vol'].rolling(window=period).sum()
        sum_flat = df['flat_vol'].rolling(window=period).sum()

        df['VR'] = (sum_up + 0.5 * sum_flat) / (sum_down + 0.5 * sum_flat) * 100
        df['VR'] = df['VR'].replace([np.inf, -np.inf], np.nan)

        # VR均线
        df['VR_MA6'] = df['VR'].rolling(window=6).mean()

        return df

    @staticmethod
    def calc_pvt(df):
        """
        计算PVT (Price-Volume Trend 量价趋势)

        PVT = 前日PVT + (当日收盘价 - 前日收盘价) / 前日收盘价 * 当日成交量

        PVT与OBV的区别：
        - OBV只考虑价格涨跌，加减全部成交量
        - PVT考虑价格变动幅度，按比例加减成交量
        """
        df = df.copy()
        df['pct_chg_calc'] = df['close'].pct_change()
        df['pvt_change'] = df['pct_chg_calc'] * df['vol']
        df['PVT'] = df['pvt_change'].cumsum()

        # PVT均线
        df['PVT_MA5'] = df['PVT'].rolling(window=5).mean()
        df['PVT_MA10'] = df['PVT'].rolling(window=10).mean()

        return df

    @staticmethod
    def calc_mfi(df, period=14):
        """
        计算MFI (Money Flow Index 资金流量指标)

        典型价格 = (最高价 + 最低价 + 收盘价) / 3
        资金流量 = 典型价格 * 成交量

        正资金流量：当日典型价格 > 前日典型价格时的资金流量
        负资金流量：当日典型价格 < 前日典型价格时的资金流量

        资金流量比率 = N日内正资金流量总和 / N日内负资金流量总和
        MFI = 100 - 100 / (1 + 资金流量比率)

        MFI类似于RSI，但加入了成交量因素

        MFI判断：
        - MFI > 80: 超买区域
        - MFI < 20: 超卖区域
        """
        df = df.copy()

        # 典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # 资金流量
        df['money_flow'] = df['typical_price'] * df['vol']

        # 判断正负资金流量
        df['tp_change'] = df['typical_price'].diff()
        df['positive_mf'] = np.where(df['tp_change'] > 0, df['money_flow'], 0)
        df['negative_mf'] = np.where(df['tp_change'] < 0, df['money_flow'], 0)

        # 计算MFI
        sum_positive = df['positive_mf'].rolling(window=period).sum()
        sum_negative = df['negative_mf'].rolling(window=period).sum()

        money_ratio = sum_positive / sum_negative
        df['MFI'] = 100 - 100 / (1 + money_ratio)
        df['MFI'] = df['MFI'].replace([np.inf, -np.inf], np.nan)

        return df

    def calc_all_indicators(self, df):
        """计算所有成交量指标"""
        df = self.calc_obv(df)
        df = self.calc_vr(df)
        df = self.calc_pvt(df)
        df = self.calc_mfi(df)
        return df


class VolumePriceAnalysis:
    """量价关系分析类"""

    @staticmethod
    def detect_volume_price_pattern(df, vol_threshold=1.5, price_threshold=0.02):
        """
        检测量价关系模式

        参数：
        - vol_threshold: 成交量变化阈值（相对于20日均量）
        - price_threshold: 价格变化阈值

        返回模式：
        - 量增价涨 (Volume Up, Price Up)
        - 量增价跌 (Volume Up, Price Down)
        - 量缩价涨 (Volume Down, Price Up)
        - 量缩价跌 (Volume Down, Price Down)
        """
        df = df.copy()

        # 计算成交量均值
        df['vol_ma20'] = df['vol'].rolling(window=20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        # 判断量增量缩
        df['vol_up'] = df['vol_ratio'] > vol_threshold
        df['vol_down'] = df['vol_ratio'] < (1 / vol_threshold)

        # 判断价涨价跌
        df['price_up'] = df['pct_chg'] > price_threshold * 100
        df['price_down'] = df['pct_chg'] < -price_threshold * 100

        # 量价模式
        conditions = [
            (df['vol_up']) & (df['price_up']),       # 量增价涨
            (df['vol_up']) & (df['price_down']),     # 量增价跌
            (df['vol_down']) & (df['price_up']),     # 量缩价涨
            (df['vol_down']) & (df['price_down']),   # 量缩价跌
        ]
        choices = ['量增价涨', '量增价跌', '量缩价涨', '量缩价跌']
        df['volume_price_pattern'] = np.select(conditions, choices, default='正常')

        return df

    @staticmethod
    def detect_divergence(df, lookback=20):
        """
        检测量价背离

        顶背离：价格创新高，但成交量未创新高
        底背离：价格创新低，但成交量未创新低
        """
        df = df.copy()

        # 计算滚动最高/最低
        df['price_high_n'] = df['close'].rolling(window=lookback).max()
        df['price_low_n'] = df['close'].rolling(window=lookback).min()
        df['vol_high_n'] = df['vol'].rolling(window=lookback).max()
        df['vol_low_n'] = df['vol'].rolling(window=lookback).min()

        # 顶背离：当前价格 = N日最高价，但成交量 < N日最高成交量的80%
        df['top_divergence'] = (df['close'] == df['price_high_n']) & \
                                (df['vol'] < df['vol_high_n'] * 0.8)

        # 底背离：当前价格 = N日最低价，但成交量 > N日最低成交量的80%
        df['bottom_divergence'] = (df['close'] == df['price_low_n']) & \
                                   (df['vol'] > df['vol_low_n'] * 1.2)

        return df

    @staticmethod
    def detect_volume_breakout(df, vol_multiplier=2.0, lookback=20):
        """
        检测量能突破

        量能突破定义：
        - 成交量突破N日均量的M倍
        - 同时价格突破N日高点
        """
        df = df.copy()

        # 计算均量和最高价
        df['vol_ma'] = df['vol'].rolling(window=lookback).mean()
        df['price_high'] = df['close'].rolling(window=lookback).max()

        # 量能突破
        df['vol_breakout'] = df['vol'] > df['vol_ma'] * vol_multiplier

        # 价格突破
        df['price_breakout'] = df['close'] > df['price_high'].shift(1)

        # 量价同时突破
        df['volume_price_breakout'] = df['vol_breakout'] & df['price_breakout']

        return df


class VolumeBacktest:
    """成交量策略回测类"""

    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital

    def obv_strategy(self, df, short_period=5, long_period=20):
        """
        OBV策略

        买入信号：OBV短期均线上穿OBV长期均线
        卖出信号：OBV短期均线下穿OBV长期均线
        """
        df = df.copy()

        # 确保已计算OBV
        if 'OBV' not in df.columns:
            df = VolumeIndicators.calc_obv(df)

        # 计算均线
        df['obv_short'] = df['OBV'].rolling(window=short_period).mean()
        df['obv_long'] = df['OBV'].rolling(window=long_period).mean()

        # 生成信号
        df['signal'] = 0
        df.loc[df['obv_short'] > df['obv_long'], 'signal'] = 1
        df.loc[df['obv_short'] <= df['obv_long'], 'signal'] = -1

        # 交易信号（信号变化时）
        df['trade_signal'] = df['signal'].diff()

        return self._backtest(df)

    def volume_confirmation_strategy(self, df, vol_threshold=1.5, price_threshold=0.03):
        """
        量能确认策略

        买入信号：量增价涨（放量上涨）
        卖出信号：量增价跌（放量下跌）或连续缩量
        """
        df = df.copy()

        # 计算成交量均值
        df['vol_ma20'] = df['vol'].rolling(window=20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        # 生成信号
        df['signal'] = 0

        # 买入：放量上涨
        df.loc[(df['vol_ratio'] > vol_threshold) &
               (df['pct_chg'] > price_threshold * 100), 'signal'] = 1

        # 卖出：放量下跌
        df.loc[(df['vol_ratio'] > vol_threshold) &
               (df['pct_chg'] < -price_threshold * 100), 'signal'] = -1

        # 持仓状态
        df['position'] = 0
        position = 0
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:
                position = 1
            elif df['signal'].iloc[i] == -1:
                position = 0
            df.loc[df.index[i], 'position'] = position

        df['trade_signal'] = df['position'].diff()

        return self._backtest_with_position(df)

    def volume_price_strategy(self, df, mfi_oversold=20, mfi_overbought=80, vr_low=40, vr_high=250):
        """
        量价配合策略

        结合MFI和VR指标：
        买入信号：MFI < 20 (超卖) 且 VR < 70 (安全区)
        卖出信号：MFI > 80 (超买) 或 VR > 250 (危险区)
        """
        df = df.copy()

        # 确保已计算指标
        if 'MFI' not in df.columns:
            df = VolumeIndicators.calc_mfi(df)
        if 'VR' not in df.columns:
            df = VolumeIndicators.calc_vr(df)

        # 生成信号
        df['signal'] = 0

        # 买入信号
        df.loc[(df['MFI'] < mfi_oversold) & (df['VR'] < 70), 'signal'] = 1

        # 卖出信号
        df.loc[(df['MFI'] > mfi_overbought) | (df['VR'] > vr_high), 'signal'] = -1

        # 持仓状态
        df['position'] = 0
        position = 0
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1 and position == 0:
                position = 1
            elif df['signal'].iloc[i] == -1 and position == 1:
                position = 0
            df.loc[df.index[i], 'position'] = position

        df['trade_signal'] = df['position'].diff()

        return self._backtest_with_position(df)

    def _backtest(self, df):
        """执行回测"""
        df = df.copy()
        df = df.dropna()

        if len(df) < 2:
            return None

        # 计算收益
        df['strategy_return'] = df['signal'].shift(1) * df['pct_chg'] / 100
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['buy_hold_return'] = (1 + df['pct_chg'] / 100).cumprod()

        # 计算最大回撤
        df['rolling_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = df['cumulative_return'] / df['rolling_max'] - 1

        results = {
            'total_return': df['cumulative_return'].iloc[-1] - 1,
            'buy_hold_return': df['buy_hold_return'].iloc[-1] - 1,
            'max_drawdown': df['drawdown'].min(),
            'sharpe_ratio': self._calc_sharpe(df['strategy_return']),
            'win_rate': self._calc_win_rate(df),
            'trade_count': (df['trade_signal'].abs() > 0).sum(),
            'df': df
        }

        return results

    def _backtest_with_position(self, df):
        """基于持仓的回测"""
        df = df.copy()
        df = df.dropna()

        if len(df) < 2:
            return None

        # 计算收益
        df['strategy_return'] = df['position'].shift(1) * df['pct_chg'] / 100
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['buy_hold_return'] = (1 + df['pct_chg'] / 100).cumprod()

        # 计算最大回撤
        df['rolling_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = df['cumulative_return'] / df['rolling_max'] - 1

        results = {
            'total_return': df['cumulative_return'].iloc[-1] - 1,
            'buy_hold_return': df['buy_hold_return'].iloc[-1] - 1,
            'max_drawdown': df['drawdown'].min(),
            'sharpe_ratio': self._calc_sharpe(df['strategy_return']),
            'win_rate': self._calc_win_rate_position(df),
            'trade_count': (df['trade_signal'].abs() > 0).sum(),
            'df': df
        }

        return results

    def _calc_sharpe(self, returns, risk_free_rate=0.03):
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calc_win_rate(self, df):
        """计算胜率"""
        trades = df[df['trade_signal'] != 0]
        if len(trades) == 0:
            return 0
        wins = (trades['strategy_return'] > 0).sum()
        return wins / len(trades)

    def _calc_win_rate_position(self, df):
        """计算胜率（基于持仓）"""
        positive_days = (df['strategy_return'] > 0).sum()
        total_holding_days = (df['position'].shift(1) == 1).sum()
        if total_holding_days == 0:
            return 0
        return positive_days / total_holding_days


def generate_report():
    """生成研究报告"""
    print("=" * 60)
    print("成交量技术指标研究报告")
    print("=" * 60)

    # 初始化
    vi = VolumeIndicators()
    vpa = VolumePriceAnalysis()
    bt = VolumeBacktest()

    # 选取几只代表性股票进行分析
    test_stocks = ['000001.SZ', '600519.SH', '000858.SZ', '601318.SH', '002415.SZ']

    report_content = []
    report_content.append("# 成交量技术指标研究报告\n")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # ========== 第一部分：量能指标计算示例 ==========
    report_content.append("## 一、量能指标计算\n\n")

    print("\n1. 计算量能指标...")

    # 以平安银行为例
    df = vi.get_stock_data('000001.SZ', '20230101', '20260130')
    df = vi.calc_all_indicators(df)

    report_content.append("### 1.1 OBV (能量潮)\n\n")
    report_content.append("**计算公式：**\n")
    report_content.append("- 当日收盘价 > 前日收盘价: OBV = 前日OBV + 当日成交量\n")
    report_content.append("- 当日收盘价 < 前日收盘价: OBV = 前日OBV - 当日成交量\n")
    report_content.append("- 当日收盘价 = 前日收盘价: OBV = 前日OBV\n\n")
    report_content.append("**指标说明：**\n")
    report_content.append("OBV是通过累积成交量来衡量买卖压力的指标。当价格上涨时累加成交量，下跌时累减。\n\n")

    # OBV统计
    obv_latest = df['OBV'].iloc[-1]
    obv_ma20 = df['OBV_MA20'].iloc[-1]
    report_content.append(f"**000001.SZ 最新OBV数据:**\n")
    report_content.append(f"- 当前OBV: {obv_latest:,.0f}\n")
    report_content.append(f"- OBV 20日均线: {obv_ma20:,.0f}\n")
    report_content.append(f"- OBV趋势: {'多头' if obv_latest > obv_ma20 else '空头'}\n\n")

    report_content.append("### 1.2 VR (成交量比率)\n\n")
    report_content.append("**计算公式：**\n")
    report_content.append("VR = (N日上涨日成交量 + 0.5×平盘日成交量) / (N日下跌日成交量 + 0.5×平盘日成交量) × 100\n\n")
    report_content.append("**判断标准：**\n")
    report_content.append("- VR < 40: 低价区，可买入\n")
    report_content.append("- 40 < VR < 150: 安全区\n")
    report_content.append("- 150 < VR < 250: 警戒区\n")
    report_content.append("- VR > 250: 危险区，应卖出\n\n")

    vr_latest = df['VR'].iloc[-1]
    report_content.append(f"**000001.SZ 最新VR数据:**\n")
    report_content.append(f"- 当前VR(26日): {vr_latest:.2f}\n")
    if vr_latest < 40:
        vr_status = "低价区（买入区）"
    elif vr_latest < 150:
        vr_status = "安全区"
    elif vr_latest < 250:
        vr_status = "警戒区"
    else:
        vr_status = "危险区"
    report_content.append(f"- 区域判断: {vr_status}\n\n")

    report_content.append("### 1.3 PVT (量价趋势)\n\n")
    report_content.append("**计算公式：**\n")
    report_content.append("PVT = 前日PVT + (当日收盘-前日收盘)/前日收盘 × 当日成交量\n\n")
    report_content.append("**与OBV区别：**\n")
    report_content.append("PVT考虑价格变动幅度，按比例加减成交量，而OBV只看涨跌方向。\n\n")

    pvt_latest = df['PVT'].iloc[-1]
    pvt_ma10 = df['PVT_MA10'].iloc[-1]
    report_content.append(f"**000001.SZ 最新PVT数据:**\n")
    report_content.append(f"- 当前PVT: {pvt_latest:,.0f}\n")
    report_content.append(f"- PVT 10日均线: {pvt_ma10:,.0f}\n\n")

    report_content.append("### 1.4 MFI (资金流量指标)\n\n")
    report_content.append("**计算公式：**\n")
    report_content.append("1. 典型价格 = (最高+最低+收盘) / 3\n")
    report_content.append("2. 资金流量 = 典型价格 × 成交量\n")
    report_content.append("3. MFI = 100 - 100/(1 + 正资金流/负资金流)\n\n")
    report_content.append("**判断标准：**\n")
    report_content.append("- MFI > 80: 超买区域\n")
    report_content.append("- MFI < 20: 超卖区域\n\n")

    mfi_latest = df['MFI'].iloc[-1]
    report_content.append(f"**000001.SZ 最新MFI数据:**\n")
    report_content.append(f"- 当前MFI(14日): {mfi_latest:.2f}\n")
    if mfi_latest > 80:
        mfi_status = "超买"
    elif mfi_latest < 20:
        mfi_status = "超卖"
    else:
        mfi_status = "正常"
    report_content.append(f"- 状态判断: {mfi_status}\n\n")

    # ========== 第二部分：量价关系分析 ==========
    report_content.append("## 二、量价关系分析\n\n")

    print("2. 分析量价关系...")

    # 量价模式检测
    df = vpa.detect_volume_price_pattern(df)
    df = vpa.detect_divergence(df)
    df = vpa.detect_volume_breakout(df)

    report_content.append("### 2.1 量价模式统计\n\n")
    pattern_counts = df['volume_price_pattern'].value_counts()
    report_content.append("| 模式 | 出现次数 | 占比 |\n")
    report_content.append("|------|----------|------|\n")
    total = len(df)
    for pattern, count in pattern_counts.items():
        report_content.append(f"| {pattern} | {count} | {count/total*100:.1f}% |\n")
    report_content.append("\n")

    report_content.append("### 2.2 量价背离检测\n\n")
    top_div_count = df['top_divergence'].sum()
    bottom_div_count = df['bottom_divergence'].sum()
    report_content.append(f"- 顶背离次数: {top_div_count}\n")
    report_content.append(f"- 底背离次数: {bottom_div_count}\n\n")

    # 找出最近的背离点
    recent_divs = df[df['top_divergence'] | df['bottom_divergence']].tail(5)
    if len(recent_divs) > 0:
        report_content.append("**最近5次背离:**\n\n")
        report_content.append("| 日期 | 类型 | 收盘价 | 成交量 |\n")
        report_content.append("|------|------|--------|--------|\n")
        for _, row in recent_divs.iterrows():
            div_type = "顶背离" if row['top_divergence'] else "底背离"
            report_content.append(f"| {row['trade_date'].strftime('%Y-%m-%d')} | {div_type} | {row['close']:.2f} | {row['vol']:,.0f} |\n")
        report_content.append("\n")

    report_content.append("### 2.3 量能突破\n\n")
    breakout_count = df['volume_price_breakout'].sum()
    report_content.append(f"- 量价同时突破次数: {breakout_count}\n\n")

    # 最近的突破点
    recent_breakouts = df[df['volume_price_breakout']].tail(5)
    if len(recent_breakouts) > 0:
        report_content.append("**最近5次量能突破:**\n\n")
        report_content.append("| 日期 | 收盘价 | 涨幅 | 成交量倍数 |\n")
        report_content.append("|------|--------|------|------------|\n")
        for _, row in recent_breakouts.iterrows():
            vol_mult = row['vol'] / row['vol_ma'] if row['vol_ma'] > 0 else 0
            report_content.append(f"| {row['trade_date'].strftime('%Y-%m-%d')} | {row['close']:.2f} | {row['pct_chg']:.2f}% | {vol_mult:.2f}x |\n")
        report_content.append("\n")

    # ========== 第三部分：策略回测 ==========
    report_content.append("## 三、策略回测\n\n")

    print("3. 执行策略回测...")

    # 多股票回测汇总
    all_obv_results = []
    all_vol_confirm_results = []
    all_vol_price_results = []

    for stock in test_stocks:
        try:
            df_stock = vi.get_stock_data(stock, '20230101', '20260130')
            df_stock = vi.calc_all_indicators(df_stock)

            # OBV策略
            obv_result = bt.obv_strategy(df_stock)
            if obv_result:
                all_obv_results.append({
                    'stock': stock,
                    'total_return': obv_result['total_return'],
                    'buy_hold_return': obv_result['buy_hold_return'],
                    'max_drawdown': obv_result['max_drawdown'],
                    'sharpe_ratio': obv_result['sharpe_ratio'],
                    'trade_count': obv_result['trade_count']
                })

            # 量能确认策略
            vol_confirm_result = bt.volume_confirmation_strategy(df_stock)
            if vol_confirm_result:
                all_vol_confirm_results.append({
                    'stock': stock,
                    'total_return': vol_confirm_result['total_return'],
                    'buy_hold_return': vol_confirm_result['buy_hold_return'],
                    'max_drawdown': vol_confirm_result['max_drawdown'],
                    'sharpe_ratio': vol_confirm_result['sharpe_ratio'],
                    'trade_count': vol_confirm_result['trade_count']
                })

            # 量价配合策略
            vol_price_result = bt.volume_price_strategy(df_stock)
            if vol_price_result:
                all_vol_price_results.append({
                    'stock': stock,
                    'total_return': vol_price_result['total_return'],
                    'buy_hold_return': vol_price_result['buy_hold_return'],
                    'max_drawdown': vol_price_result['max_drawdown'],
                    'sharpe_ratio': vol_price_result['sharpe_ratio'],
                    'trade_count': vol_price_result['trade_count']
                })
        except Exception as e:
            print(f"  处理 {stock} 出错: {e}")

    # OBV策略结果
    report_content.append("### 3.1 OBV策略回测\n\n")
    report_content.append("**策略逻辑：**\n")
    report_content.append("- 买入信号: OBV 5日均线上穿20日均线\n")
    report_content.append("- 卖出信号: OBV 5日均线下穿20日均线\n\n")

    if all_obv_results:
        report_content.append("**回测结果：**\n\n")
        report_content.append("| 股票代码 | 策略收益 | 持有收益 | 最大回撤 | 夏普比率 | 交易次数 |\n")
        report_content.append("|----------|----------|----------|----------|----------|----------|\n")
        for r in all_obv_results:
            report_content.append(f"| {r['stock']} | {r['total_return']*100:.2f}% | {r['buy_hold_return']*100:.2f}% | {r['max_drawdown']*100:.2f}% | {r['sharpe_ratio']:.2f} | {r['trade_count']} |\n")

        # 计算平均
        avg_return = np.mean([r['total_return'] for r in all_obv_results])
        avg_bh_return = np.mean([r['buy_hold_return'] for r in all_obv_results])
        avg_mdd = np.mean([r['max_drawdown'] for r in all_obv_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_obv_results])
        report_content.append(f"| **平均** | **{avg_return*100:.2f}%** | **{avg_bh_return*100:.2f}%** | **{avg_mdd*100:.2f}%** | **{avg_sharpe:.2f}** | - |\n")
        report_content.append("\n")

    # 量能确认策略结果
    report_content.append("### 3.2 量能确认策略回测\n\n")
    report_content.append("**策略逻辑：**\n")
    report_content.append("- 买入信号: 放量上涨（成交量>1.5倍20日均量，涨幅>3%）\n")
    report_content.append("- 卖出信号: 放量下跌（成交量>1.5倍20日均量，跌幅>3%）\n\n")

    if all_vol_confirm_results:
        report_content.append("**回测结果：**\n\n")
        report_content.append("| 股票代码 | 策略收益 | 持有收益 | 最大回撤 | 夏普比率 | 交易次数 |\n")
        report_content.append("|----------|----------|----------|----------|----------|----------|\n")
        for r in all_vol_confirm_results:
            report_content.append(f"| {r['stock']} | {r['total_return']*100:.2f}% | {r['buy_hold_return']*100:.2f}% | {r['max_drawdown']*100:.2f}% | {r['sharpe_ratio']:.2f} | {r['trade_count']} |\n")

        avg_return = np.mean([r['total_return'] for r in all_vol_confirm_results])
        avg_bh_return = np.mean([r['buy_hold_return'] for r in all_vol_confirm_results])
        avg_mdd = np.mean([r['max_drawdown'] for r in all_vol_confirm_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_vol_confirm_results])
        report_content.append(f"| **平均** | **{avg_return*100:.2f}%** | **{avg_bh_return*100:.2f}%** | **{avg_mdd*100:.2f}%** | **{avg_sharpe:.2f}** | - |\n")
        report_content.append("\n")

    # 量价配合策略结果
    report_content.append("### 3.3 量价配合策略回测\n\n")
    report_content.append("**策略逻辑：**\n")
    report_content.append("- 买入信号: MFI < 20（超卖）且 VR < 70（安全区）\n")
    report_content.append("- 卖出信号: MFI > 80（超买）或 VR > 250（危险区）\n\n")

    if all_vol_price_results:
        report_content.append("**回测结果：**\n\n")
        report_content.append("| 股票代码 | 策略收益 | 持有收益 | 最大回撤 | 夏普比率 | 交易次数 |\n")
        report_content.append("|----------|----------|----------|----------|----------|----------|\n")
        for r in all_vol_price_results:
            report_content.append(f"| {r['stock']} | {r['total_return']*100:.2f}% | {r['buy_hold_return']*100:.2f}% | {r['max_drawdown']*100:.2f}% | {r['sharpe_ratio']:.2f} | {r['trade_count']} |\n")

        avg_return = np.mean([r['total_return'] for r in all_vol_price_results])
        avg_bh_return = np.mean([r['buy_hold_return'] for r in all_vol_price_results])
        avg_mdd = np.mean([r['max_drawdown'] for r in all_vol_price_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_vol_price_results])
        report_content.append(f"| **平均** | **{avg_return*100:.2f}%** | **{avg_bh_return*100:.2f}%** | **{avg_mdd*100:.2f}%** | **{avg_sharpe:.2f}** | - |\n")
        report_content.append("\n")

    # ========== 第四部分：策略对比与总结 ==========
    report_content.append("## 四、策略对比与总结\n\n")

    report_content.append("### 4.1 策略性能对比\n\n")

    if all_obv_results and all_vol_confirm_results and all_vol_price_results:
        strategy_comparison = [
            {
                'name': 'OBV策略',
                'avg_return': np.mean([r['total_return'] for r in all_obv_results]),
                'avg_mdd': np.mean([r['max_drawdown'] for r in all_obv_results]),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_obv_results])
            },
            {
                'name': '量能确认策略',
                'avg_return': np.mean([r['total_return'] for r in all_vol_confirm_results]),
                'avg_mdd': np.mean([r['max_drawdown'] for r in all_vol_confirm_results]),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_vol_confirm_results])
            },
            {
                'name': '量价配合策略',
                'avg_return': np.mean([r['total_return'] for r in all_vol_price_results]),
                'avg_mdd': np.mean([r['max_drawdown'] for r in all_vol_price_results]),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_vol_price_results])
            }
        ]

        report_content.append("| 策略名称 | 平均收益 | 平均最大回撤 | 平均夏普比率 |\n")
        report_content.append("|----------|----------|--------------|--------------|\n")
        for s in strategy_comparison:
            report_content.append(f"| {s['name']} | {s['avg_return']*100:.2f}% | {s['avg_mdd']*100:.2f}% | {s['avg_sharpe']:.2f} |\n")
        report_content.append("\n")

    report_content.append("### 4.2 研究结论\n\n")
    report_content.append("1. **OBV指标**：适合判断趋势的持续性，OBV与价格同向变动时趋势较强\n")
    report_content.append("2. **VR指标**：适合判断市场情绪，极端值时往往是反转信号\n")
    report_content.append("3. **PVT指标**：比OBV更灵敏，考虑了价格变动幅度\n")
    report_content.append("4. **MFI指标**：结合量价的RSI，超买超卖信号更可靠\n")
    report_content.append("5. **量价关系**：量增价涨是健康上涨，量价背离需警惕\n")
    report_content.append("6. **策略建议**：单一指标效果有限，建议多指标结合使用\n\n")

    report_content.append("### 4.3 风险提示\n\n")
    report_content.append("- 历史回测结果不代表未来表现\n")
    report_content.append("- 单一股票测试可能存在过拟合风险\n")
    report_content.append("- 实际交易需考虑交易成本和滑点\n")
    report_content.append("- 量能指标在小盘股和低流动性股票上可能失效\n")

    # 保存报告
    report_text = ''.join(report_content)
    report_path = REPORT_DIR + 'volume_indicators_research.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存至: {report_path}")

    return report_text


def generate_charts():
    """生成可视化图表"""
    print("\n4. 生成可视化图表...")

    vi = VolumeIndicators()
    vpa = VolumePriceAnalysis()

    # 获取示例股票数据
    df = vi.get_stock_data('000001.SZ', '20240101', '20260130')
    df = vi.calc_all_indicators(df)
    df = vpa.detect_volume_price_pattern(df)

    # 创建图表
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('000001.SZ 成交量技术指标分析', fontsize=14, fontweight='bold')

    # 1. 价格走势
    ax1 = axes[0]
    ax1.plot(df['trade_date'], df['close'], 'b-', linewidth=1, label='收盘价')
    ax1.set_ylabel('价格', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势')

    # 2. 成交量
    ax2 = axes[1]
    colors = ['red' if c > 0 else 'green' for c in df['pct_chg']]
    ax2.bar(df['trade_date'], df['vol'], color=colors, alpha=0.7, width=0.8)
    ax2.plot(df['trade_date'], df['vol_ma20'] if 'vol_ma20' in df.columns else df['vol'].rolling(20).mean(),
             'b-', linewidth=1, label='20日均量')
    ax2.set_ylabel('成交量', fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量')

    # 3. OBV
    ax3 = axes[2]
    ax3.plot(df['trade_date'], df['OBV'], 'b-', linewidth=1, label='OBV')
    ax3.plot(df['trade_date'], df['OBV_MA20'], 'r--', linewidth=1, label='OBV MA20')
    ax3.set_ylabel('OBV', fontsize=10)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('OBV (能量潮)')

    # 4. VR
    ax4 = axes[3]
    ax4.plot(df['trade_date'], df['VR'], 'b-', linewidth=1, label='VR')
    ax4.axhline(y=40, color='g', linestyle='--', alpha=0.7, label='超卖线(40)')
    ax4.axhline(y=150, color='y', linestyle='--', alpha=0.7, label='警戒线(150)')
    ax4.axhline(y=250, color='r', linestyle='--', alpha=0.7, label='超买线(250)')
    ax4.set_ylabel('VR', fontsize=10)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('VR (成交量比率)')

    # 5. MFI
    ax5 = axes[4]
    ax5.plot(df['trade_date'], df['MFI'], 'b-', linewidth=1, label='MFI')
    ax5.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='超卖线(20)')
    ax5.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='超买线(80)')
    ax5.fill_between(df['trade_date'], 20, 80, alpha=0.1, color='blue')
    ax5.set_ylabel('MFI', fontsize=10)
    ax5.set_xlabel('日期', fontsize=10)
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('MFI (资金流量指标)')

    plt.tight_layout()
    chart_path = REPORT_DIR + 'volume_indicators_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至: {chart_path}")

    # 生成策略回测图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('成交量策略回测结果', fontsize=14, fontweight='bold')

    bt = VolumeBacktest()

    # OBV策略
    obv_result = bt.obv_strategy(df)
    if obv_result:
        ax = axes[0, 0]
        result_df = obv_result['df']
        ax.plot(result_df['trade_date'], result_df['cumulative_return'], 'b-', label='OBV策略')
        ax.plot(result_df['trade_date'], result_df['buy_hold_return'], 'gray', alpha=0.7, label='买入持有')
        ax.set_title(f"OBV策略 (收益率: {obv_result['total_return']*100:.2f}%)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # 量能确认策略
    vol_confirm_result = bt.volume_confirmation_strategy(df)
    if vol_confirm_result:
        ax = axes[0, 1]
        result_df = vol_confirm_result['df']
        ax.plot(result_df['trade_date'], result_df['cumulative_return'], 'g-', label='量能确认策略')
        ax.plot(result_df['trade_date'], result_df['buy_hold_return'], 'gray', alpha=0.7, label='买入持有')
        ax.set_title(f"量能确认策略 (收益率: {vol_confirm_result['total_return']*100:.2f}%)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # 量价配合策略
    vol_price_result = bt.volume_price_strategy(df)
    if vol_price_result:
        ax = axes[1, 0]
        result_df = vol_price_result['df']
        ax.plot(result_df['trade_date'], result_df['cumulative_return'], 'r-', label='量价配合策略')
        ax.plot(result_df['trade_date'], result_df['buy_hold_return'], 'gray', alpha=0.7, label='买入持有')
        ax.set_title(f"量价配合策略 (收益率: {vol_price_result['total_return']*100:.2f}%)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # 回撤对比
    ax = axes[1, 1]
    if obv_result:
        ax.fill_between(obv_result['df']['trade_date'], obv_result['df']['drawdown'],
                       alpha=0.3, label='OBV策略')
    if vol_confirm_result:
        ax.fill_between(vol_confirm_result['df']['trade_date'], vol_confirm_result['df']['drawdown'],
                       alpha=0.3, label='量能确认策略')
    if vol_price_result:
        ax.fill_between(vol_price_result['df']['trade_date'], vol_price_result['df']['drawdown'],
                       alpha=0.3, label='量价配合策略')
    ax.set_title('策略回撤对比')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('回撤')

    plt.tight_layout()
    backtest_chart_path = REPORT_DIR + 'volume_backtest_chart.png'
    plt.savefig(backtest_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"回测图表已保存至: {backtest_chart_path}")

    # 量价关系分布图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('量价关系分析', fontsize=14, fontweight='bold')

    # 量价模式分布
    ax1 = axes[0]
    pattern_counts = df['volume_price_pattern'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#dfe6e9']
    ax1.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%',
           colors=colors[:len(pattern_counts)])
    ax1.set_title('量价模式分布')

    # 成交量与涨跌幅散点图
    ax2 = axes[1]
    vol_ratio = df['vol'] / df['vol'].rolling(20).mean()
    ax2.scatter(vol_ratio.dropna(), df['pct_chg'].loc[vol_ratio.dropna().index],
               alpha=0.5, s=10, c=df['pct_chg'].loc[vol_ratio.dropna().index],
               cmap='RdYlGn')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('成交量比率 (相对20日均量)')
    ax2.set_ylabel('涨跌幅 (%)')
    ax2.set_title('成交量与涨跌幅关系')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    vp_chart_path = REPORT_DIR + 'volume_price_analysis.png'
    plt.savefig(vp_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"量价分析图表已保存至: {vp_chart_path}")


def run_multi_stock_analysis():
    """多股票统计分析"""
    print("\n5. 执行多股票统计分析...")

    vi = VolumeIndicators()
    bt = VolumeBacktest()

    # 获取活跃股票列表
    stocks = vi.get_multiple_stocks(start_date='20230101', limit=50)
    print(f"  分析 {len(stocks)} 只股票...")

    results = {
        'obv': [],
        'vol_confirm': [],
        'vol_price': []
    }

    for i, stock in enumerate(stocks):
        try:
            df = vi.get_stock_data(stock, '20230101', '20260130')
            df = vi.calc_all_indicators(df)

            # OBV策略
            obv_result = bt.obv_strategy(df)
            if obv_result:
                results['obv'].append({
                    'stock': stock,
                    'return': obv_result['total_return'],
                    'sharpe': obv_result['sharpe_ratio'],
                    'mdd': obv_result['max_drawdown']
                })

            # 量能确认策略
            vol_confirm_result = bt.volume_confirmation_strategy(df)
            if vol_confirm_result:
                results['vol_confirm'].append({
                    'stock': stock,
                    'return': vol_confirm_result['total_return'],
                    'sharpe': vol_confirm_result['sharpe_ratio'],
                    'mdd': vol_confirm_result['max_drawdown']
                })

            # 量价配合策略
            vol_price_result = bt.volume_price_strategy(df)
            if vol_price_result:
                results['vol_price'].append({
                    'stock': stock,
                    'return': vol_price_result['total_return'],
                    'sharpe': vol_price_result['sharpe_ratio'],
                    'mdd': vol_price_result['max_drawdown']
                })

            if (i + 1) % 10 == 0:
                print(f"  已完成 {i+1}/{len(stocks)} 只股票")

        except Exception as e:
            pass

    # 生成统计报告
    stats_content = []
    stats_content.append("# 多股票成交量策略统计分析\n\n")
    stats_content.append(f"分析股票数: {len(stocks)}\n")
    stats_content.append(f"分析时间段: 2023-01-01 至 2026-01-30\n\n")

    for strategy_name, strategy_results in [('OBV策略', results['obv']),
                                             ('量能确认策略', results['vol_confirm']),
                                             ('量价配合策略', results['vol_price'])]:
        if strategy_results:
            returns = [r['return'] for r in strategy_results]
            sharpes = [r['sharpe'] for r in strategy_results]
            mdds = [r['mdd'] for r in strategy_results]

            stats_content.append(f"## {strategy_name}\n\n")
            stats_content.append(f"- 有效样本数: {len(strategy_results)}\n")
            stats_content.append(f"- 平均收益率: {np.mean(returns)*100:.2f}%\n")
            stats_content.append(f"- 收益率中位数: {np.median(returns)*100:.2f}%\n")
            stats_content.append(f"- 收益率标准差: {np.std(returns)*100:.2f}%\n")
            stats_content.append(f"- 正收益率比例: {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%\n")
            stats_content.append(f"- 平均夏普比率: {np.mean(sharpes):.2f}\n")
            stats_content.append(f"- 平均最大回撤: {np.mean(mdds)*100:.2f}%\n\n")

            # 最佳表现股票
            best_stocks = sorted(strategy_results, key=lambda x: x['return'], reverse=True)[:5]
            stats_content.append("**最佳表现股票:**\n\n")
            stats_content.append("| 股票代码 | 收益率 | 夏普比率 | 最大回撤 |\n")
            stats_content.append("|----------|--------|----------|----------|\n")
            for s in best_stocks:
                stats_content.append(f"| {s['stock']} | {s['return']*100:.2f}% | {s['sharpe']:.2f} | {s['mdd']*100:.2f}% |\n")
            stats_content.append("\n")

    stats_path = REPORT_DIR + 'volume_multi_stock_stats.md'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(''.join(stats_content))
    print(f"  多股票统计报告已保存至: {stats_path}")

    # 生成收益分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('多股票策略收益分布', fontsize=14, fontweight='bold')

    strategy_names = ['OBV策略', '量能确认策略', '量价配合策略']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (name, key) in enumerate([('OBV策略', 'obv'),
                                      ('量能确认策略', 'vol_confirm'),
                                      ('量价配合策略', 'vol_price')]):
        if results[key]:
            returns = [r['return'] * 100 for r in results[key]]
            axes[i].hist(returns, bins=20, color=colors[i], alpha=0.7, edgecolor='black')
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[i].axvline(x=np.mean(returns), color='blue', linestyle='-', alpha=0.7,
                          label=f'均值: {np.mean(returns):.1f}%')
            axes[i].set_title(name)
            axes[i].set_xlabel('收益率 (%)')
            axes[i].set_ylabel('股票数量')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = REPORT_DIR + 'volume_return_distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  收益分布图已保存至: {dist_path}")


if __name__ == '__main__':
    print("开始成交量技术指标研究...\n")

    # 生成研究报告
    report = generate_report()

    # 生成可视化图表
    generate_charts()

    # 多股票统计分析
    run_multi_stock_analysis()

    print("\n" + "=" * 60)
    print("研究完成！所有报告已保存至:")
    print(f"  {REPORT_DIR}")
    print("=" * 60)
