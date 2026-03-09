#!/usr/bin/env python3
"""
有效因子多空成功率分析

分析13个通过验证的因子在历史数据上的实际表现：
1. 信号触发频率
2. 看多/看空成功率（不同持有期）
3. 平均收益率
4. 盈亏比
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tushare_db import DataReader
from src.tushare_db.factor_validation import FactorRegistry


@dataclass
class FactorSignalAnalysis:
    """因子信号分析结果"""
    factor_name: str
    signal_direction: str  # BULLISH / BEARISH / NEUTRAL
    total_signals: int

    # 不同持有期的成功率
    success_rate_1d: float
    success_rate_5d: float
    success_rate_20d: float

    # 平均收益率
    avg_return_1d: float
    avg_return_5d: float
    avg_return_20d: float

    # 盈亏比
    profit_loss_ratio: float

    # 最大/最小收益
    max_return: float
    min_return: float


class FactorSuccessRateAnalyzer:
    """因子成功率分析器"""

    # 有效因子及其方向
    VALID_FACTORS = {
        # 蜡烛图形态
        'shooting_star': 'BEARISH',
        'hammer': 'BULLISH',
        'bullish_engulfing': 'BULLISH',
        'bearish_engulfing': 'BEARISH',

        # 波动率
        'volatility_expansion': 'NEUTRAL',  # 双向都可能
        'volatility_contraction': 'NEUTRAL',  # 通常是突破前兆

        # 突破类
        'price_momentum_breakout': 'BULLISH',
        'support_resistance_breakout': 'BULLISH',
        'atr_breakout': 'BULLISH',
        'atr_breakdown': 'BEARISH',

        # 成交量
        'volume_weighted_momentum': 'BULLISH',

        # 跳空
        'gap_up': 'BULLISH',
        'gap_down': 'BEARISH',
    }

    def __init__(self, start_date: str = '20200101', end_date: str = None):
        self.reader = DataReader()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')

    def load_stock_data(self, ts_code: str) -> Optional[pd.DataFrame]:
        """加载单只股票数据"""
        try:
            query = """
                SELECT ts_code, trade_date, open, high, low, close, vol
                FROM daily
                WHERE ts_code = ?
                AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date
            """
            df = self.reader.db.execute_query(query, [ts_code, self.start_date, self.end_date])
            if len(df) < 252:  # 至少需要一年数据
                return None
            return df
        except Exception as e:
            print(f"  加载 {ts_code} 失败: {e}")
            return None

    def calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算未来收益率"""
        df = df.copy()
        df['return_1d'] = df['close'].shift(-1) / df['close'] - 1
        df['return_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['return_20d'] = df['close'].shift(-20) / df['close'] - 1
        return df

    def analyze_factor(self, factor_name: str, ts_codes: List[str]) -> Optional[FactorSignalAnalysis]:
        """分析单个因子在多个股票上的表现"""
        print(f"\n分析因子: {factor_name}")

        factor = FactorRegistry.get(factor_name)
        signal_direction = self.VALID_FACTORS.get(factor_name, 'NEUTRAL')

        all_signals = []

        for ts_code in ts_codes:
            df = self.load_stock_data(ts_code)
            if df is None:
                continue

            # 计算未来收益
            df = self.calculate_forward_returns(df)

            # 计算因子信号
            try:
                signals = factor.evaluate(df[['open', 'high', 'low', 'close', 'vol']])
                df['signal'] = signals
            except Exception as e:
                print(f"  {ts_code} 计算信号失败: {e}")
                continue

            # 提取信号日
            signal_days = df[df['signal'] == True].copy()

            if len(signal_days) > 0:
                all_signals.append(signal_days[['ts_code', 'trade_date', 'close',
                                               'return_1d', 'return_5d', 'return_20d']])

        if not all_signals:
            print(f"  无信号数据")
            return None

        # 合并所有信号
        all_signals_df = pd.concat(all_signals, ignore_index=True)
        total_signals = len(all_signals_df)

        print(f"  总信号数: {total_signals}")

        if total_signals < 10:
            print(f"  信号太少，跳过")
            return None

        # 计算成功率（根据信号方向）
        # 注意：对于BEARISH信号，我们期待价格*下跌*，所以收益<0是成功
        if signal_direction == 'BULLISH':
            success_1d = (all_signals_df['return_1d'] > 0).mean()
            success_5d = (all_signals_df['return_5d'] > 0).mean()
            success_20d = (all_signals_df['return_20d'] > 0).mean()
        elif signal_direction == 'BEARISH':
            success_1d = (all_signals_df['return_1d'] < 0).mean()
            success_5d = (all_signals_df['return_5d'] < 0).mean()
            success_20d = (all_signals_df['return_20d'] < 0).mean()
        else:
            success_1d = success_5d = success_20d = np.nan

        # 计算平均收益（对于BEARISH因子，使用做空收益 = -实际收益）
        if signal_direction == 'BEARISH':
            avg_return_1d = -all_signals_df['return_1d'].mean()  # 做空收益
            avg_return_5d = -all_signals_df['return_5d'].mean()
            avg_return_20d = -all_signals_df['return_20d'].mean()
        else:
            avg_return_1d = all_signals_df['return_1d'].mean()
            avg_return_5d = all_signals_df['return_5d'].mean()
            avg_return_20d = all_signals_df['return_20d'].mean()

        # 计算盈亏比
        returns = all_signals_df['return_1d'].dropna()
        if signal_direction == 'BULLISH':
            avg_profit = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        elif signal_direction == 'BEARISH':
            # 对于做空，盈利是价格下跌（return < 0）
            avg_profit = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
            avg_loss = returns[returns > 0].mean() if (returns > 0).any() else 1
        else:
            avg_profit = avg_loss = np.nan

        profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else np.inf

        return FactorSignalAnalysis(
            factor_name=factor_name,
            signal_direction=signal_direction,
            total_signals=total_signals,
            success_rate_1d=success_1d,
            success_rate_5d=success_5d,
            success_rate_20d=success_20d,
            avg_return_1d=avg_return_1d,
            avg_return_5d=avg_return_5d,
            avg_return_20d=avg_return_20d,
            profit_loss_ratio=profit_loss_ratio,
            max_return=returns.max(),
            min_return=returns.min()
        )

    def run_analysis(self, max_stocks: int = 100) -> pd.DataFrame:
        """运行全部分析"""
        print("="*80)
        print("有效因子多空成功率分析")
        print("="*80)
        print(f"数据范围: {self.start_date} 至 {self.end_date}")
        print(f"分析股票数: 最多 {max_stocks} 只")
        print()

        # 获取股票列表
        stocks = self.reader.get_stock_basic(list_status='L')
        # 按市值排序，取大盘股
        stocks = stocks.head(max_stocks)
        ts_codes = stocks['ts_code'].tolist()

        print(f"实际分析股票: {len(ts_codes)} 只")
        print()

        # 分析每个因子
        results = []
        for factor_name in self.VALID_FACTORS.keys():
            result = self.analyze_factor(factor_name, ts_codes)
            if result:
                results.append(result)

        # 转换为DataFrame
        df_results = pd.DataFrame([{
            '因子': r.factor_name,
            '方向': r.signal_direction,
            '信号数': r.total_signals,
            '1日成功率': r.success_rate_1d,
            '5日成功率': r.success_rate_5d,
            '20日成功率': r.success_rate_20d,
            '1日平均收益': r.avg_return_1d,
            '5日平均收益': r.avg_return_5d,
            '20日平均收益': r.avg_return_20d,
            '盈亏比': r.profit_loss_ratio,
            '最大收益': r.max_return,
            '最大亏损': r.min_return
        } for r in results])

        return df_results


def print_report(df: pd.DataFrame):
    """打印分析报告"""
    print("\n" + "="*80)
    print("分析结果汇总")
    print("="*80)

    # 按方向分组
    bullish_df = df[df['方向'] == 'BULLISH'].sort_values('5日成功率', ascending=False)
    bearish_df = df[df['方向'] == 'BEARISH'].sort_values('5日成功率', ascending=False)

    print("\n【看多因子成功率排名】")
    print("-"*80)
    print(f"{'因子':<25} {'信号数':<8} {'1日':<8} {'5日':<8} {'20日':<8} {'平均收益':<10}")
    print("-"*80)
    for _, row in bullish_df.iterrows():
        print(f"{row['因子']:<25} {row['信号数']:<8} {row['1日成功率']:.1%}    {row['5日成功率']:.1%}    {row['20日成功率']:.1%}    {row['5日平均收益']:>+.2%}")

    print("\n【看空因子成功率排名】")
    print("-"*80)
    print(f"{'因子':<25} {'信号数':<8} {'1日':<8} {'5日':<8} {'20日':<8} {'平均收益':<10}")
    print("-"*80)
    for _, row in bearish_df.iterrows():
        print(f"{row['因子']:<25} {row['信号数']:<8} {row['1日成功率']:.1%}    {row['5日成功率']:.1%}    {row['20日成功率']:.1%}    {row['5日平均收益']:>+.2%}")

    print("\n" + "="*80)
    print("关键洞察")
    print("="*80)

    # 找出最佳因子
    best_bullish = bullish_df.iloc[0] if len(bullish_df) > 0 else None
    best_bearish = bearish_df.iloc[0] if len(bearish_df) > 0 else None

    if best_bullish is not None:
        print(f"\n最佳看多因子: {best_bullish['因子']}")
        print(f"  5日成功率: {best_bullish['5日成功率']:.1%}")
        print(f"  5日平均收益: {best_bullish['5日平均收益']:+.2%}")

    if best_bearish is not None:
        print(f"\n最佳看空因子: {best_bearish['因子']}")
        print(f"  5日成功率: {best_bearish['5日成功率']:.1%}")
        print(f"  5日平均收益: {best_bearish['5日平均收益']:+.2%}")


def main():
    analyzer = FactorSuccessRateAnalyzer(
        start_date='20200101',  # 使用2020年以来的数据
        end_date='20260309'
    )

    df_results = analyzer.run_analysis(max_stocks=100)

    # 保存结果
    output_file = f"validation_results/factor_success_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 打印报告
    print_report(df_results)


if __name__ == '__main__':
    main()
