"""
资金信号回测框架

目标：验证资金潜伏信号是否有效
方法：统计信号后N日收益率，计算胜率、平均收益等指标
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


class MoneySignalBacktest:
    """资金信号回测器"""

    def __init__(self, db_path: str):
        self.reader = DataReader(db_path=db_path)

    def get_sector_data(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        获取板块综合数据（资金+价格）

        Returns:
            DataFrame with: trade_date, sector_code, sector_name,
                           net_amount, lg_net_amount, pct_chg, vol, amount
        """

        if level == 'L1':
            code_col = 'l1_code'
            name_col = 'l1_name'
        elif level == 'L2':
            code_col = 'l2_code'
            name_col = 'l2_name'
        else:
            code_col = 'l3_code'
            name_col = 'l3_name'

        # 获取板块资金流向和价格数据
        query = f"""
        WITH sector_stocks AS (
            SELECT DISTINCT
                ts_code,
                {code_col} as sector_code,
                {name_col} as sector_name
            FROM index_member_all
            WHERE is_new = 'Y'
            AND {code_col} IS NOT NULL
            AND in_date <= '{end_date}'
            AND (out_date IS NULL OR out_date >= '{start_date}')
        ),
        sector_money AS (
            SELECT
                m.trade_date,
                s.sector_code,
                s.sector_name,
                SUM(m.net_mf_amount) as net_amount,
                SUM(m.buy_lg_amount + m.buy_elg_amount) as buy_lg_amount,
                SUM(m.sell_lg_amount + m.sell_elg_amount) as sell_lg_amount,
                COUNT(DISTINCT m.ts_code) as stock_count
            FROM moneyflow m
            INNER JOIN sector_stocks s ON m.ts_code = s.ts_code
            WHERE m.trade_date >= '{start_date}'
            AND m.trade_date <= '{end_date}'
            GROUP BY m.trade_date, s.sector_code, s.sector_name
        ),
        sector_price AS (
            SELECT
                d.trade_date,
                s.sector_code,
                SUM(d.vol) as total_vol,
                SUM(d.amount) as total_amount,
                AVG(d.pct_chg) as avg_pct_chg
            FROM daily d
            INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
            WHERE d.trade_date >= '{start_date}'
            AND d.trade_date <= '{end_date}'
            GROUP BY d.trade_date, s.sector_code
        )
        SELECT
            m.trade_date,
            m.sector_code,
            m.sector_name,
            m.net_amount,
            (m.buy_lg_amount - m.sell_lg_amount) as lg_net_amount,
            m.buy_lg_amount,
            m.sell_lg_amount,
            p.avg_pct_chg as pct_chg,
            p.total_vol as vol,
            p.total_amount as amount,
            m.stock_count
        FROM sector_money m
        LEFT JOIN sector_price p ON m.trade_date = p.trade_date
            AND m.sector_code = p.sector_code
        ORDER BY m.trade_date, m.sector_code
        """

        df = self.reader.db.con.execute(query).fetchdf()

        # 计算大单占比
        df['lg_ratio'] = df['lg_net_amount'] / (df['buy_lg_amount'] + df['sell_lg_amount'] + 1)

        return df

    def detect_signals(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        money_threshold: float = 1.5,  # 资金流入倍数
        price_threshold: float = 0.5   # 涨幅倍数
    ) -> pd.DataFrame:
        """
        检测资金信号

        参数：
            lookback: 历史回看天数
            money_threshold: 大单净流入需超过均值的倍数
            price_threshold: 涨幅需低于均值的倍数
        """

        df = df.copy().sort_values(['sector_code', 'trade_date'])

        # 计算滚动统计
        for sector in df['sector_code'].unique():
            mask = df['sector_code'] == sector

            # 资金流入统计
            df.loc[mask, 'lg_net_ma'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback, min_periods=5
            ).mean()
            df.loc[mask, 'lg_net_std'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback, min_periods=5
            ).std()

            # 涨跌幅统计
            df.loc[mask, 'pct_chg_ma'] = df.loc[mask, 'pct_chg'].rolling(
                lookback, min_periods=5
            ).mean()
            df.loc[mask, 'pct_chg_std'] = df.loc[mask, 'pct_chg'].rolling(
                lookback, min_periods=5
            ).std()

        # 计算标准化指标
        df['lg_net_zscore'] = (df['lg_net_amount'] - df['lg_net_ma']) / (df['lg_net_std'] + 1e-6)
        df['pct_chg_zscore'] = (df['pct_chg'] - df['pct_chg_ma']) / (df['pct_chg_std'] + 0.01)

        # 信号定义
        df['signal'] = (
            (df['lg_net_zscore'] > money_threshold) &  # 资金流入超过阈值
            (df['pct_chg_zscore'] < price_threshold) &  # 涨幅相对较小
            (df['lg_net_amount'] > 0) &  # 大单净流入为正
            (df['pct_chg'].notna())  # 有价格数据
        )

        return df

    def calculate_forward_returns(
        self,
        signals_df: pd.DataFrame,
        all_data_df: pd.DataFrame,
        periods: list = [1, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        计算信号后N日收益率

        参数：
            signals_df: 包含信号的DataFrame
            all_data_df: 所有日期的数据（用于查找后续收益）
            periods: 需要计算的持有期列表
        """

        results = []

        # 只保留有信号的行
        signals = signals_df[signals_df['signal']].copy()

        for idx, signal in signals.iterrows():
            sector_code = signal['sector_code']
            signal_date = signal['trade_date']

            # 获取该板块所有数据
            sector_data = all_data_df[
                all_data_df['sector_code'] == sector_code
            ].sort_values('trade_date')

            # 找到信号日期的位置
            signal_idx = sector_data[sector_data['trade_date'] == signal_date].index

            if len(signal_idx) == 0:
                continue

            signal_idx = signal_idx[0]

            # 计算各期收益
            result = {
                'signal_date': signal_date,
                'sector_code': sector_code,
                'sector_name': signal['sector_name'],
                'signal_pct_chg': signal['pct_chg'],
                'lg_net_amount': signal['lg_net_amount'],
                'lg_net_zscore': signal['lg_net_zscore'],
                'pct_chg_zscore': signal['pct_chg_zscore']
            }

            for period in periods:
                # 获取N日后的收益（累计）
                future_data = sector_data.loc[signal_idx:].head(period + 1)

                if len(future_data) > period:
                    # 计算累计收益
                    cumulative_return = future_data['pct_chg'].iloc[1:period+1].sum()
                    result[f'return_{period}d'] = cumulative_return
                else:
                    result[f'return_{period}d'] = np.nan

            results.append(result)

        return pd.DataFrame(results)

    def analyze_performance(
        self,
        returns_df: pd.DataFrame,
        periods: list = [1, 3, 5, 10]
    ) -> dict:
        """
        分析策略表现

        Returns:
            包含各项指标的字典
        """

        metrics = {}

        for period in periods:
            col = f'return_{period}d'

            if col not in returns_df.columns:
                continue

            # 过滤掉缺失值
            valid_returns = returns_df[col].dropna()

            if len(valid_returns) == 0:
                continue

            # 计算指标
            metrics[f'{period}d'] = {
                '信号数': len(valid_returns),
                '胜率': (valid_returns > 0).sum() / len(valid_returns) * 100,
                '平均收益': valid_returns.mean(),
                '中位数收益': valid_returns.median(),
                '最大收益': valid_returns.max(),
                '最小收益': valid_returns.min(),
                '标准差': valid_returns.std(),
                '夏普比率': valid_returns.mean() / (valid_returns.std() + 1e-6) if len(valid_returns) > 1 else 0,
                '盈亏比': valid_returns[valid_returns > 0].mean() / abs(valid_returns[valid_returns < 0].mean())
                          if (valid_returns < 0).sum() > 0 else np.inf
            }

        return metrics

    def close(self):
        self.reader.close()


def run_backtest(
    start_date: str = '20200101',
    end_date: str = '20241231',
    level: str = 'L1'
):
    """运行完整回测"""

    print("=" * 80)
    print("资金信号策略回测")
    print("=" * 80)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"板块层级: {level}")
    print()

    # 初始化
    backtester = MoneySignalBacktest('tushare.db')

    # 1. 获取数据
    print("[1/4] 正在获取板块数据...")
    sector_data = backtester.get_sector_data(start_date, end_date, level)
    print(f"      数据量: {len(sector_data)} 条")
    print(f"      板块数: {sector_data['sector_code'].nunique()}")
    print(f"      交易日: {sector_data['trade_date'].nunique()}")

    # 2. 检测信号
    print("\n[2/4] 正在检测资金信号...")
    signals_data = backtester.detect_signals(
        sector_data,
        lookback=20,
        money_threshold=1.5,  # 资金流入>均值1.5倍
        price_threshold=0.5    # 涨幅<均值0.5倍
    )
    signal_count = signals_data['signal'].sum()
    print(f"      检测到信号: {signal_count} 个")

    if signal_count == 0:
        print("\n⚠️  未检测到任何信号，请调整参数")
        backtester.close()
        return

    # 3. 计算后续收益
    print("\n[3/4] 正在计算后续收益...")
    returns_df = backtester.calculate_forward_returns(
        signals_data,
        sector_data,
        periods=[1, 3, 5, 10, 20]
    )
    print(f"      成功计算: {len(returns_df)} 个信号的收益")

    # 4. 分析表现
    print("\n[4/4] 正在分析策略表现...")
    metrics = backtester.analyze_performance(returns_df, periods=[1, 3, 5, 10, 20])

    # 输出结果
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)

    for period, data in metrics.items():
        print(f"\n【{period}持有期】")
        print(f"  信号数量: {data['信号数']}")
        print(f"  胜率: {data['胜率']:.2f}%")
        print(f"  平均收益: {data['平均收益']:.2f}%")
        print(f"  中位数收益: {data['中位数收益']:.2f}%")
        print(f"  最大收益: {data['最大收益']:.2f}%")
        print(f"  最小收益: {data['最小收益']:.2f}%")
        print(f"  标准差: {data['标准差']:.2f}%")
        print(f"  夏普比率: {data['夏普比率']:.3f}")
        print(f"  盈亏比: {data['盈亏比']:.2f}")

    # 详细信号分析
    print("\n" + "=" * 80)
    print("信号详细分析")
    print("=" * 80)

    # 按板块统计
    print("\n各板块信号统计:")
    sector_stats = returns_df.groupby('sector_name').agg({
        'signal_date': 'count',
        'return_5d': 'mean'
    }).rename(columns={'signal_date': '信号数', 'return_5d': '平均5日收益'})
    sector_stats = sector_stats.sort_values('信号数', ascending=False).head(10)
    print(sector_stats.to_string())

    # 最佳和最差信号
    print("\n最佳信号Top 5（5日收益）:")
    best_signals = returns_df.nlargest(5, 'return_5d')[
        ['signal_date', 'sector_name', 'signal_pct_chg', 'return_5d']
    ]
    print(best_signals.to_string(index=False))

    print("\n最差信号Top 5（5日收益）:")
    worst_signals = returns_df.nsmallest(5, 'return_5d')[
        ['signal_date', 'sector_name', 'signal_pct_chg', 'return_5d']
    ]
    print(worst_signals.to_string(index=False))

    # 保存结果
    output_dir = Path('output/backtest_money_signal')
    output_dir.mkdir(parents=True, exist_ok=True)

    returns_df.to_csv(output_dir / 'signal_returns.csv', index=False, encoding='utf-8-sig')
    signals_data[signals_data['signal']].to_csv(
        output_dir / 'all_signals.csv', index=False, encoding='utf-8-sig'
    )

    print(f"\n详细数据已保存至: {output_dir}")

    backtester.close()

    return metrics, returns_df


def parameter_optimization():
    """参数优化：测试不同参数组合"""

    print("=" * 80)
    print("参数优化测试")
    print("=" * 80)

    backtester = MoneySignalBacktest('tushare.db')

    # 获取数据（2年数据用于快速测试）
    sector_data = backtester.get_sector_data('20230101', '20241231', 'L1')

    # 测试不同参数组合
    param_grid = {
        'money_threshold': [1.0, 1.5, 2.0],
        'price_threshold': [0.0, 0.5, 1.0]
    }

    results = []

    for money_th in param_grid['money_threshold']:
        for price_th in param_grid['price_threshold']:
            print(f"\n测试参数: 资金阈值={money_th}, 价格阈值={price_th}")

            signals_data = backtester.detect_signals(
                sector_data,
                lookback=20,
                money_threshold=money_th,
                price_threshold=price_th
            )

            signal_count = signals_data['signal'].sum()
            print(f"  信号数: {signal_count}")

            if signal_count < 5:
                print("  信号过少，跳过")
                continue

            returns_df = backtester.calculate_forward_returns(
                signals_data, sector_data, periods=[5]
            )

            if len(returns_df) == 0:
                continue

            # 5日收益表现
            valid_returns = returns_df['return_5d'].dropna()
            win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
            avg_return = valid_returns.mean()

            print(f"  5日胜率: {win_rate:.2f}%")
            print(f"  5日平均收益: {avg_return:.2f}%")

            results.append({
                'money_threshold': money_th,
                'price_threshold': price_th,
                'signal_count': signal_count,
                'win_rate_5d': win_rate,
                'avg_return_5d': avg_return
            })

    # 输出最佳参数
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("参数优化结果（按5日平均收益排序）:")
        print("=" * 80)
        print(results_df.sort_values('avg_return_5d', ascending=False).to_string(index=False))

    backtester.close()


if __name__ == '__main__':
    # 运行完整回测
    metrics, returns_df = run_backtest(
        start_date='20200101',
        end_date='20241231',
        level='L1'
    )

    # 参数优化（可选）
    print("\n\n")
    user_input = input("是否进行参数优化测试？(y/n): ")
    if user_input.lower() == 'y':
        parameter_optimization()
