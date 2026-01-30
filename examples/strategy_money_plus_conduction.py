"""
资金潜伏 + 传导关系 组合策略

核心逻辑：
1. 检测领涨板块的资金潜伏信号
2. 基于历史传导关系，预测跟随板块
3. 提前布局跟随板块，等待传导兑现

优势：
- 资金信号提供入场时机
- 传导关系提供方向指引
- 双重确认，提高胜率
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader
from tushare_db.sector_analysis import SectorAnalyzer


class MoneyConductionStrategy:
    """资金传导组合策略"""

    def __init__(self, db_path: str):
        self.reader = DataReader(db_path=db_path)
        self.analyzer = SectorAnalyzer(db_path)

    def get_conduction_pairs(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        min_correlation: float = 0.2
    ) -> pd.DataFrame:
        """
        获取历史传导关系

        Returns:
            DataFrame with: sector_lead, sector_lag, lag_days, correlation
        """

        # 使用超额收益法计算传导关系
        lead_lag_df = self.analyzer.calculate_lead_lag_excess(
            start_date=start_date,
            end_date=end_date,
            max_lag=5,
            level=level,
            period='daily',
            min_correlation=min_correlation
        )

        return lead_lag_df

    def detect_money_signals(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        lookback: int = 30,
        money_threshold: float = 3.0,
        price_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        检测资金潜伏信号

        Returns:
            DataFrame with money lurking signals
        """

        if level == 'L1':
            code_col = 'l1_code'
            name_col = 'l1_name'
        else:
            code_col = 'l3_code'
            name_col = 'l3_name'

        # 获取板块综合数据
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
                SUM(m.buy_lg_amount + m.buy_elg_amount) as buy_lg_amount,
                SUM(m.sell_lg_amount + m.sell_elg_amount) as sell_lg_amount
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
            (m.buy_lg_amount - m.sell_lg_amount) as lg_net_amount,
            p.avg_pct_chg as pct_chg
        FROM sector_money m
        LEFT JOIN sector_price p ON m.trade_date = p.trade_date
            AND m.sector_code = p.sector_code
        ORDER BY m.trade_date, m.sector_code
        """

        df = self.reader.db.con.execute(query).fetchdf()

        # 计算滚动统计
        df = df.sort_values(['sector_code', 'trade_date'])

        for sector in df['sector_code'].unique():
            mask = df['sector_code'] == sector

            df.loc[mask, 'lg_net_ma'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback, min_periods=5
            ).mean()
            df.loc[mask, 'lg_net_std'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback, min_periods=5
            ).std()

            df.loc[mask, 'pct_chg_ma'] = df.loc[mask, 'pct_chg'].rolling(
                lookback, min_periods=5
            ).mean()
            df.loc[mask, 'pct_chg_std'] = df.loc[mask, 'pct_chg'].rolling(
                lookback, min_periods=5
            ).std()

        # 信号检测
        df['lg_net_zscore'] = (df['lg_net_amount'] - df['lg_net_ma']) / (df['lg_net_std'] + 1e-6)
        df['pct_chg_zscore'] = (df['pct_chg'] - df['pct_chg_ma']) / (df['pct_chg_std'] + 0.01)

        df['signal'] = (
            (df['lg_net_zscore'] > money_threshold) &
            (df['pct_chg_zscore'] < price_threshold) &
            (df['lg_net_amount'] > 0) &
            (df['pct_chg'].notna())
        )

        return df[df['signal']].copy()

    def generate_strategy_signals(
        self,
        money_signals: pd.DataFrame,
        conduction_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成组合策略信号

        逻辑：
        1. 领涨板块出现资金潜伏
        2. 查找该板块对应的跟随板块
        3. 在T+lag_days日布局跟随板块

        Returns:
            DataFrame with strategy signals
        """

        strategy_signals = []

        for idx, money_signal in money_signals.iterrows():
            sector_lead = money_signal['sector_code']
            signal_date = money_signal['trade_date']

            # 查找该板块作为领涨板块的传导关系
            conduction = conduction_pairs[
                conduction_pairs['sector_lead'] == sector_lead
            ]

            if len(conduction) == 0:
                continue

            # 为每个跟随板块生成信号
            for _, pair in conduction.iterrows():
                strategy_signals.append({
                    'signal_date': signal_date,
                    'sector_lead': sector_lead,
                    'sector_lead_name': money_signal['sector_name'],
                    'sector_lag': pair['sector_lag'],
                    'sector_lag_name': pair.get('sector_lag_name', pair['sector_lag']),
                    'lag_days': int(pair['lag_days']),
                    'correlation': pair['correlation'],
                    'lg_net_zscore': money_signal['lg_net_zscore'],
                    'pct_chg_zscore': money_signal['pct_chg_zscore']
                })

        return pd.DataFrame(strategy_signals)

    def calculate_strategy_returns(
        self,
        strategy_signals: pd.DataFrame,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        hold_days: int = 10
    ) -> pd.DataFrame:
        """
        计算策略收益

        在跟随板块上布局，持有N日
        """

        # 获取所有板块的价格数据
        if level == 'L1':
            code_col = 'l1_code'
        else:
            code_col = 'l3_code'

        query = f"""
        WITH sector_stocks AS (
            SELECT DISTINCT
                ts_code,
                {code_col} as sector_code
            FROM index_member_all
            WHERE is_new = 'Y'
            AND {code_col} IS NOT NULL
            AND in_date <= '{end_date}'
            AND (out_date IS NULL OR out_date >= '{start_date}')
        )
        SELECT
            d.trade_date,
            s.sector_code,
            AVG(d.pct_chg) as avg_pct_chg
        FROM daily d
        INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
        WHERE d.trade_date >= '{start_date}'
        AND d.trade_date <= '{end_date}'
        GROUP BY d.trade_date, s.sector_code
        ORDER BY d.trade_date, s.sector_code
        """

        price_data = self.reader.db.con.execute(query).fetchdf()

        # 计算每个信号的收益
        results = []

        for idx, signal in strategy_signals.iterrows():
            sector_lag = signal['sector_lag']
            signal_date = signal['signal_date']
            lag_days = signal['lag_days']

            # 获取跟随板块的价格序列
            sector_prices = price_data[
                price_data['sector_code'] == sector_lag
            ].sort_values('trade_date')

            # 找到信号日期
            signal_idx = sector_prices[sector_prices['trade_date'] == signal_date].index

            if len(signal_idx) == 0:
                continue

            signal_idx = signal_idx[0]

            # 在T+lag_days日买入
            entry_data = sector_prices.loc[signal_idx:].head(lag_days + 1)

            if len(entry_data) <= lag_days:
                continue  # 数据不足

            # 持有hold_days计算收益
            hold_data = sector_prices.loc[signal_idx:].head(lag_days + hold_days + 1)

            if len(hold_data) <= lag_days + hold_days:
                continue

            # 计算从T+lag_days到T+lag_days+hold_days的累计收益
            returns = hold_data['avg_pct_chg'].iloc[lag_days+1:lag_days+hold_days+1].sum()

            result = {
                **signal.to_dict(),
                'hold_days': hold_days,
                'forward_return': returns
            }

            results.append(result)

        return pd.DataFrame(results)

    def close(self):
        self.reader.close()


def backtest_combined_strategy(
    train_start: str = '20200101',
    train_end: str = '20221231',
    test_start: str = '20230101',
    test_end: str = '20241231'
):
    """
    回测组合策略

    训练期：学习传导关系
    测试期：应用策略
    """

    print("=" * 80)
    print("资金潜伏 + 传导关系 组合策略回测")
    print("=" * 80)
    print(f"训练期: {train_start} ~ {train_end}")
    print(f"测试期: {test_start} ~ {test_end}")
    print()

    strategy = MoneyConductionStrategy('tushare.db')

    # 第一步：学习传导关系（训练期）
    print("[1/5] 学习历史传导关系（训练期）...")
    conduction_pairs = strategy.get_conduction_pairs(
        start_date=train_start,
        end_date=train_end,
        level='L1',
        min_correlation=0.2
    )
    print(f"      找到 {len(conduction_pairs)} 对传导关系")

    if len(conduction_pairs) == 0:
        print("未找到传导关系，退出")
        strategy.close()
        return

    # 显示传导关系
    print("\n      传导关系示例（Top 10）:")
    top_pairs = conduction_pairs.nlargest(10, 'correlation')[
        ['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation']
    ]
    for _, pair in top_pairs.iterrows():
        print(f"        {pair['sector_lead_name']} → {pair['sector_lag_name']} "
              f"(滞后{pair['lag_days']}天, r={pair['correlation']:.3f})")

    # 第二步：检测资金信号（测试期）
    print(f"\n[2/5] 检测资金潜伏信号（测试期）...")
    money_signals = strategy.detect_money_signals(
        start_date=test_start,
        end_date=test_end,
        level='L1',
        lookback=30,
        money_threshold=3.0,  # 使用优化后的参数
        price_threshold=1.0
    )
    print(f"      找到 {len(money_signals)} 个资金信号")

    if len(money_signals) == 0:
        print("未找到资金信号，退出")
        strategy.close()
        return

    # 显示资金信号
    print("\n      资金信号示例（Top 5）:")
    for idx, signal in money_signals.head(5).iterrows():
        print(f"        {signal['trade_date']} - {signal['sector_name']} "
              f"(资金强度={signal['lg_net_zscore']:.2f}σ)")

    # 第三步：生成策略信号
    print(f"\n[3/5] 生成组合策略信号...")
    strategy_signals = strategy.generate_strategy_signals(
        money_signals=money_signals,
        conduction_pairs=conduction_pairs
    )
    print(f"      生成 {len(strategy_signals)} 个策略信号")

    if len(strategy_signals) == 0:
        print("未生成策略信号（资金信号的板块不在传导关系中）")
        strategy.close()
        return

    # 显示策略信号
    print("\n      策略信号示例（Top 5）:")
    for idx, signal in strategy_signals.head(5).iterrows():
        print(f"        {signal['signal_date']}: {signal['sector_lead_name']}出现资金潜伏 "
              f"→ {signal['lag_days']}天后布局{signal['sector_lag_name']} "
              f"(传导r={signal['correlation']:.3f})")

    # 第四步：计算收益
    print(f"\n[4/5] 计算策略收益...")

    # 测试不同持有期
    hold_periods = [5, 10, 20]
    all_results = {}

    for hold_days in hold_periods:
        returns_df = strategy.calculate_strategy_returns(
            strategy_signals=strategy_signals,
            start_date=test_start,
            end_date=test_end,
            level='L1',
            hold_days=hold_days
        )

        if len(returns_df) > 0:
            all_results[hold_days] = returns_df
            print(f"      {hold_days}日持有: {len(returns_df)} 个有效信号")

    # 第五步：分析表现
    print(f"\n[5/5] 分析策略表现...")

    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)

    for hold_days, returns_df in all_results.items():
        valid_returns = returns_df['forward_return'].dropna()

        if len(valid_returns) == 0:
            continue

        win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
        avg_return = valid_returns.mean()
        median_return = valid_returns.median()
        max_return = valid_returns.max()
        min_return = valid_returns.min()
        std_return = valid_returns.std()
        sharpe = avg_return / (std_return + 1e-6)

        print(f"\n【{hold_days}日持有期】")
        print(f"  有效信号: {len(valid_returns)}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  中位数收益: {median_return:.2f}%")
        print(f"  最大收益: {max_return:.2f}%")
        print(f"  最小收益: {min_return:.2f}%")
        print(f"  标准差: {std_return:.2f}%")
        print(f"  夏普比率: {sharpe:.3f}")

    # 详细分析
    if 10 in all_results:
        returns_df = all_results[10]

        print("\n" + "=" * 80)
        print("信号详细分析（10日持有期）")
        print("=" * 80)

        # 按领涨板块统计
        print("\n各领涨板块信号统计:")
        lead_stats = returns_df.groupby('sector_lead_name').agg({
            'signal_date': 'count',
            'forward_return': 'mean'
        }).rename(columns={'signal_date': '信号数', 'forward_return': '平均收益'})
        lead_stats = lead_stats.sort_values('信号数', ascending=False).head(10)
        print(lead_stats.to_string())

        # 最佳和最差信号
        print("\n最佳信号Top 5:")
        best = returns_df.nlargest(5, 'forward_return')[
            ['signal_date', 'sector_lead_name', 'sector_lag_name', 'lag_days', 'forward_return']
        ]
        for _, row in best.iterrows():
            print(f"  {row['signal_date']}: {row['sector_lead_name']}→{row['sector_lag_name']} "
                  f"(滞后{row['lag_days']}天, 收益{row['forward_return']:.2f}%)")

        print("\n最差信号Top 5:")
        worst = returns_df.nsmallest(5, 'forward_return')[
            ['signal_date', 'sector_lead_name', 'sector_lag_name', 'lag_days', 'forward_return']
        ]
        for _, row in worst.iterrows():
            print(f"  {row['signal_date']}: {row['sector_lead_name']}→{row['sector_lag_name']} "
                  f"(滞后{row['lag_days']}天, 收益{row['forward_return']:.2f}%)")

    # 保存结果
    output_dir = Path('output/money_conduction_strategy')
    output_dir.mkdir(parents=True, exist_ok=True)

    for hold_days, returns_df in all_results.items():
        returns_df.to_csv(
            output_dir / f'signals_{hold_days}d.csv',
            index=False,
            encoding='utf-8-sig'
        )

    conduction_pairs.to_csv(
        output_dir / 'conduction_pairs.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print(f"\n详细结果已保存至: {output_dir}")

    strategy.close()

    return all_results


if __name__ == '__main__':
    # 运行回测
    results = backtest_combined_strategy(
        train_start='20200101',
        train_end='20221231',
        test_start='20230101',
        test_end='20241231'
    )
