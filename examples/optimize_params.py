"""
参数优化：测试不同参数组合的效果
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


def get_sector_data(reader, start_date, end_date, level='L1'):
    """获取板块数据"""
    if level == 'L1':
        code_col = 'l1_code'
        name_col = 'l1_name'
    else:
        code_col = 'l3_code'
        name_col = 'l3_name'

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

    df = reader.db.con.execute(query).fetchdf()
    df['lg_ratio'] = df['lg_net_amount'] / (df['buy_lg_amount'] + df['sell_lg_amount'] + 1)
    return df


def detect_signals(df, lookback=20, money_threshold=1.5, price_threshold=0.5):
    """检测资金信号"""
    df = df.copy().sort_values(['sector_code', 'trade_date'])

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

    df['lg_net_zscore'] = (df['lg_net_amount'] - df['lg_net_ma']) / (df['lg_net_std'] + 1e-6)
    df['pct_chg_zscore'] = (df['pct_chg'] - df['pct_chg_ma']) / (df['pct_chg_std'] + 0.01)

    df['signal'] = (
        (df['lg_net_zscore'] > money_threshold) &
        (df['pct_chg_zscore'] < price_threshold) &
        (df['lg_net_amount'] > 0) &
        (df['pct_chg'].notna())
    )

    return df


def calculate_forward_returns(signals_df, all_data_df, periods=[1, 3, 5]):
    """计算信号后N日收益"""
    results = []
    signals = signals_df[signals_df['signal']].copy()

    for idx, signal in signals.iterrows():
        sector_code = signal['sector_code']
        signal_date = signal['trade_date']

        sector_data = all_data_df[
            all_data_df['sector_code'] == sector_code
        ].sort_values('trade_date')

        signal_idx = sector_data[sector_data['trade_date'] == signal_date].index

        if len(signal_idx) == 0:
            continue

        signal_idx = signal_idx[0]

        result = {
            'signal_date': signal_date,
            'sector_code': sector_code,
            'sector_name': signal['sector_name'],
        }

        for period in periods:
            future_data = sector_data.loc[signal_idx:].head(period + 1)

            if len(future_data) > period:
                cumulative_return = future_data['pct_chg'].iloc[1:period+1].sum()
                result[f'return_{period}d'] = cumulative_return
            else:
                result[f'return_{period}d'] = None

        results.append(result)

    return pd.DataFrame(results)


def main():
    """运行参数优化"""
    print("=" * 80)
    print("参数优化测试")
    print("=" * 80)

    reader = DataReader('tushare.db')

    # 使用2年数据快速测试
    print("\n加载数据（2023-2024）...")
    sector_data = get_sector_data(reader, '20230101', '20241231', 'L1')
    print(f"数据量: {len(sector_data)} 条")

    # 测试参数网格
    param_grid = {
        'money_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],
        'price_threshold': [-0.5, 0.0, 0.5, 1.0],
        'lookback': [10, 20, 30]
    }

    results = []
    total_tests = len(param_grid['money_threshold']) * len(param_grid['price_threshold']) * len(param_grid['lookback'])
    test_count = 0

    print(f"\n开始测试 {total_tests} 组参数...\n")

    for lookback in param_grid['lookback']:
        for money_th in param_grid['money_threshold']:
            for price_th in param_grid['price_threshold']:
                test_count += 1

                signals_data = detect_signals(
                    sector_data,
                    lookback=lookback,
                    money_threshold=money_th,
                    price_threshold=price_th
                )

                signal_count = signals_data['signal'].sum()

                # 至少需要10个信号才有意义
                if signal_count < 10:
                    continue

                returns_df = calculate_forward_returns(
                    signals_data, sector_data, periods=[1, 3, 5, 10]
                )

                if len(returns_df) == 0:
                    continue

                # 计算各期指标
                metrics = {}
                for period in [1, 3, 5, 10]:
                    col = f'return_{period}d'
                    if col in returns_df.columns:
                        valid_returns = returns_df[col].dropna()
                        if len(valid_returns) > 0:
                            metrics[f'win_rate_{period}d'] = (valid_returns > 0).sum() / len(valid_returns) * 100
                            metrics[f'avg_return_{period}d'] = valid_returns.mean()
                            metrics[f'sharpe_{period}d'] = valid_returns.mean() / (valid_returns.std() + 1e-6)

                result = {
                    'lookback': lookback,
                    'money_threshold': money_th,
                    'price_threshold': price_th,
                    'signal_count': signal_count,
                    **metrics
                }

                results.append(result)

                if test_count % 10 == 0:
                    print(f"进度: {test_count}/{total_tests}")

    reader.close()

    if not results:
        print("\n未找到有效的参数组合")
        return

    results_df = pd.DataFrame(results)

    # 保存完整结果
    output_dir = Path('output/param_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'all_params.csv', index=False, encoding='utf-8-sig')

    # 显示结果
    print("\n" + "=" * 80)
    print("参数优化结果")
    print("=" * 80)

    print("\n【按5日平均收益排序 Top 10】")
    top_by_return = results_df.nlargest(10, 'avg_return_5d')[
        ['lookback', 'money_threshold', 'price_threshold', 'signal_count',
         'win_rate_5d', 'avg_return_5d', 'sharpe_5d']
    ]
    print(top_by_return.to_string(index=False))

    print("\n【按5日夏普比率排序 Top 10】")
    top_by_sharpe = results_df.nlargest(10, 'sharpe_5d')[
        ['lookback', 'money_threshold', 'price_threshold', 'signal_count',
         'win_rate_5d', 'avg_return_5d', 'sharpe_5d']
    ]
    print(top_by_sharpe.to_string(index=False))

    print("\n【按10日平均收益排序 Top 10】")
    if 'avg_return_10d' in results_df.columns:
        top_by_return_10d = results_df.nlargest(10, 'avg_return_10d')[
            ['lookback', 'money_threshold', 'price_threshold', 'signal_count',
             'win_rate_10d', 'avg_return_10d', 'sharpe_10d']
        ]
        print(top_by_return_10d.to_string(index=False))

    print(f"\n详细结果已保存至: {output_dir}")

    # 最佳参数建议
    best_params = results_df.loc[results_df['avg_return_5d'].idxmax()]
    print("\n" + "=" * 80)
    print("🏆 最佳参数组合（基于5日平均收益）")
    print("=" * 80)
    print(f"  回看周期: {best_params['lookback']}")
    print(f"  资金阈值: {best_params['money_threshold']}")
    print(f"  价格阈值: {best_params['price_threshold']}")
    print(f"  信号数量: {int(best_params['signal_count'])}")
    print(f"  5日胜率: {best_params['win_rate_5d']:.2f}%")
    print(f"  5日平均收益: {best_params['avg_return_5d']:.2f}%")
    print(f"  5日夏普比率: {best_params['sharpe_5d']:.3f}")


if __name__ == '__main__':
    main()
