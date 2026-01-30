"""
板块特征全景分析

目标：理解板块的内在特征，而非预测涨跌
- 波动性特征（高波动/低波动）
- 市场相关性（Beta高低）
- 资金偏好（机构/散户）
- 周期性（顺周期/逆周期）
- 防御性（牛市/熊市表现）
- 领先/滞后性（对市场变化的响应速度）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


class SectorCharacteristicsAnalyzer:
    """板块特征分析器"""

    def __init__(self, db_path: str):
        self.reader = DataReader(db_path)

    def get_sector_returns(self, start_date: str, end_date: str, level: str = 'L1'):
        """获取板块收益率数据"""
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
        daily_returns AS (
            SELECT
                d.trade_date,
                s.sector_code,
                s.sector_name,
                d.ts_code,
                d.pct_chg
            FROM daily d
            INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
            WHERE d.trade_date >= '{start_date}'
            AND d.trade_date <= '{end_date}'
            AND d.pct_chg IS NOT NULL
        )
        SELECT
            trade_date,
            sector_code,
            sector_name,
            AVG(pct_chg) as avg_pct_chg,
            COUNT(DISTINCT ts_code) as stock_count
        FROM daily_returns
        GROUP BY trade_date, sector_code, sector_name
        ORDER BY trade_date, sector_code
        """

        return self.reader.db.con.execute(query).fetchdf()

    def get_market_returns(self, start_date: str, end_date: str):
        """获取市场收益率（使用沪深300）"""
        query = f"""
        SELECT
            trade_date,
            pct_chg as market_return
        FROM daily
        WHERE ts_code = '000300.SH'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """

        df = self.reader.db.con.execute(query).fetchdf()

        if len(df) == 0:
            print("⚠️  未找到沪深300数据，使用全市场平均")
            # 使用全市场平均
            query = f"""
            SELECT
                trade_date,
                AVG(pct_chg) as market_return
            FROM daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            AND pct_chg IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date
            """
            df = self.reader.db.con.execute(query).fetchdf()

        return df

    def calculate_volatility_characteristics(self, sector_returns: pd.DataFrame):
        """
        计算波动性特征
        - 收益率标准差
        - 最大回撤
        - 上涨日占比
        """
        results = []

        for sector in sector_returns['sector_code'].unique():
            sector_data = sector_returns[sector_returns['sector_code'] == sector].copy()
            sector_data = sector_data.sort_values('trade_date')

            returns = sector_data['avg_pct_chg'].values
            sector_name = sector_data['sector_name'].iloc[0]

            # 波动率
            volatility = np.std(returns)

            # 最大回撤
            cumulative = (1 + returns / 100).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()

            # 上涨日占比
            up_ratio = (returns > 0).sum() / len(returns) * 100

            # 年化收益
            total_days = len(returns)
            annualized_return = ((1 + returns / 100).prod() ** (252 / total_days) - 1) * 100

            results.append({
                'sector_code': sector,
                'sector_name': sector_name,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'up_ratio': up_ratio,
                'annualized_return': annualized_return
            })

        return pd.DataFrame(results)

    def calculate_market_sensitivity(self, sector_returns: pd.DataFrame, market_returns: pd.DataFrame):
        """
        计算市场敏感性（Beta）
        - Beta系数
        - R²（与市场的相关性）
        - Alpha（超额收益）
        """
        # 合并数据
        market_df = market_returns[['trade_date', 'market_return']].copy()

        results = []

        for sector in sector_returns['sector_code'].unique():
            sector_data = sector_returns[sector_returns['sector_code'] == sector].copy()
            sector_name = sector_data['sector_name'].iloc[0]

            # 合并
            merged = sector_data[['trade_date', 'avg_pct_chg']].merge(
                market_df, on='trade_date', how='inner'
            )

            if len(merged) < 20:
                continue

            # 计算Beta和Alpha
            X = merged['market_return'].values
            y = merged['avg_pct_chg'].values

            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            beta = slope
            alpha = intercept
            r_squared = r_value ** 2

            results.append({
                'sector_code': sector,
                'sector_name': sector_name,
                'beta': beta,
                'alpha': alpha,
                'r_squared': r_squared
            })

        return pd.DataFrame(results)

    def calculate_money_flow_characteristics(self, start_date: str, end_date: str, level: str = 'L1'):
        """
        计算资金流特征
        - 大单占比
        - 机构资金偏好
        """
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
                s.sector_code,
                s.sector_name,
                AVG((m.buy_lg_amount + m.buy_elg_amount) /
                    (m.buy_sm_amount + m.buy_md_amount + m.buy_lg_amount + m.buy_elg_amount + 1)) as avg_large_buy_ratio,
                AVG(m.net_mf_amount) as avg_net_flow,
                AVG((m.buy_lg_amount - m.sell_lg_amount) /
                    (m.buy_lg_amount + m.sell_lg_amount + 1)) as avg_large_net_ratio
            FROM moneyflow m
            INNER JOIN sector_stocks s ON m.ts_code = s.ts_code
            WHERE m.trade_date >= '{start_date}'
            AND m.trade_date <= '{end_date}'
            GROUP BY s.sector_code, s.sector_name
        )
        SELECT * FROM sector_money
        ORDER BY sector_code
        """

        return self.reader.db.con.execute(query).fetchdf()

    def classify_sectors(
        self,
        volatility_df: pd.DataFrame,
        sensitivity_df: pd.DataFrame,
        money_df: pd.DataFrame
    ):
        """
        板块分类
        """
        # 合并数据
        df = volatility_df.merge(sensitivity_df, on=['sector_code', 'sector_name'], how='inner')
        df = df.merge(money_df, on=['sector_code', 'sector_name'], how='left')

        # 分类标准
        df['type'] = ''

        # 1. 高Beta + 高波动 = 进攻型
        df.loc[(df['beta'] > 1.1) & (df['volatility'] > df['volatility'].median()), 'type'] = '进攻型'

        # 2. 低Beta + 低波动 = 防御型
        df.loc[(df['beta'] < 0.9) & (df['volatility'] < df['volatility'].median()), 'type'] = '防御型'

        # 3. 高Beta + 低相关性 = 独立型
        df.loc[(df['beta'] > 1.0) & (df['r_squared'] < 0.5), 'type'] = '独立型'

        # 4. 低Beta + 正Alpha = 稳健增长型
        df.loc[(df['beta'] < 1.0) & (df['alpha'] > 0), 'type'] = '稳健增长型'

        # 5. 其他 = 中性型
        df.loc[df['type'] == '', 'type'] = '中性型'

        return df

    def analyze_market_regimes(self, sector_returns: pd.DataFrame, market_returns: pd.DataFrame):
        """
        分析不同市场环境下的板块表现
        - 牛市（市场涨幅>1%）
        - 熊市（市场跌幅<-1%）
        - 震荡市（-1% ~ 1%）
        """
        market_df = market_returns[['trade_date', 'market_return']].copy()

        # 定义市场状态
        market_df['regime'] = 'sideways'
        market_df.loc[market_df['market_return'] > 1, 'regime'] = 'bull'
        market_df.loc[market_df['market_return'] < -1, 'regime'] = 'bear'

        results = []

        for sector in sector_returns['sector_code'].unique():
            sector_data = sector_returns[sector_returns['sector_code'] == sector].copy()
            sector_name = sector_data['sector_name'].iloc[0]

            # 合并市场状态
            merged = sector_data.merge(market_df, on='trade_date', how='inner')

            # 计算各状态下的平均收益
            regime_performance = merged.groupby('regime')['avg_pct_chg'].agg(['mean', 'count'])

            bull_return = regime_performance.loc['bull', 'mean'] if 'bull' in regime_performance.index else 0
            bear_return = regime_performance.loc['bear', 'mean'] if 'bear' in regime_performance.index else 0
            sideways_return = regime_performance.loc['sideways', 'mean'] if 'sideways' in regime_performance.index else 0

            # 牛熊比率（牛市收益/熊市亏损的绝对值）
            bull_bear_ratio = bull_return / abs(bear_return) if bear_return != 0 else 0

            results.append({
                'sector_code': sector,
                'sector_name': sector_name,
                'bull_return': bull_return,
                'bear_return': bear_return,
                'sideways_return': sideways_return,
                'bull_bear_ratio': bull_bear_ratio
            })

        return pd.DataFrame(results)

    def close(self):
        self.reader.close()


def main():
    print("=" * 80)
    print("板块特征全景分析")
    print("=" * 80)

    analyzer = SectorCharacteristicsAnalyzer('tushare.db')

    # 分析近3年数据
    start_date = '20220101'
    end_date = '20241231'

    print(f"\n分析期间: {start_date} ~ {end_date}")

    # 1. 获取数据
    print("\n[1/6] 获取板块收益数据...")
    sector_returns = analyzer.get_sector_returns(start_date, end_date, 'L1')
    print(f"      数据量: {len(sector_returns)} 条")

    print("\n[2/6] 获取市场收益数据...")
    market_returns = analyzer.get_market_returns(start_date, end_date)
    print(f"      数据量: {len(market_returns)} 条")

    # 2. 计算波动性特征
    print("\n[3/6] 计算波动性特征...")
    volatility_df = analyzer.calculate_volatility_characteristics(sector_returns)

    # 3. 计算市场敏感性
    print("\n[4/6] 计算市场敏感性（Beta）...")
    sensitivity_df = analyzer.calculate_market_sensitivity(sector_returns, market_returns)

    # 4. 计算资金流特征
    print("\n[5/6] 计算资金流特征...")
    money_df = analyzer.calculate_money_flow_characteristics(start_date, end_date, 'L1')

    # 5. 市场环境分析
    print("\n[6/6] 分析不同市场环境表现...")
    regime_df = analyzer.analyze_market_regimes(sector_returns, market_returns)

    # 6. 板块分类
    print("\n分析完成，正在生成报告...")
    classified_df = analyzer.classify_sectors(volatility_df, sensitivity_df, money_df)

    # 合并市场环境数据
    final_df = classified_df.merge(
        regime_df[['sector_code', 'bull_return', 'bear_return', 'bull_bear_ratio']],
        on='sector_code',
        how='left'
    )

    # ============ 输出报告 ============

    print("\n" + "=" * 80)
    print("板块分类结果")
    print("=" * 80)

    for sector_type in ['进攻型', '防御型', '稳健增长型', '独立型', '中性型']:
        type_data = final_df[final_df['type'] == sector_type]

        if len(type_data) == 0:
            continue

        print(f"\n【{sector_type}】({len(type_data)}个)")
        print(type_data[['sector_name', 'beta', 'volatility', 'annualized_return']].to_string(index=False))

    # 波动性排名
    print("\n" + "=" * 80)
    print("波动性排名 Top 10")
    print("=" * 80)
    top_volatile = final_df.nlargest(10, 'volatility')[
        ['sector_name', 'volatility', 'max_drawdown', 'annualized_return']
    ]
    print(top_volatile.to_string(index=False))

    # Beta排名
    print("\n" + "=" * 80)
    print("市场敏感性（Beta）排名")
    print("=" * 80)
    print("\n高Beta（进攻型） Top 10:")
    high_beta = final_df.nlargest(10, 'beta')[['sector_name', 'beta', 'r_squared', 'alpha']]
    print(high_beta.to_string(index=False))

    print("\n低Beta（防御型） Top 10:")
    low_beta = final_df.nsmallest(10, 'beta')[['sector_name', 'beta', 'r_squared', 'alpha']]
    print(low_beta.to_string(index=False))

    # 牛熊市表现
    print("\n" + "=" * 80)
    print("牛熊市表现对比")
    print("=" * 80)
    print("\n牛市表现最佳 Top 10:")
    bull_best = final_df.nlargest(10, 'bull_return')[
        ['sector_name', 'bull_return', 'bear_return', 'bull_bear_ratio']
    ]
    print(bull_best.to_string(index=False))

    print("\n熊市抗跌能力最强 Top 10:")
    bear_best = final_df.nlargest(10, 'bear_return')[
        ['sector_name', 'bull_return', 'bear_return', 'bull_bear_ratio']
    ]
    print(bear_best.to_string(index=False))

    # 资金偏好
    print("\n" + "=" * 80)
    print("机构资金偏好")
    print("=" * 80)
    print("\n大单占比最高 Top 10:")
    large_money = final_df.nlargest(10, 'avg_large_buy_ratio')[
        ['sector_name', 'avg_large_buy_ratio', 'avg_large_net_ratio', 'beta']
    ]
    print(large_money.to_string(index=False))

    # 保存完整结果
    output_dir = Path('output/sector_characteristics')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_dir / 'sector_characteristics.csv', index=False, encoding='utf-8-sig')
    print(f"\n完整分析结果已保存至: {output_dir}")

    # 总结
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)

    print(f"\n板块分类统计:")
    type_counts = final_df['type'].value_counts()
    for sector_type, count in type_counts.items():
        print(f"  {sector_type}: {count}个")

    print(f"\n市场特征:")
    print(f"  平均Beta: {final_df['beta'].mean():.2f}")
    print(f"  平均波动率: {final_df['volatility'].mean():.2f}%")
    print(f"  平均年化收益: {final_df['annualized_return'].mean():.2f}%")

    analyzer.close()


if __name__ == '__main__':
    main()
