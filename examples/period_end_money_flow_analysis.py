"""
期末资金流入与未来收益分析

分析逻辑：
1. 识别"期末"交易日：每周五，或节假日前最后一个交易日
2. 统计期末资金流入排名前N的板块（分主力资金和总资金两个榜单）
3. 计算这些板块在未来1、3、5个交易日的涨跌幅统计量

数据来源：
- moneyflow: 个股资金流向（聚合到板块）
- index_member_all: 申万行业成分股
- trade_cal: 交易日历（判断期末）
- sw_daily 或 daily: 板块/个股涨跌幅
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


class PeriodEndMoneyFlowAnalyzer:
    """期末资金流分析器"""

    def __init__(self, db_path: str = 'tushare.db'):
        self.reader = DataReader(db_path)

    def get_period_end_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取期末交易日列表

        期末定义：
        1. 每周五（如果是交易日）
        2. 节假日前最后一个交易日

        Returns:
            DataFrame with columns: trade_date, period_type ('weekly' or 'holiday')
        """
        # 获取所有交易日
        query = """
        SELECT cal_date, is_open, pretrade_date
        FROM trade_cal
        WHERE cal_date >= ? AND cal_date <= ?
        AND exchange = 'SSE'
        ORDER BY cal_date
        """
        cal_df = self.reader.db.con.execute(query, [start_date, end_date]).fetchdf()

        # 只保留交易日
        trading_days = cal_df[cal_df['is_open'] == 1]['cal_date'].tolist()

        period_end_dates = []

        for i, trade_date in enumerate(trading_days):
            dt = datetime.strptime(trade_date, '%Y%m%d')
            is_period_end = False
            period_type = None

            # 检查是否是周五
            if dt.weekday() == 4:  # 周五
                is_period_end = True
                period_type = 'weekly'

            # 检查是否是节假日前最后一个交易日
            # 如果下一个交易日不是下一个自然日，说明中间有休市
            if i < len(trading_days) - 1:
                next_trade_date = trading_days[i + 1]
                next_dt = datetime.strptime(next_trade_date, '%Y%m%d')
                days_gap = (next_dt - dt).days

                # 如果间隔超过3天（排除普通周末），认为是节假日
                if days_gap > 3:
                    is_period_end = True
                    period_type = 'holiday'
                # 如果是周四但下周一才开市（长周末），也算期末
                elif dt.weekday() == 3 and days_gap > 1 and next_dt.weekday() == 0:
                    is_period_end = True
                    period_type = 'holiday'

            if is_period_end:
                period_end_dates.append({
                    'trade_date': trade_date,
                    'period_type': period_type
                })

        return pd.DataFrame(period_end_dates)

    def get_sector_money_flow(self, trade_date: str, level: str = 'L1') -> pd.DataFrame:
        """
        获取某日的板块资金流入数据

        Args:
            trade_date: 交易日期
            level: 板块层级 ('L1' 或 'L3')

        Returns:
            DataFrame with columns:
            - sector_code, sector_name
            - net_mf_amount: 总资金净流入
            - net_main_amount: 主力资金净流入（大单+超大单）
            - stock_count: 成分股数量
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
        )
        SELECT
            s.sector_code,
            s.sector_name,
            SUM(m.net_mf_amount) as net_mf_amount,
            SUM(m.buy_elg_amount + m.buy_lg_amount - m.sell_elg_amount - m.sell_lg_amount) as net_main_amount,
            COUNT(DISTINCT m.ts_code) as stock_count
        FROM moneyflow m
        INNER JOIN sector_stocks s ON m.ts_code = s.ts_code
        WHERE m.trade_date = ?
        GROUP BY s.sector_code, s.sector_name
        ORDER BY net_mf_amount DESC
        """

        return self.reader.db.con.execute(query, [trade_date]).fetchdf()

    def get_sector_returns(self, trade_date: str, periods: list, level: str = 'L1') -> pd.DataFrame:
        """
        获取某日起的板块未来N日收益

        Args:
            trade_date: 起始日期
            periods: 持有期列表，如 [1, 3, 5]
            level: 板块层级

        Returns:
            DataFrame with sector returns for each period
        """
        if level == 'L1':
            code_col = 'l1_code'
            name_col = 'l1_name'
        else:
            code_col = 'l3_code'
            name_col = 'l3_name'

        # 获取未来交易日
        max_period = max(periods)
        future_dates_query = """
        SELECT cal_date FROM trade_cal
        WHERE cal_date > ? AND is_open = 1
        ORDER BY cal_date
        LIMIT ?
        """
        future_dates_df = self.reader.db.con.execute(
            future_dates_query, [trade_date, max_period]
        ).fetchdf()

        if len(future_dates_df) < max_period:
            return pd.DataFrame()

        future_dates = future_dates_df['cal_date'].tolist()

        # 获取板块每日收益
        all_dates = [trade_date] + future_dates
        start_d = all_dates[0]
        end_d = all_dates[-1]

        query = f"""
        WITH sector_stocks AS (
            SELECT DISTINCT
                ts_code,
                {code_col} as sector_code,
                {name_col} as sector_name
            FROM index_member_all
            WHERE is_new = 'Y'
            AND {code_col} IS NOT NULL
        )
        SELECT
            d.trade_date,
            s.sector_code,
            s.sector_name,
            AVG(d.pct_chg) as avg_pct_chg
        FROM daily d
        INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
        WHERE d.trade_date >= ? AND d.trade_date <= ?
        GROUP BY d.trade_date, s.sector_code, s.sector_name
        ORDER BY d.trade_date, s.sector_code
        """

        returns_df = self.reader.db.con.execute(query, [start_d, end_d]).fetchdf()

        if returns_df.empty:
            return pd.DataFrame()

        # 计算各期累计收益
        results = []
        for sector in returns_df['sector_code'].unique():
            sector_data = returns_df[returns_df['sector_code'] == sector].copy()
            sector_data = sector_data.sort_values('trade_date')
            sector_name = sector_data['sector_name'].iloc[0]

            # 确保起始日在数据中
            if trade_date not in sector_data['trade_date'].values:
                continue

            result = {
                'sector_code': sector,
                'sector_name': sector_name
            }

            # 计算各期收益
            for period in periods:
                if period <= len(future_dates):
                    # 获取未来N日的收益
                    target_dates = future_dates[:period]
                    period_returns = sector_data[
                        sector_data['trade_date'].isin(target_dates)
                    ]['avg_pct_chg']

                    if len(period_returns) == period:
                        result[f'return_{period}d'] = period_returns.sum()
                    else:
                        result[f'return_{period}d'] = None
                else:
                    result[f'return_{period}d'] = None

            results.append(result)

        return pd.DataFrame(results)

    def analyze_period_end_performance(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 5,
        level: str = 'L1',
        periods: list = [1, 3, 5]
    ) -> dict:
        """
        分析期末资金流入前N板块的未来表现

        Args:
            start_date: 开始日期
            end_date: 结束日期
            top_n: 取前N名
            level: 板块层级
            periods: 持有期列表

        Returns:
            dict with 'main_flow' and 'total_flow' analysis results
        """
        # 获取期末日期
        period_end_df = self.get_period_end_dates(start_date, end_date)
        print(f"找到 {len(period_end_df)} 个期末交易日")

        main_flow_results = []  # 主力资金榜
        total_flow_results = []  # 总资金榜

        for _, row in period_end_df.iterrows():
            trade_date = row['trade_date']
            period_type = row['period_type']

            # 获取当日资金流
            money_flow = self.get_sector_money_flow(trade_date, level)

            if money_flow.empty:
                continue

            # 获取未来收益
            returns_df = self.get_sector_returns(trade_date, periods, level)

            if returns_df.empty:
                continue

            # 合并数据
            merged = money_flow.merge(returns_df, on=['sector_code', 'sector_name'], how='inner')

            if merged.empty:
                continue

            # 主力资金榜 Top N
            main_top = merged.nlargest(top_n, 'net_main_amount').copy()
            main_top['trade_date'] = trade_date
            main_top['period_type'] = period_type
            main_top['rank'] = range(1, len(main_top) + 1)
            main_flow_results.append(main_top)

            # 总资金榜 Top N
            total_top = merged.nlargest(top_n, 'net_mf_amount').copy()
            total_top['trade_date'] = trade_date
            total_top['period_type'] = period_type
            total_top['rank'] = range(1, len(total_top) + 1)
            total_flow_results.append(total_top)

        return {
            'main_flow': pd.concat(main_flow_results, ignore_index=True) if main_flow_results else pd.DataFrame(),
            'total_flow': pd.concat(total_flow_results, ignore_index=True) if total_flow_results else pd.DataFrame()
        }

    def calculate_statistics(self, df: pd.DataFrame, periods: list) -> dict:
        """
        计算统计量

        Args:
            df: 包含收益数据的 DataFrame
            periods: 持有期列表

        Returns:
            统计结果字典
        """
        stats = {}

        for period in periods:
            col = f'return_{period}d'
            if col not in df.columns:
                continue

            returns = df[col].dropna()

            if len(returns) == 0:
                continue

            stats[f'{period}d'] = {
                '样本数': len(returns),
                '平均收益': returns.mean(),
                '中位数': returns.median(),
                '标准差': returns.std(),
                '胜率': (returns > 0).sum() / len(returns) * 100,
                '最大值': returns.max(),
                '最小值': returns.min(),
                '夏普比率': returns.mean() / (returns.std() + 1e-6),
                '正收益均值': returns[returns > 0].mean() if (returns > 0).any() else 0,
                '负收益均值': returns[returns < 0].mean() if (returns < 0).any() else 0,
            }

        return stats

    def close(self):
        self.reader.close()


def main():
    print("=" * 80)
    print("期末资金流入与未来收益分析")
    print("=" * 80)

    analyzer = PeriodEndMoneyFlowAnalyzer('tushare.db')

    # 分析参数
    start_date = '20220101'
    end_date = '20241231'
    top_n = 5  # 取前5名
    level = 'L1'  # L1板块
    periods = [1, 3, 5]  # 未来1、3、5个交易日

    print(f"\n分析期间: {start_date} ~ {end_date}")
    print(f"板块层级: {level}")
    print(f"榜单取前: {top_n} 名")
    print(f"持有期: {periods} 交易日")

    # 先检查期末日期
    print("\n[1/3] 识别期末交易日...")
    period_end_df = analyzer.get_period_end_dates(start_date, end_date)
    print(f"      共 {len(period_end_df)} 个期末交易日")

    weekly_count = len(period_end_df[period_end_df['period_type'] == 'weekly'])
    holiday_count = len(period_end_df[period_end_df['period_type'] == 'holiday'])
    print(f"      - 周五: {weekly_count} 个")
    print(f"      - 节假日前: {holiday_count} 个")

    # 显示部分期末日期示例
    print("\n      最近10个期末日期:")
    recent = period_end_df.tail(10)
    for _, row in recent.iterrows():
        print(f"        {row['trade_date']} ({row['period_type']})")

    # 执行分析
    print("\n[2/3] 分析资金流入与未来收益...")
    results = analyzer.analyze_period_end_performance(
        start_date, end_date, top_n, level, periods
    )

    main_df = results['main_flow']
    total_df = results['total_flow']

    print(f"      主力资金榜: {len(main_df)} 条记录")
    print(f"      总资金榜: {len(total_df)} 条记录")

    # 计算统计量
    print("\n[3/3] 计算统计量...")

    # ============ 主力资金榜统计 ============
    print("\n" + "=" * 80)
    print("【主力资金净流入榜】Top {} 板块未来收益统计".format(top_n))
    print("=" * 80)

    main_stats = analyzer.calculate_statistics(main_df, periods)

    for period_key, stats in main_stats.items():
        print(f"\n  【{period_key}持有期】")
        print(f"    样本数: {stats['样本数']}")
        print(f"    平均收益: {stats['平均收益']:.2f}%")
        print(f"    中位数: {stats['中位数']:.2f}%")
        print(f"    标准差: {stats['标准差']:.2f}%")
        print(f"    胜率: {stats['胜率']:.1f}%")
        print(f"    夏普比率: {stats['夏普比率']:.3f}")
        print(f"    收益区间: [{stats['最小值']:.2f}%, {stats['最大值']:.2f}%]")
        print(f"    盈利均值: {stats['正收益均值']:.2f}%  |  亏损均值: {stats['负收益均值']:.2f}%")

    # ============ 总资金榜统计 ============
    print("\n" + "=" * 80)
    print("【总资金净流入榜】Top {} 板块未来收益统计".format(top_n))
    print("=" * 80)

    total_stats = analyzer.calculate_statistics(total_df, periods)

    for period_key, stats in total_stats.items():
        print(f"\n  【{period_key}持有期】")
        print(f"    样本数: {stats['样本数']}")
        print(f"    平均收益: {stats['平均收益']:.2f}%")
        print(f"    中位数: {stats['中位数']:.2f}%")
        print(f"    标准差: {stats['标准差']:.2f}%")
        print(f"    胜率: {stats['胜率']:.1f}%")
        print(f"    夏普比率: {stats['夏普比率']:.3f}")
        print(f"    收益区间: [{stats['最小值']:.2f}%, {stats['最大值']:.2f}%]")
        print(f"    盈利均值: {stats['正收益均值']:.2f}%  |  亏损均值: {stats['负收益均值']:.2f}%")

    # ============ 按排名分层统计 ============
    print("\n" + "=" * 80)
    print("按排名分层统计（主力资金榜）")
    print("=" * 80)

    for rank in range(1, top_n + 1):
        rank_df = main_df[main_df['rank'] == rank]
        if len(rank_df) == 0:
            continue

        rank_stats = analyzer.calculate_statistics(rank_df, periods)
        print(f"\n  第 {rank} 名:")
        for period_key, stats in rank_stats.items():
            print(f"    {period_key}: 平均{stats['平均收益']:.2f}%, 胜率{stats['胜率']:.1f}%, 夏普{stats['夏普比率']:.3f} (n={stats['样本数']})")

    # ============ 周五 vs 节假日前 对比 ============
    print("\n" + "=" * 80)
    print("周五 vs 节假日前 对比（主力资金榜）")
    print("=" * 80)

    for period_type, type_name in [('weekly', '周五'), ('holiday', '节假日前')]:
        type_df = main_df[main_df['period_type'] == period_type]
        if len(type_df) == 0:
            continue

        type_stats = analyzer.calculate_statistics(type_df, periods)
        print(f"\n  【{type_name}】")
        for period_key, stats in type_stats.items():
            print(f"    {period_key}: 平均{stats['平均收益']:.2f}%, 胜率{stats['胜率']:.1f}%, 夏普{stats['夏普比率']:.3f} (n={stats['样本数']})")

    # ============ 板块出现频次统计 ============
    print("\n" + "=" * 80)
    print("板块上榜频次统计（主力资金榜 Top 10）")
    print("=" * 80)

    sector_freq = main_df.groupby('sector_name').agg({
        'trade_date': 'count',
        'return_1d': 'mean',
        'return_3d': 'mean',
        'return_5d': 'mean'
    }).rename(columns={'trade_date': '上榜次数'})

    sector_freq = sector_freq.sort_values('上榜次数', ascending=False).head(10)
    print("\n" + sector_freq.to_string())

    # 保存详细结果
    output_dir = Path('output/period_end_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df.to_csv(output_dir / 'main_flow_details.csv', index=False, encoding='utf-8-sig')
    total_df.to_csv(output_dir / 'total_flow_details.csv', index=False, encoding='utf-8-sig')

    print(f"\n详细结果已保存至: {output_dir}")

    # 总结
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)

    if main_stats and '5d' in main_stats:
        main_5d = main_stats['5d']
        total_5d = total_stats.get('5d', {})

        print(f"\n主力资金榜 Top{top_n} (5日持有):")
        print(f"  - 平均收益: {main_5d['平均收益']:.2f}%")
        print(f"  - 胜率: {main_5d['胜率']:.1f}%")
        print(f"  - 夏普比率: {main_5d['夏普比率']:.3f}")

        if total_5d:
            print(f"\n总资金榜 Top{top_n} (5日持有):")
            print(f"  - 平均收益: {total_5d['平均收益']:.2f}%")
            print(f"  - 胜率: {total_5d['胜率']:.1f}%")
            print(f"  - 夏普比率: {total_5d['夏普比率']:.3f}")

        # 简单结论
        if main_5d['平均收益'] > 0.5 and main_5d['胜率'] > 55:
            print("\n结论: 期末主力资金流入前排板块在未来5日有正向超额收益")
        elif main_5d['平均收益'] < -0.5:
            print("\n结论: 期末资金流入可能是短期高点，未来5日表现偏弱")
        else:
            print("\n结论: 期末资金流入与未来收益无明显规律")

    analyzer.close()


if __name__ == '__main__':
    main()
