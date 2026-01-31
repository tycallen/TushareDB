"""
期末 vs 每日 资金流入信号对比分析

目标：
1. 对比期末（周五/节假日前）与普通交易日的资金流入信号效果
2. 使用超额收益（板块收益 - 市场指数收益）代替绝对收益
3. 分析期末是否有额外信号价值
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


class PeriodEndVsDailyAnalyzer:
    """期末 vs 每日 资金流入分析器"""

    def __init__(self, db_path: str = 'tushare.db'):
        self.reader = DataReader(db_path)
        self._market_returns_cache = None

    def get_market_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取市场指数收益率（用于计算超额收益）

        使用全市场等权平均作为基准
        """
        if self._market_returns_cache is not None:
            return self._market_returns_cache

        query = """
        SELECT
            trade_date,
            AVG(pct_chg) as market_return
        FROM daily
        WHERE trade_date >= ? AND trade_date <= ?
        AND pct_chg IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """

        df = self.reader.db.con.execute(query, [start_date, end_date]).fetchdf()
        self._market_returns_cache = df
        return df

    def classify_trading_days(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        对所有交易日进行分类

        分类：
        - period_end_holiday: 节假日前最后一个交易日
        - period_end_friday: 普通周五
        - normal: 普通交易日（非周五、非节假日前）
        """
        # 获取所有交易日
        query = """
        SELECT cal_date, is_open
        FROM trade_cal
        WHERE cal_date >= ? AND cal_date <= ?
        AND exchange = 'SSE'
        ORDER BY cal_date
        """
        cal_df = self.reader.db.con.execute(query, [start_date, end_date]).fetchdf()

        trading_days = cal_df[cal_df['is_open'] == 1]['cal_date'].tolist()

        results = []

        for i, trade_date in enumerate(trading_days):
            dt = datetime.strptime(trade_date, '%Y%m%d')
            day_type = 'normal'

            # 检查是否是节假日前
            if i < len(trading_days) - 1:
                next_trade_date = trading_days[i + 1]
                next_dt = datetime.strptime(next_trade_date, '%Y%m%d')
                days_gap = (next_dt - dt).days

                if days_gap > 3:
                    day_type = 'period_end_holiday'
                elif dt.weekday() == 3 and days_gap > 1 and next_dt.weekday() == 0:
                    day_type = 'period_end_holiday'
                elif dt.weekday() == 4 and day_type != 'period_end_holiday':
                    day_type = 'period_end_friday'
            elif dt.weekday() == 4:
                day_type = 'period_end_friday'

            results.append({
                'trade_date': trade_date,
                'day_type': day_type,
                'weekday': dt.weekday()
            })

        return pd.DataFrame(results)

    def get_sector_money_flow_batch(
        self,
        trade_dates: list,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        批量获取多个日期的板块资金流数据
        """
        if level == 'L1':
            code_col = 'l1_code'
            name_col = 'l1_name'
        else:
            code_col = 'l3_code'
            name_col = 'l3_name'

        dates_str = "','".join(trade_dates)

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
            m.trade_date,
            s.sector_code,
            s.sector_name,
            SUM(m.net_mf_amount) as net_mf_amount,
            SUM(m.buy_elg_amount + m.buy_lg_amount - m.sell_elg_amount - m.sell_lg_amount) as net_main_amount,
            COUNT(DISTINCT m.ts_code) as stock_count
        FROM moneyflow m
        INNER JOIN sector_stocks s ON m.ts_code = s.ts_code
        WHERE m.trade_date IN ('{dates_str}')
        GROUP BY m.trade_date, s.sector_code, s.sector_name
        ORDER BY m.trade_date, net_main_amount DESC
        """

        return self.reader.db.con.execute(query).fetchdf()

    def get_sector_returns_batch(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        批量获取板块日收益率
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
            d.trade_date,
            s.sector_code,
            s.sector_name,
            AVG(d.pct_chg) as avg_pct_chg
        FROM daily d
        INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
        WHERE d.trade_date >= ? AND d.trade_date <= ?
        AND d.pct_chg IS NOT NULL
        GROUP BY d.trade_date, s.sector_code, s.sector_name
        ORDER BY d.trade_date, s.sector_code
        """

        return self.reader.db.con.execute(query, [start_date, end_date]).fetchdf()

    def calculate_forward_excess_returns(
        self,
        sector_returns: pd.DataFrame,
        market_returns: pd.DataFrame,
        trade_date: str,
        sector_code: str,
        periods: list
    ) -> dict:
        """
        计算某板块从某日起的未来N日超额收益
        """
        # 获取该板块的收益序列
        sector_data = sector_returns[
            sector_returns['sector_code'] == sector_code
        ].sort_values('trade_date')

        # 获取trade_date之后的日期
        future_dates = sector_data[sector_data['trade_date'] > trade_date]['trade_date'].tolist()

        results = {}

        for period in periods:
            if len(future_dates) < period:
                results[f'excess_return_{period}d'] = None
                results[f'abs_return_{period}d'] = None
                continue

            target_dates = future_dates[:period]

            # 板块累计收益
            sector_period_returns = sector_data[
                sector_data['trade_date'].isin(target_dates)
            ]['avg_pct_chg'].sum()

            # 市场累计收益
            market_period_returns = market_returns[
                market_returns['trade_date'].isin(target_dates)
            ]['market_return'].sum()

            # 超额收益
            excess_return = sector_period_returns - market_period_returns

            results[f'excess_return_{period}d'] = excess_return
            results[f'abs_return_{period}d'] = sector_period_returns

        return results

    def analyze_all_days(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 5,
        level: str = 'L1',
        periods: list = [1, 3, 5],
        sample_ratio: float = 1.0  # 采样比例，用于加速测试
    ) -> pd.DataFrame:
        """
        分析所有交易日的资金流入信号效果
        """
        print("正在分类交易日...")
        day_types_df = self.classify_trading_days(start_date, end_date)
        all_trade_dates = day_types_df['trade_date'].tolist()

        # 采样
        if sample_ratio < 1.0:
            sample_size = int(len(all_trade_dates) * sample_ratio)
            all_trade_dates = sorted(np.random.choice(all_trade_dates, sample_size, replace=False))
            day_types_df = day_types_df[day_types_df['trade_date'].isin(all_trade_dates)]

        print(f"共 {len(all_trade_dates)} 个交易日待分析")

        # 扩展日期范围以计算未来收益
        max_period = max(periods)
        extended_end = (datetime.strptime(end_date, '%Y%m%d') + timedelta(days=max_period * 2)).strftime('%Y%m%d')

        print("正在加载市场收益数据...")
        market_returns = self.get_market_returns(start_date, extended_end)

        print("正在加载板块收益数据...")
        sector_returns = self.get_sector_returns_batch(start_date, extended_end, level)

        print("正在加载资金流数据...")
        money_flow_df = self.get_sector_money_flow_batch(all_trade_dates, level)

        print("正在计算各日信号效果...")
        all_results = []

        for trade_date in tqdm(all_trade_dates, desc="分析进度"):
            day_type = day_types_df[day_types_df['trade_date'] == trade_date]['day_type'].iloc[0]

            # 获取当日资金流数据
            day_money = money_flow_df[money_flow_df['trade_date'] == trade_date].copy()

            if day_money.empty:
                continue

            # 主力资金榜 Top N
            main_top = day_money.nlargest(top_n, 'net_main_amount')

            for rank, (_, row) in enumerate(main_top.iterrows(), 1):
                sector_code = row['sector_code']
                sector_name = row['sector_name']

                # 计算超额收益
                returns = self.calculate_forward_excess_returns(
                    sector_returns, market_returns,
                    trade_date, sector_code, periods
                )

                result = {
                    'trade_date': trade_date,
                    'day_type': day_type,
                    'rank_type': 'main_flow',
                    'rank': rank,
                    'sector_code': sector_code,
                    'sector_name': sector_name,
                    'net_main_amount': row['net_main_amount'],
                    **returns
                }
                all_results.append(result)

            # 总资金榜 Top N
            total_top = day_money.nlargest(top_n, 'net_mf_amount')

            for rank, (_, row) in enumerate(total_top.iterrows(), 1):
                sector_code = row['sector_code']
                sector_name = row['sector_name']

                returns = self.calculate_forward_excess_returns(
                    sector_returns, market_returns,
                    trade_date, sector_code, periods
                )

                result = {
                    'trade_date': trade_date,
                    'day_type': day_type,
                    'rank_type': 'total_flow',
                    'rank': rank,
                    'sector_code': sector_code,
                    'sector_name': sector_name,
                    'net_mf_amount': row['net_mf_amount'],
                    **returns
                }
                all_results.append(result)

        return pd.DataFrame(all_results)

    def calculate_statistics(self, df: pd.DataFrame, periods: list, use_excess: bool = True) -> dict:
        """计算统计量"""
        stats = {}
        prefix = 'excess_return' if use_excess else 'abs_return'

        for period in periods:
            col = f'{prefix}_{period}d'
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
                '夏普比率': returns.mean() / (returns.std() + 1e-6),
                't统计量': returns.mean() / (returns.std() / np.sqrt(len(returns)) + 1e-6),
            }

        return stats

    def close(self):
        self.reader.close()


def main():
    print("=" * 80)
    print("期末 vs 每日 资金流入信号对比分析（超额收益版）")
    print("=" * 80)

    analyzer = PeriodEndVsDailyAnalyzer('tushare.db')

    # 分析参数
    start_date = '20220101'
    end_date = '20241231'
    top_n = 5
    level = 'L1'
    periods = [1, 3, 5]

    print(f"\n分析期间: {start_date} ~ {end_date}")
    print(f"板块层级: {level}")
    print(f"榜单取前: {top_n} 名")
    print(f"持有期: {periods} 交易日")
    print(f"收益指标: 超额收益（板块收益 - 市场收益）")

    # 执行分析
    print("\n" + "=" * 80)
    results_df = analyzer.analyze_all_days(
        start_date, end_date, top_n, level, periods
    )

    print(f"\n总记录数: {len(results_df)}")

    # 按日期类型统计
    day_type_counts = results_df.groupby('day_type')['trade_date'].nunique()
    print(f"\n交易日分布:")
    print(f"  - 节假日前: {day_type_counts.get('period_end_holiday', 0)} 天")
    print(f"  - 周五: {day_type_counts.get('period_end_friday', 0)} 天")
    print(f"  - 普通日: {day_type_counts.get('normal', 0)} 天")

    # ============ 主力资金榜分析 ============
    main_df = results_df[results_df['rank_type'] == 'main_flow']

    print("\n" + "=" * 80)
    print("【主力资金榜】超额收益统计")
    print("=" * 80)

    # 按日期类型分组统计
    for day_type, type_name in [
        ('period_end_holiday', '节假日前'),
        ('period_end_friday', '周五'),
        ('normal', '普通交易日')
    ]:
        type_df = main_df[main_df['day_type'] == day_type]

        if len(type_df) == 0:
            continue

        stats = analyzer.calculate_statistics(type_df, periods, use_excess=True)

        print(f"\n  【{type_name}】(n={len(type_df)}条记录)")
        for period_key, s in stats.items():
            print(f"    {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, "
                  f"夏普{s['夏普比率']:.3f}, t={s['t统计量']:.2f}")

    # ============ 期末 vs 非期末 对比 ============
    print("\n" + "=" * 80)
    print("【期末 vs 非期末 对比】（主力资金榜）")
    print("=" * 80)

    # 期末 = 节假日前 + 周五
    period_end_df = main_df[main_df['day_type'].isin(['period_end_holiday', 'period_end_friday'])]
    non_period_end_df = main_df[main_df['day_type'] == 'normal']

    print("\n期末（周五+节假日前）:")
    period_end_stats = analyzer.calculate_statistics(period_end_df, periods, use_excess=True)
    for period_key, s in period_end_stats.items():
        print(f"  {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, "
              f"夏普{s['夏普比率']:.3f}, t={s['t统计量']:.2f} (n={s['样本数']})")

    print("\n非期末（普通交易日）:")
    non_period_end_stats = analyzer.calculate_statistics(non_period_end_df, periods, use_excess=True)
    for period_key, s in non_period_end_stats.items():
        print(f"  {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, "
              f"夏普{s['夏普比率']:.3f}, t={s['t统计量']:.2f} (n={s['样本数']})")

    # 计算差异
    print("\n差异（期末 - 非期末）:")
    for period in periods:
        period_key = f'{period}d'
        if period_key in period_end_stats and period_key in non_period_end_stats:
            diff = period_end_stats[period_key]['平均收益'] - non_period_end_stats[period_key]['平均收益']
            print(f"  {period_key}: {diff:+.2f}%")

    # ============ 节假日前单独分析 ============
    print("\n" + "=" * 80)
    print("【节假日前 vs 其他所有日】")
    print("=" * 80)

    holiday_df = main_df[main_df['day_type'] == 'period_end_holiday']
    other_df = main_df[main_df['day_type'] != 'period_end_holiday']

    print("\n节假日前:")
    holiday_stats = analyzer.calculate_statistics(holiday_df, periods, use_excess=True)
    for period_key, s in holiday_stats.items():
        print(f"  {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, "
              f"夏普{s['夏普比率']:.3f}, t={s['t统计量']:.2f} (n={s['样本数']})")

    print("\n其他所有交易日:")
    other_stats = analyzer.calculate_statistics(other_df, periods, use_excess=True)
    for period_key, s in other_stats.items():
        print(f"  {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, "
              f"夏普{s['夏普比率']:.3f}, t={s['t统计量']:.2f} (n={s['样本数']})")

    print("\n差异（节假日前 - 其他）:")
    for period in periods:
        period_key = f'{period}d'
        if period_key in holiday_stats and period_key in other_stats:
            diff = holiday_stats[period_key]['平均收益'] - other_stats[period_key]['平均收益']
            print(f"  {period_key}: {diff:+.2f}%")

    # ============ 绝对收益 vs 超额收益对比 ============
    print("\n" + "=" * 80)
    print("【绝对收益 vs 超额收益 对比】（节假日前，主力资金榜）")
    print("=" * 80)

    print("\n超额收益:")
    for period_key, s in holiday_stats.items():
        print(f"  {period_key}: {s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%")

    print("\n绝对收益:")
    abs_stats = analyzer.calculate_statistics(holiday_df, periods, use_excess=False)
    for period_key, s in abs_stats.items():
        print(f"  {period_key}: {s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%")

    # ============ 总资金榜简要统计 ============
    print("\n" + "=" * 80)
    print("【总资金榜】超额收益统计（简要）")
    print("=" * 80)

    total_df = results_df[results_df['rank_type'] == 'total_flow']

    for day_type, type_name in [
        ('period_end_holiday', '节假日前'),
        ('period_end_friday', '周五'),
        ('normal', '普通交易日')
    ]:
        type_df = total_df[total_df['day_type'] == day_type]

        if len(type_df) == 0:
            continue

        stats = analyzer.calculate_statistics(type_df, periods, use_excess=True)

        print(f"\n  【{type_name}】")
        for period_key, s in stats.items():
            print(f"    {period_key}: 超额{s['平均收益']:+.2f}%, 胜率{s['胜率']:.1f}%, t={s['t统计量']:.2f}")

    # 保存结果
    output_dir = Path('output/period_end_vs_daily')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'all_results.csv', index=False, encoding='utf-8-sig')

    # ============ 总结 ============
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)

    # 提取关键指标
    if holiday_stats and other_stats and '5d' in holiday_stats and '5d' in other_stats:
        holiday_5d = holiday_stats['5d']
        other_5d = other_stats['5d']
        diff_5d = holiday_5d['平均收益'] - other_5d['平均收益']

        print(f"\n5日超额收益对比:")
        print(f"  - 节假日前: {holiday_5d['平均收益']:+.2f}% (胜率{holiday_5d['胜率']:.1f}%, t={holiday_5d['t统计量']:.2f})")
        print(f"  - 其他交易日: {other_5d['平均收益']:+.2f}% (胜率{other_5d['胜率']:.1f}%, t={other_5d['t统计量']:.2f})")
        print(f"  - 差异: {diff_5d:+.2f}%")

        # 统计显著性判断
        if abs(holiday_5d['t统计量']) > 2:
            print(f"\n结论: 节假日前资金流入信号在统计上显著（t={holiday_5d['t统计量']:.2f} > 2）")
        else:
            print(f"\n结论: 节假日前资金流入信号在统计上不显著（t={holiday_5d['t统计量']:.2f}）")

        if diff_5d > 0.3:
            print(f"       相比普通交易日，节假日前有额外信号价值（差异{diff_5d:+.2f}%）")
        else:
            print(f"       相比普通交易日，节假日前无明显额外信号价值")

    print(f"\n详细结果已保存至: {output_dir}")

    analyzer.close()


if __name__ == '__main__':
    main()
