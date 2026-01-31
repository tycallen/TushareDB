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

    def get_market_returns(self, start_date: str, end_date: str, index_code: str = '000001.SH') -> pd.DataFrame:
        """
        获取市场指数收益率（用于计算超额收益）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            index_code: 基准指数代码，默认上证指数 000001.SH
                       可选: 000001.SH (上证指数), 000300.SH (沪深300), 399001.SZ (深证成指)
        """
        if self._market_returns_cache is not None:
            return self._market_returns_cache

        # 从 index_daily 表获取指数数据
        query = """
        SELECT
            trade_date,
            pct_chg as market_return
        FROM index_daily
        WHERE ts_code = ?
        AND trade_date >= ? AND trade_date <= ?
        AND pct_chg IS NOT NULL
        ORDER BY trade_date
        """

        df = self.reader.db.con.execute(query, [index_code, start_date, end_date]).fetchdf()
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

    def calculate_combined_statistics(self, df: pd.DataFrame, periods: list) -> dict:
        """计算超额收益和绝对收益的综合统计量"""
        stats = {}

        for period in periods:
            excess_col = f'excess_return_{period}d'
            abs_col = f'abs_return_{period}d'

            if excess_col not in df.columns or abs_col not in df.columns:
                continue

            excess_returns = df[excess_col].dropna()
            abs_returns = df[abs_col].dropna()

            if len(excess_returns) == 0:
                continue

            stats[f'{period}d'] = {
                '样本数': len(excess_returns),
                '超额收益': excess_returns.mean(),
                '超额胜率': (excess_returns > 0).sum() / len(excess_returns) * 100,
                '超额t值': excess_returns.mean() / (excess_returns.std() / np.sqrt(len(excess_returns)) + 1e-6),
                '绝对收益': abs_returns.mean(),
                '绝对胜率': (abs_returns > 0).sum() / len(abs_returns) * 100,
                '绝对t值': abs_returns.mean() / (abs_returns.std() / np.sqrt(len(abs_returns)) + 1e-6),
            }

        return stats

    def calculate_yearly_statistics(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """按年度计算统计量"""
        df = df.copy()
        df['year'] = df['trade_date'].str[:4]

        yearly_stats = []

        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]

            for period in periods:
                excess_col = f'excess_return_{period}d'
                abs_col = f'abs_return_{period}d'

                if excess_col not in df.columns:
                    continue

                excess_returns = year_df[excess_col].dropna()
                abs_returns = year_df[abs_col].dropna()

                if len(excess_returns) == 0:
                    continue

                yearly_stats.append({
                    '年份': year,
                    '持有期': f'{period}d',
                    '样本数': len(excess_returns),
                    '超额收益%': excess_returns.mean(),
                    '超额胜率%': (excess_returns > 0).sum() / len(excess_returns) * 100,
                    '超额t值': excess_returns.mean() / (excess_returns.std() / np.sqrt(len(excess_returns)) + 1e-6),
                    '绝对收益%': abs_returns.mean(),
                    '绝对胜率%': (abs_returns > 0).sum() / len(abs_returns) * 100,
                    '绝对t值': abs_returns.mean() / (abs_returns.std() / np.sqrt(len(abs_returns)) + 1e-6),
                })

        return pd.DataFrame(yearly_stats)

    def close(self):
        self.reader.close()


def print_combined_table(title: str, data_by_type: dict, periods: list):
    """
    打印综合对比表格，同时显示超额收益和绝对收益

    data_by_type: {day_type: stats_dict}
    """
    print(f"\n{title}")
    print("=" * 120)

    # 表头
    header = f"{'交易日类型':<12} {'持有期':<6}"
    header += f"{'样本':>6} {'超额收益%':>10} {'超额胜率%':>10} {'超额t值':>8}"
    header += f"{'绝对收益%':>10} {'绝对胜率%':>10} {'绝对t值':>8}"
    print(header)
    print("-" * 120)

    for day_type, type_name in [
        ('period_end_holiday', '节假日前'),
        ('period_end_friday', '周五'),
        ('normal', '普通交易日'),
        ('period_end', '期末合计'),
    ]:
        if day_type not in data_by_type:
            continue

        stats = data_by_type[day_type]
        for period in periods:
            period_key = f'{period}d'
            if period_key not in stats:
                continue
            s = stats[period_key]

            row = f"{type_name:<12} {period_key:<6}"
            row += f"{s['样本数']:>6} {s['超额收益']:>+10.2f} {s['超额胜率']:>10.1f} {s['超额t值']:>8.2f}"
            row += f"{s['绝对收益']:>+10.2f} {s['绝对胜率']:>10.1f} {s['绝对t值']:>8.2f}"
            print(row)

    print("=" * 120)


def print_yearly_table(title: str, yearly_df: pd.DataFrame):
    """打印年度统计表格"""
    print(f"\n{title}")
    print("=" * 120)

    # 表头
    header = f"{'年份':<6} {'持有期':<6}"
    header += f"{'样本':>6} {'超额收益%':>10} {'超额胜率%':>10} {'超额t值':>8}"
    header += f"{'绝对收益%':>10} {'绝对胜率%':>10} {'绝对t值':>8}"
    print(header)
    print("-" * 120)

    for _, row in yearly_df.iterrows():
        line = f"{row['年份']:<6} {row['持有期']:<6}"
        line += f"{row['样本数']:>6} {row['超额收益%']:>+10.2f} {row['超额胜率%']:>10.1f} {row['超额t值']:>8.2f}"
        line += f"{row['绝对收益%']:>+10.2f} {row['绝对胜率%']:>10.1f} {row['绝对t值']:>8.2f}"
        print(line)

    print("=" * 120)


def main():
    print("=" * 80)
    print("期末 vs 每日 资金流入信号对比分析（超额收益版）")
    print("=" * 80)

    analyzer = PeriodEndVsDailyAnalyzer('tushare.db')

    # 分析参数
    start_date = '20200101'
    end_date = '20260131'
    top_n = 5
    level = 'L1'
    periods = [1, 3, 5]

    print(f"\n分析期间: {start_date} ~ {end_date}")
    print(f"板块层级: {level}")
    print(f"榜单取前: {top_n} 名")
    print(f"持有期: {periods} 交易日")
    print(f"基准指数: 申万A指 (801003.SI)")
    print(f"收益指标: 超额收益（板块收益 - 申万A指收益）+ 绝对收益")

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

    # 按日期类型计算综合统计
    stats_by_type = {}
    for day_type in ['period_end_holiday', 'period_end_friday', 'normal']:
        type_df = main_df[main_df['day_type'] == day_type]
        if len(type_df) > 0:
            stats_by_type[day_type] = analyzer.calculate_combined_statistics(type_df, periods)

    # 期末合计
    period_end_df = main_df[main_df['day_type'].isin(['period_end_holiday', 'period_end_friday'])]
    if len(period_end_df) > 0:
        stats_by_type['period_end'] = analyzer.calculate_combined_statistics(period_end_df, periods)

    # 打印整体统计表格
    print_combined_table("【主力资金榜】整体统计 - 超额收益 vs 绝对收益", stats_by_type, periods)

    # ============ 年度统计 ============
    # 节假日前 - 年度统计
    holiday_df = main_df[main_df['day_type'] == 'period_end_holiday']
    if len(holiday_df) > 0:
        holiday_yearly = analyzer.calculate_yearly_statistics(holiday_df, periods)
        print_yearly_table("【主力资金榜 - 节假日前】年度统计", holiday_yearly)

    # 周五 - 年度统计
    friday_df = main_df[main_df['day_type'] == 'period_end_friday']
    if len(friday_df) > 0:
        friday_yearly = analyzer.calculate_yearly_statistics(friday_df, periods)
        print_yearly_table("【主力资金榜 - 周五】年度统计", friday_yearly)

    # 普通交易日 - 年度统计
    normal_df = main_df[main_df['day_type'] == 'normal']
    if len(normal_df) > 0:
        normal_yearly = analyzer.calculate_yearly_statistics(normal_df, periods)
        print_yearly_table("【主力资金榜 - 普通交易日】年度统计", normal_yearly)

    # ============ 总资金榜分析 ============
    total_df = results_df[results_df['rank_type'] == 'total_flow']

    total_stats_by_type = {}
    for day_type in ['period_end_holiday', 'period_end_friday', 'normal']:
        type_df = total_df[total_df['day_type'] == day_type]
        if len(type_df) > 0:
            total_stats_by_type[day_type] = analyzer.calculate_combined_statistics(type_df, periods)

    total_period_end_df = total_df[total_df['day_type'].isin(['period_end_holiday', 'period_end_friday'])]
    if len(total_period_end_df) > 0:
        total_stats_by_type['period_end'] = analyzer.calculate_combined_statistics(total_period_end_df, periods)

    print_combined_table("【总资金榜】整体统计 - 超额收益 vs 绝对收益", total_stats_by_type, periods)

    # 总资金榜年度统计
    total_holiday_df = total_df[total_df['day_type'] == 'period_end_holiday']
    if len(total_holiday_df) > 0:
        total_holiday_yearly = analyzer.calculate_yearly_statistics(total_holiday_df, periods)
        print_yearly_table("【总资金榜 - 节假日前】年度统计", total_holiday_yearly)

    # 保存结果
    output_dir = Path('output/period_end_vs_daily')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'all_results.csv', index=False, encoding='utf-8-sig')

    # 保存年度统计
    if len(holiday_df) > 0:
        holiday_yearly.to_csv(output_dir / 'main_holiday_yearly.csv', index=False, encoding='utf-8-sig')
    if len(friday_df) > 0:
        friday_yearly.to_csv(output_dir / 'main_friday_yearly.csv', index=False, encoding='utf-8-sig')
    if len(normal_df) > 0:
        normal_yearly.to_csv(output_dir / 'main_normal_yearly.csv', index=False, encoding='utf-8-sig')

    # ============ 总结 ============
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)

    # 主力资金榜关键指标
    if 'period_end_holiday' in stats_by_type and 'normal' in stats_by_type:
        holiday_5d = stats_by_type['period_end_holiday'].get('5d', {})
        normal_5d = stats_by_type['normal'].get('5d', {})

        if holiday_5d and normal_5d:
            print("\n主力资金榜 - 5日收益对比:")
            print(f"  {'类型':<12} {'超额收益%':>10} {'绝对收益%':>10} {'超额t值':>10}")
            print(f"  {'-' * 44}")
            print(f"  {'节假日前':<12} {holiday_5d['超额收益']:>+10.2f} {holiday_5d['绝对收益']:>+10.2f} {holiday_5d['超额t值']:>10.2f}")
            print(f"  {'普通交易日':<12} {normal_5d['超额收益']:>+10.2f} {normal_5d['绝对收益']:>+10.2f} {normal_5d['超额t值']:>10.2f}")
            print(f"  {'-' * 44}")
            diff_excess = holiday_5d['超额收益'] - normal_5d['超额收益']
            diff_abs = holiday_5d['绝对收益'] - normal_5d['绝对收益']
            print(f"  {'差异':<12} {diff_excess:>+10.2f} {diff_abs:>+10.2f}")

            # 统计显著性判断
            if abs(holiday_5d['超额t值']) > 2:
                print(f"\n结论: 节假日前资金流入信号在统计上显著（超额t={holiday_5d['超额t值']:.2f} > 2）")
            else:
                print(f"\n结论: 节假日前资金流入信号在统计上不显著（超额t={holiday_5d['超额t值']:.2f}）")

            if diff_excess > 0.3:
                print(f"       相比普通交易日，节假日前有额外超额信号价值（差异{diff_excess:+.2f}%）")
            else:
                print(f"       相比普通交易日，节假日前无明显额外超额信号价值")

    print(f"\n详细结果已保存至: {output_dir}")

    analyzer.close()


if __name__ == '__main__':
    main()
