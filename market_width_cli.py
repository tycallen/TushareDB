#!/usr/bin/env python
"""
市场宽度图 - 命令行工具
支持自定义参数的市场宽度分析
"""
import argparse
import datetime
from tushare_db import DataReader
from market_width import get_industry_width, show_industry_width


def parse_date(date_str):
    """解析日期字符串"""
    try:
        if date_str.lower() == 'today':
            # 使用数据库中最新的日期
            return datetime.date(2025, 12, 16)
        return datetime.datetime.strptime(date_str, '%Y%m%d').date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的日期格式: {date_str}，请使用 YYYYMMDD 格式或 'today'")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='市场宽度分析工具 - 基于 Tushare-DuckDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 分析最近60个交易日，至少30只股票的行业
  python market_width_cli.py -d 60 -m 30

  # 指定结束日期
  python market_width_cli.py -e 20251216 -d 40

  # 只输出数据，不显示图表
  python market_width_cli.py -d 30 --no-plot

  # 分析最近100天，只关注大行业（50+股票）
  python market_width_cli.py -d 100 -m 50

  # 显示最近20天的图表
  python market_width_cli.py -d 60 -c 20
        '''
    )

    parser.add_argument(
        '-e', '--end-date',
        type=parse_date,
        default='today',
        help='结束日期 (格式: YYYYMMDD 或 "today"，默认: today)'
    )

    parser.add_argument(
        '-d', '--days',
        type=int,
        default=60,
        help='统计的交易日数量 (默认: 60)'
    )

    parser.add_argument(
        '-m', '--min-stocks',
        type=int,
        default=30,
        help='行业最少包含的股票数量 (默认: 30)'
    )

    parser.add_argument(
        '-c', '--chart-days',
        type=int,
        default=None,
        help='图表显示的交易日数量 (默认: 显示全部)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='不显示图表，只输出统计数据'
    )

    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='导出数据到文件 (CSV 或 Excel 格式，例如: output.csv 或 output.xlsx)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    args = parser.parse_args()

    # 显示参数
    print("=" * 70)
    print("市场宽度分析工具")
    print("=" * 70)
    print(f"\n参数设置:")
    print(f"  结束日期: {args.end_date}")
    print(f"  交易日数: {args.days}")
    print(f"  最小行业股票数: {args.min_stocks}")
    if args.chart_days:
        print(f"  图表显示天数: {args.chart_days}")
    if args.export:
        print(f"  导出文件: {args.export}")
    print()

    # 初始化数据库
    if args.verbose:
        print("正在连接数据库...")
    reader = DataReader()

    try:
        # 计算市场宽度
        print("正在计算市场宽度...")
        df = get_industry_width(
            reader,
            end_date=args.end_date,
            days=args.days,
            min_stocks_per_industry=args.min_stocks
        )

        # 输出统计信息
        print("\n" + "=" * 70)
        print("统计结果")
        print("=" * 70)
        print(f"\n数据概览:")
        print(f"  日期范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"  交易日数: {len(df)}")
        print(f"  行业数量: {len(df.columns) - 1}")

        print(f"\n市场总分统计:")
        stats = df['总分'].describe()
        print(f"  平均值: {stats['mean']:.0f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  最小值: {stats['min']:.0f}")
        print(f"  最大值: {stats['max']:.0f}")

        # 最强/弱日期
        max_idx = df['总分'].idxmax()
        min_idx = df['总分'].idxmin()
        print(f"  最强日期: {max_idx} (总分: {df.loc[max_idx, '总分']:.0f})")
        print(f"  最弱日期: {min_idx} (总分: {df.loc[min_idx, '总分']:.0f})")

        # 强势行业 Top 5
        print(f"\n强势行业 Top 5:")
        industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
        for i, (industry, avg) in enumerate(industry_avg.head(5).items(), 1):
            print(f"  {i}. {industry:10s} {avg:.1f}%")

        # 弱势行业 Top 5
        print(f"\n弱势行业 Top 5:")
        for i, (industry, avg) in enumerate(industry_avg.tail(5).items(), 1):
            print(f"  {i}. {industry:10s} {avg:.1f}%")

        # 趋势分析
        if len(df) >= 10:
            recent_5 = df['总分'].tail(5).mean()
            previous_5 = df['总分'].iloc[-10:-5].mean()
            trend = recent_5 - previous_5
            trend_str = "上升 📈" if trend > 0 else "下降 📉" if trend < 0 else "持平 ➡️"

            print(f"\n市场趋势:")
            print(f"  最近5天平均: {recent_5:.0f}")
            print(f"  前5天平均: {previous_5:.0f}")
            print(f"  变化: {trend:+.0f} ({trend_str})")

        # 导出数据
        if args.export:
            if args.export.endswith('.csv'):
                df.to_csv(args.export, encoding='utf-8-sig')
                print(f"\n✓ 数据已导出至: {args.export}")
            elif args.export.endswith('.xlsx'):
                df.to_excel(args.export)
                print(f"\n✓ 数据已导出至: {args.export}")
            else:
                print(f"\n✗ 不支持的文件格式: {args.export}")
                print("  支持的格式: .csv, .xlsx")

        # 显示图表
        if not args.no_plot:
            print("\n正在生成图表...")
            show_industry_width(df, count=args.chart_days)

        print("\n" + "=" * 70)
        print("分析完成!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        reader.close()
        if args.verbose:
            print("\n数据库连接已关闭")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
