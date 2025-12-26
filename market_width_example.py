"""
市场宽度图使用示例
演示如何使用 market_width 模块进行市场宽度分析
"""
import datetime
from tushare_db import DataReader
from market_width import get_industry_width, show_industry_width


def example_1_basic_usage():
    """示例1：基本使用 - 分析最近100个交易日的市场宽度"""
    print("\n" + "=" * 60)
    print("示例1：基本使用 - 最近100个交易日")
    print("=" * 60)

    reader = DataReader()
    try:
        # 使用今天作为结束日期
        end_date = datetime.date.today()
        days = 100

        # 获取市场宽度数据
        df = get_industry_width(reader, end_date, days)

        # 显示数据
        print("\n数据预览（最后10天）:")
        print(df.tail(10))

        # 可视化（显示最近50天）
        show_industry_width(df, count=50)

    finally:
        reader.close()


def example_2_specific_date_range():
    """示例2：指定日期范围 - 分析2024年的市场宽度"""
    print("\n" + "=" * 60)
    print("示例2：指定日期范围 - 2024年数据")
    print("=" * 60)

    reader = DataReader()
    try:
        # 指定结束日期
        end_date = '20241231'
        days = 200

        # 获取市场宽度数据
        df = get_industry_width(reader, end_date, days)

        # 显示统计信息
        print("\n总分统计:")
        print(df['总分'].describe())

        # 找出市场最强和最弱的交易日
        print(f"\n市场最强交易日: {df['总分'].idxmax()} (总分: {df['总分'].max():.0f})")
        print(f"市场最弱交易日: {df['总分'].idxmin()} (总分: {df['总分'].min():.0f})")

        # 可视化
        show_industry_width(df, count=100)

    finally:
        reader.close()


def example_3_recent_short_term():
    """示例3：短期分析 - 最近20个交易日"""
    print("\n" + "=" * 60)
    print("示例3：短期分析 - 最近20个交易日")
    print("=" * 60)

    reader = DataReader()
    try:
        end_date = datetime.date.today()
        days = 20

        # 获取市场宽度数据
        df = get_industry_width(reader, end_date, days)

        # 显示完整数据
        print("\n市场宽度数据:")
        print(df.to_string())

        # 分析各行业平均表现
        print("\n各行业平均上涨占比:")
        industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
        for industry, avg_pct in industry_avg.items():
            print(f"  {industry}: {avg_pct:.1f}%")

        # 可视化
        show_industry_width(df)

    finally:
        reader.close()


def example_4_custom_analysis():
    """示例4：自定义分析 - 找出强势行业"""
    print("\n" + "=" * 60)
    print("示例4：自定义分析 - 找出强势行业")
    print("=" * 60)

    reader = DataReader()
    try:
        end_date = datetime.date.today()
        days = 60

        # 获取市场宽度数据
        df = get_industry_width(reader, end_date, days)

        # 分析：找出持续强势的行业（上涨占比 > 50% 的天数）
        print("\n行业强势天数统计（上涨占比 > 50%）:")
        strong_days = (df.drop(columns=['总分']) > 50).sum().sort_values(ascending=False)
        for industry, days_count in strong_days.items():
            pct = days_count / len(df) * 100
            print(f"  {industry}: {days_count}天 ({pct:.1f}%)")

        # 分析：近期趋势（最近10天 vs 前10天）
        print("\n近期趋势分析（最近10天 vs 前10天）:")
        recent_10 = df.tail(10).drop(columns=['总分']).mean()
        previous_10 = df.iloc[-20:-10].drop(columns=['总分']).mean()
        trend = recent_10 - previous_10
        trend = trend.sort_values(ascending=False)

        print("\n改善最明显的行业:")
        for industry, change in trend.head(5).items():
            print(f"  {industry}: +{change:.1f}%")

        print("\n恶化最明显的行业:")
        for industry, change in trend.tail(5).items():
            print(f"  {industry}: {change:.1f}%")

        # 可视化
        show_industry_width(df, count=40)

    finally:
        reader.close()


if __name__ == '__main__':
    # 运行示例（取消注释想要运行的示例）

    # 基本使用
    example_1_basic_usage()

    # 指定日期范围
    # example_2_specific_date_range()

    # 短期分析
    # example_3_recent_short_term()

    # 自定义分析
    # example_4_custom_analysis()
