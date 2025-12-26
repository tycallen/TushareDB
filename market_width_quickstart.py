#!/usr/bin/env python
"""
市场宽度图 - 快速开始脚本
运行此脚本即可快速生成市场宽度分析图表
"""
import datetime
from tushare_db import DataReader
from market_width import get_industry_width, show_industry_width


def main():
    """快速开始 - 一键运行市场宽度分析"""
    print("=" * 70)
    print("市场宽度分析工具 - 快速开始")
    print("=" * 70)
    print()

    # 初始化数据读取器
    print("正在连接数据库...")
    reader = DataReader()

    try:
        # 使用最新数据
        end_date = datetime.date(2025, 12, 16)  # 数据库中最新日期
        days = 60  # 分析最近60个交易日
        min_stocks = 30  # 至少包含30只股票的行业

        print(f"分析参数:")
        print(f"  - 结束日期: {end_date}")
        print(f"  - 交易日数: {days}")
        print(f"  - 最小行业股票数: {min_stocks}")
        print()

        # 获取市场宽度数据
        print("正在计算市场宽度...")
        df = get_industry_width(
            reader,
            end_date=end_date,
            days=days,
            min_stocks_per_industry=min_stocks
        )

        # 显示基本统计
        print("\n" + "=" * 70)
        print("数据概览")
        print("=" * 70)
        print(f"分析周期: {df.index[0]} 至 {df.index[-1]}")
        print(f"行业数量: {len(df.columns) - 1}")
        print(f"\n市场总分统计:")
        print(f"  - 平均值: {df['总分'].mean():.0f}")
        print(f"  - 最高值: {df['总分'].max():.0f} (日期: {df['总分'].idxmax()})")
        print(f"  - 最低值: {df['总分'].min():.0f} (日期: {df['总分'].idxmin()})")
        print(f"  - 标准差: {df['总分'].std():.2f}")

        # 显示强势行业 Top 10
        print(f"\n强势行业 Top 10（按平均上涨占比）:")
        industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
        for i, (industry, avg) in enumerate(industry_avg.head(10).items(), 1):
            print(f"  {i:2d}. {industry:8s} {avg:5.1f}%")

        # 显示弱势行业 Top 10
        print(f"\n弱势行业 Top 10（按平均上涨占比）:")
        for i, (industry, avg) in enumerate(industry_avg.tail(10).items(), 1):
            print(f"  {i:2d}. {industry:8s} {avg:5.1f}%")

        # 趋势分析
        recent_score = df['总分'].tail(5).mean()
        previous_score = df['总分'].iloc[-10:-5].mean()
        trend = recent_score - previous_score

        print(f"\n市场趋势分析（最近5天 vs 前5天）:")
        print(f"  - 最近5天平均总分: {recent_score:.0f}")
        print(f"  - 前5天平均总分: {previous_score:.0f}")
        print(f"  - 变化: {trend:+.0f} ({'上升趋势 📈' if trend > 0 else '下降趋势 📉'})")

        # 可视化
        print("\n正在生成可视化图表...")
        print("提示：关闭图表窗口以继续...")
        show_industry_width(df, count=min(40, days))

        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()
        print("\n数据库连接已关闭")


if __name__ == '__main__':
    main()
