#!/usr/bin/env python
"""
市场宽度图演示 - 保存图片版本
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，保存图片
import datetime
from tushare_db import DataReader
from market_width import get_industry_width, show_industry_width
import matplotlib.pyplot as plt


def main():
    """运行市场宽度分析并保存图片"""
    print("=" * 70)
    print("市场宽度分析演示")
    print("=" * 70)

    reader = DataReader()

    try:
        # 参数设置
        end_date = datetime.date(2025, 12, 16)  # 数据库最新日期
        days = 60
        min_stocks = 30

        print(f"\n分析参数:")
        print(f"  结束日期: {end_date}")
        print(f"  交易日数: {days}")
        print(f"  最小行业股票数: {min_stocks}")
        print()

        # 计算市场宽度
        print("正在计算市场宽度...")
        df = get_industry_width(
            reader,
            end_date=end_date,
            days=days,
            min_stocks_per_industry=min_stocks
        )

        # 显示统计
        print("\n" + "=" * 70)
        print("数据概览")
        print("=" * 70)
        print(f"分析周期: {df.index[0]} 至 {df.index[-1]}")
        print(f"交易日数: {len(df)}")
        print(f"行业数量: {len(df.columns) - 1}")

        print(f"\n市场总分统计:")
        print(f"  平均: {df['总分'].mean():.0f}")
        print(f"  最高: {df['总分'].max():.0f} ({df['总分'].idxmax()})")
        print(f"  最低: {df['总分'].min():.0f} ({df['总分'].idxmin()})")

        # 显示最近5天数据
        print(f"\n最近5天数据:")
        print(df.tail(5).to_string())

        # 强势行业
        print(f"\n强势行业 Top 5:")
        industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
        for i, (ind, val) in enumerate(industry_avg.head(5).items(), 1):
            print(f"  {i}. {ind:10s} {val:.1f}%")

        # 弱势行业
        print(f"\n弱势行业 Top 5:")
        for i, (ind, val) in enumerate(industry_avg.tail(5).items(), 1):
            print(f"  {i}. {ind:10s} {val:.1f}%")

        # 生成热力图
        print("\n正在生成热力图...")
        import seaborn as sns

        # 取最近30天数据
        df_plot = df.tail(30)

        fig = plt.figure(figsize=(16, max(8, len(df_plot) * 0.3)))
        grid = plt.GridSpec(1, 10)

        cmap = sns.diverging_palette(200, 10, as_cmap=True)

        # 左侧热力图
        heatmap1 = fig.add_subplot(grid[:, :-1])
        heatmap1.xaxis.set_ticks_position('top')
        sns.heatmap(
            df_plot[df_plot.columns[:-1]],
            vmin=0, vmax=100,
            annot=True, fmt=".0f",
            cmap=cmap,
            annot_kws={'size': 8},
            cbar=False,
            linewidths=0.5,
            linecolor='white'
        )
        # 旋转行业名称标签，便于阅读
        heatmap1.set_xticklabels(heatmap1.get_xticklabels(), rotation=45, ha='left', fontsize=9)
        heatmap1.set_ylabel('交易日', fontsize=12)

        # 右侧总分
        heatmap2 = fig.add_subplot(grid[:, -1])
        heatmap2.xaxis.set_ticks_position('top')
        max_score = (len(df_plot.columns) - 1) * 100
        sns.heatmap(
            df_plot[['总分']],
            vmin=0, vmax=max_score,
            annot=True, fmt=".0f",
            cmap=cmap,
            annot_kws={'size': 10, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='white'
        )

        plt.suptitle('市场宽度热力图 - 最近30个交易日', fontsize=16, y=1.0, fontweight='bold')
        plt.tight_layout()

        heatmap_file = 'market_width_heatmap.png'
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        print(f"✓ 热力图已保存: {heatmap_file}")
        plt.close()

        # 生成趋势图
        print("正在生成趋势图...")
        plt.figure(figsize=(16, 6))
        plt.plot(range(len(df)), df['总分'].values,
                linewidth=2, marker='o', markersize=4, color='#2E86AB')

        # 添加均线
        ma5 = df['总分'].rolling(5).mean()
        ma10 = df['总分'].rolling(10).mean()
        plt.plot(range(len(df)), ma5.values,
                linewidth=1.5, linestyle='--', alpha=0.7, color='#A23B72', label='5日均线')
        plt.plot(range(len(df)), ma10.values,
                linewidth=1.5, linestyle='--', alpha=0.7, color='#F18F01', label='10日均线')

        plt.title('市场宽度总分趋势图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('交易日', fontsize=12)
        plt.ylabel('总分', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 设置x轴标签
        step = max(1, len(df) // 10)
        plt.xticks(range(0, len(df), step), df.index[::step], rotation=45)

        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()

        trend_file = 'market_width_trend.png'
        plt.savefig(trend_file, dpi=150, bbox_inches='tight')
        print(f"✓ 趋势图已保存: {trend_file}")
        plt.close()

        # 生成行业对比图
        print("正在生成行业对比图...")
        plt.figure(figsize=(14, 8))

        # 取最近的数据计算平均值
        recent_avg = df.tail(20).drop(columns=['总分']).mean().sort_values(ascending=False)

        colors = ['#2E86AB' if x > 50 else '#A23B72' for x in recent_avg.values]

        plt.barh(range(len(recent_avg)), recent_avg.values, color=colors, alpha=0.8)
        plt.yticks(range(len(recent_avg)), recent_avg.index, fontsize=10)
        plt.xlabel('平均上涨占比 (%)', fontsize=12)
        plt.title('行业市场宽度对比 (最近20个交易日平均)', fontsize=16, fontweight='bold', pad=20)
        plt.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% 基准线')
        plt.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()

        compare_file = 'market_width_industry_compare.png'
        plt.savefig(compare_file, dpi=150, bbox_inches='tight')
        print(f"✓ 行业对比图已保存: {compare_file}")
        plt.close()

        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)
        print(f"\n生成的图片文件:")
        print(f"  1. {heatmap_file} - 市场宽度热力图")
        print(f"  2. {trend_file} - 总分趋势图")
        print(f"  3. {compare_file} - 行业对比图")
        print("\n请打开这些图片查看中文显示效果！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


if __name__ == '__main__':
    main()
