#!/usr/bin/env python
"""
报告管理系统演示

展示如何：
1. 保存因子验证结果
2. 查询历史记录
3. 对比分析
4. 生成统计报告
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.tushare_db.factor_validation import FactorFilter, FactorRegistry
from src.tushare_db.factor_validation.report_manager import ReportManager


def demo_save_reports():
    """演示：保存多个因子验证结果"""
    print("=" * 80)
    print("演示 1: 保存因子验证结果")
    print("=" * 80)

    # 创建报告管理器
    manager = ReportManager()
    print(f"\n数据库路径: {manager.db_path}")

    # 创建过滤器
    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=1000,
        simulation_days=100,
        alpha_threshold=1.5
    )

    # 测试的因子
    test_factors = ['macd_golden_cross', 'rsi_oversold', 'kdj_golden_cross', 'doji']

    print(f"\n正在测试 {len(test_factors)} 个因子...")

    for factor_name in test_factors:
        print(f"\n  测试: {factor_name}...", end=" ")

        # 运行验证
        report = filter_obj.filter(
            factor=factor_name,
            ts_codes=[],
            use_sample=True
        )

        # 保存到数据库
        record_id = manager.save_report(
            report=report,
            ts_code='000001.SZ',
            stock_name='平安银行',
            start_date='20240101',
            end_date='20241231',
            notes=f'批量测试 - {datetime.now().strftime("%Y%m%d")}'
        )

        alpha = report.summary.get('alpha_ratio_median', 0)
        print(f"✓ 保存成功 (ID={record_id}, Alpha={alpha:.2f})")

    print(f"\n✓ 共保存 {len(test_factors)} 条记录")


def demo_query_history():
    """演示：查询历史记录"""
    print("\n" + "=" * 80)
    print("演示 2: 查询历史记录")
    print("=" * 80)

    manager = ReportManager()

    # 查询所有记录
    print("\n所有记录:")
    df = manager.query_records(limit=10)
    if not df.empty:
        print(df[['timestamp', 'factor_name', 'ts_code', 'alpha_ratio', 'recommendation']].to_string(index=False))
        print(f"\n共 {len(df)} 条记录")
    else:
        print("暂无记录")

    # 按因子查询
    print("\n\n查询 'macd_golden_cross' 的历史记录:")
    df_macd = manager.get_factor_history('macd_golden_cross')
    if not df_macd.empty:
        print(df_macd[['timestamp', 'ts_code', 'alpha_ratio', 'p_actual', 'p_random']].to_string(index=False))
    else:
        print("暂无记录")

    # 按股票查询
    print("\n\n查询股票 '000001.SZ' 的所有测试:")
    df_stock = manager.get_stock_history('000001.SZ')
    if not df_stock.empty:
        print(df_stock[['timestamp', 'factor_name', 'alpha_ratio', 'recommendation']].to_string(index=False))
    else:
        print("暂无记录")


def demo_compare_analysis():
    """演示：对比分析"""
    print("\n" + "=" * 80)
    print("演示 3: 对比分析")
    print("=" * 80)

    manager = ReportManager()

    # 时间序列对比
    print("\n同一因子不同时间的表现 (macd_golden_cross):")
    df_time = manager.compare_factor_over_time('macd_golden_cross', '000001.SZ')
    if not df_time.empty and len(df_time) > 1:
        print(df_time[['timestamp', 'alpha_ratio', 'alpha_ratio_change']].to_string(index=False))

        # 趋势分析
        if 'alpha_ratio_change' in df_time.columns:
            avg_change = df_time['alpha_ratio_change'].mean()
            print(f"\n平均变化: {avg_change:.3f}")
            if avg_change > 0:
                print("趋势: 改善 ⬆")
            elif avg_change < 0:
                print("趋势: 恶化 ⬇")
            else:
                print("趋势: 稳定 →")
    else:
        print("数据点不足，需要多次测试同一因子才能进行时间对比")

    # 因子排名
    print("\n\n因子排名 (按 Alpha Ratio):")
    df_all = manager.query_records(limit=100)
    if not df_all.empty:
        ranking = df_all.groupby('factor_name')['alpha_ratio'].mean().sort_values(ascending=False)
        print(ranking.to_string())


def demo_statistics():
    """演示：统计摘要"""
    print("\n" + "=" * 80)
    print("演示 4: 统计摘要")
    print("=" * 80)

    manager = ReportManager()

    # 生成统计摘要
    stats = manager.get_statistics_summary(days=30)

    if "message" in stats:
        print(stats["message"])
        return

    print(f"\n统计期间: 最近 {stats['period_days']} 天")
    print(f"\n总体概况:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  测试因子数: {stats['unique_factors']}")
    print(f"  测试股票数: {stats['unique_stocks']}")

    print(f"\n质检结果分布:")
    print(f"  建议保留: {stats['keep_count']} ({stats['keep_count']/stats['total_records']*100:.1f}%)")
    print(f"  建议废弃: {stats['discard_count']} ({stats['discard_count']/stats['total_records']*100:.1f}%)")
    print(f"  显著因子: {stats['significant_count']} ({stats['significant_count']/stats['total_records']*100:.1f}%)")

    print(f"\nAlpha Ratio 统计:")
    print(f"  平均值: {stats['avg_alpha_ratio']:.3f}")
    print(f"  中位数: {stats['median_alpha_ratio']:.3f}")
    print(f"  最佳因子: {stats['best_factor']} ({stats['best_alpha']:.3f})")
    print(f"  最差因子: {stats['worst_factor']} ({stats['worst_alpha']:.3f})")

    # 生成报告
    print("\n\n生成完整报告...")
    report = manager.generate_report(days=30)
    print(report[:500] + "...")


def demo_export():
    """演示：导出功能"""
    print("\n" + "=" * 80)
    print("演示 5: 导出数据")
    print("=" * 80)

    manager = ReportManager()

    # 导出到CSV
    print("\n导出到 CSV...")
    csv_path = "factor_reports_export.csv"
    manager.export_to_csv(csv_path, limit=1000)

    # 导出到Excel
    print("\n导出到 Excel...")
    excel_path = "factor_reports_export.xlsx"
    try:
        manager.export_to_excel(excel_path, limit=1000)
    except ImportError:
        print("  需要安装 openpyxl: pip install openpyxl")

    print("\n✓ 导出完成")


def demo_batch_analysis():
    """演示：批量分析场景"""
    print("\n" + "=" * 80)
    print("演示 6: 批量分析场景")
    print("=" * 80)

    manager = ReportManager()

    # 场景1: 找出 consistently good 的因子
    print("\n场景1: 找出表现稳定的因子")
    df = manager.query_records(limit=1000)

    if not df.empty:
        # 按因子统计
        factor_stats = df.groupby('factor_name').agg({
            'alpha_ratio': ['mean', 'std', 'count'],
            'recommendation': lambda x: (x == 'KEEP').sum()
        }).round(3)

        factor_stats.columns = ['avg_alpha', 'std_alpha', 'test_count', 'keep_count']
        factor_stats['stability_score'] = factor_stats['avg_alpha'] / (factor_stats['std_alpha'] + 0.01)
        factor_stats = factor_stats.sort_values('stability_score', ascending=False)

        print(factor_stats.head(10).to_string())

    # 场景2: 找出最活跃的股票
    print("\n\n场景2: 测试最活跃的股票")
    if not df.empty:
        stock_stats = df.groupby('ts_code').agg({
            'factor_name': 'count',
            'alpha_ratio': 'mean'
        }).sort_values('factor_name', ascending=False)

        print(stock_stats.head(10).to_string())


def main():
    """主函数"""
    print("=" * 80)
    print("报告管理系统 - 完整演示")
    print("=" * 80)
    print("\n本演示展示如何保存、查询和分析因子验证历史记录")

    try:
        # 运行演示
        demo_save_reports()
        demo_query_history()
        demo_compare_analysis()
        demo_statistics()
        demo_export()
        demo_batch_analysis()

        print("\n" + "=" * 80)
        print("所有演示完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
