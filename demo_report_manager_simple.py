#!/usr/bin/env python
"""
报告管理系统演示 - 使用模拟数据

展示 ReportManager 的核心功能：
1. 保存记录
2. 查询历史
3. 对比分析
4. 统计报表
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.tushare_db.factor_validation.report_manager import ReportManager, ValidationRecord


def create_mock_records(manager: ReportManager, n_records: int = 20):
    """创建模拟记录用于演示"""
    print(f"创建 {n_records} 条模拟记录...")

    factors = [
        ('macd_golden_cross', 'MACD金叉'),
        ('rsi_oversold', 'RSI超卖'),
        ('kdj_golden_cross', 'KDJ金叉'),
        ('doji', '十字星'),
        ('golden_cross', '均线金叉'),
        ('bollinger_lower_break', '布林带下轨突破'),
    ]

    stocks = [
        ('000001.SZ', '平安银行'),
        ('000002.SZ', '万科A'),
        ('600000.SH', '浦发银行'),
    ]

    np.random.seed(42)

    for i in range(n_records):
        factor_name, factor_desc = factors[i % len(factors)]
        ts_code, stock_name = stocks[i % len(stocks)]

        # 随机生成结果
        p_random = np.random.uniform(0.02, 0.10)
        alpha_ratio = np.random.uniform(0.5, 2.5)
        p_actual = min(p_random * alpha_ratio, 0.5)

        record = ValidationRecord(
            timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
            factor_name=factor_name,
            factor_description=factor_desc,
            ts_code=ts_code,
            stock_name=stock_name,
            start_date='20240101',
            end_date='20241231',
            n_days=252,
            n_simulations=5000,
            simulation_days=252,
            alpha_threshold=1.5,
            p_actual=p_actual,
            p_random=p_random,
            alpha_ratio=alpha_ratio,
            n_signals_actual=int(p_actual * 252),
            n_signals_random=int(p_random * 5000 * 252),
            p_value=np.random.uniform(0, 0.1),
            is_significant=alpha_ratio >= 1.5,
            recommendation='KEEP' if alpha_ratio >= 1.5 else 'DISCARD',
            mu=np.random.uniform(0.05, 0.15),
            sigma=np.random.uniform(0.15, 0.35),
            notes=f'模拟记录 #{i+1}'
        )

        manager._insert_record(record)

    print(f"✓ 已创建 {n_records} 条记录")


def demo_query_and_display():
    """演示查询和显示"""
    print("\n" + "=" * 80)
    print("演示 1: 查询所有记录")
    print("=" * 80)

    manager = ReportManager()

    # 创建模拟数据
    create_mock_records(manager, n_records=20)

    # 查询所有记录
    df = manager.query_records(limit=10)
    print(f"\n最近10条记录:")
    print(df[['timestamp', 'factor_name', 'ts_code', 'alpha_ratio', 'recommendation']].to_string(index=False))


def demo_factor_history():
    """演示因子历史查询"""
    print("\n" + "=" * 80)
    print("演示 2: 查询特定因子的历史")
    print("=" * 80)

    manager = ReportManager()

    print("\nmacd_golden_cross 的历史表现:")
    df = manager.get_factor_history('macd_golden_cross')

    if not df.empty:
        print(df[['timestamp', 'ts_code', 'alpha_ratio', 'p_actual', 'p_random']].to_string(index=False))
        print(f"\n统计:")
        print(f"  平均 Alpha: {df['alpha_ratio'].mean():.3f}")
        print(f"  最大 Alpha: {df['alpha_ratio'].max():.3f}")
        print(f"  最小 Alpha: {df['alpha_ratio'].min():.3f}")
        print(f"  标准差: {df['alpha_ratio'].std():.3f}")


def demo_comparison():
    """演示对比分析"""
    print("\n" + "=" * 80)
    print("演示 3: 因子对比分析")
    print("=" * 80)

    manager = ReportManager()

    # 获取所有记录
    df = manager.query_records(limit=100)

    if not df.empty:
        # 按因子分组统计
        print("\n各因子平均表现:")
        factor_stats = df.groupby('factor_name').agg({
            'alpha_ratio': ['mean', 'std', 'min', 'max', 'count'],
            'recommendation': lambda x: (x == 'KEEP').sum()
        }).round(3)

        factor_stats.columns = ['avg_alpha', 'std_alpha', 'min_alpha', 'max_alpha', 'test_count', 'keep_count']
        factor_stats['keep_rate'] = (factor_stats['keep_count'] / factor_stats['test_count'] * 100).round(1)
        factor_stats = factor_stats.sort_values('avg_alpha', ascending=False)

        print(factor_stats.to_string())

        # 找出最佳因子
        best_factor = factor_stats.index[0]
        print(f"\n最佳因子: {best_factor}")
        print(f"  平均 Alpha: {factor_stats.loc[best_factor, 'avg_alpha']}")
        print(f"  通过率: {factor_stats.loc[best_factor, 'keep_rate']}%")


def demo_statistics():
    """演示统计摘要"""
    print("\n" + "=" * 80)
    print("演示 4: 统计摘要")
    print("=" * 80)

    manager = ReportManager()

    stats = manager.get_statistics_summary(days=30)

    if "message" in stats:
        print(stats["message"])
        return

    print(f"\n统计期间: 最近 {stats['period_days']} 天")
    print(f"\n总体概况:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  测试因子数: {stats['unique_factors']}")
    print(f"  测试股票数: {stats['unique_stocks']}")

    print(f"\n质检结果:")
    print(f"  建议保留: {stats['keep_count']} ({stats['keep_count']/stats['total_records']*100:.1f}%)")
    print(f"  建议废弃: {stats['discard_count']} ({stats['discard_count']/stats['total_records']*100:.1f}%)")

    print(f"\nAlpha Ratio:")
    print(f"  平均值: {stats['avg_alpha_ratio']:.3f}")
    print(f"  中位数: {stats['median_alpha_ratio']:.3f}")
    print(f"  最佳: {stats['best_factor']} ({stats['best_alpha']:.3f})")
    print(f"  最差: {stats['worst_factor']} ({stats['worst_alpha']:.3f})")


def demo_export():
    """演示导出"""
    print("\n" + "=" * 80)
    print("演示 5: 导出数据")
    print("=" * 80)

    manager = ReportManager()

    # 导出CSV
    csv_path = "demo_factor_reports.csv"
    manager.export_to_csv(csv_path, limit=100)

    # 生成报告
    print("\n生成统计报告:")
    report = manager.generate_report(days=30)
    print(report)


def main():
    """主函数"""
    print("=" * 80)
    print("报告管理系统演示")
    print("=" * 80)

    demo_query_and_display()
    demo_factor_history()
    demo_comparison()
    demo_statistics()
    demo_export()

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print(f"\n数据库位置: ~/.factor_validation/reports.db")
    print("你可以使用 SQLite 浏览器查看数据，或通过 ReportManager API 查询")


if __name__ == "__main__":
    main()
