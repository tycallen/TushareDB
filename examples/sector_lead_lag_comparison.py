"""
板块传导关系对比分析

对比分析：
1. 原始收益率的传导关系（包含市场效应）
2. 超额收益率的传导关系（剔除市场效应）

帮助识别真实的产业链传导关系
"""

import logging
from tushare_db.sector_analysis import SectorAnalyzer, OutputManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """主函数"""

    print("=" * 70)
    print("板块传导关系对比分析：原始收益 vs 超额收益")
    print("=" * 70)

    # === 1. 初始化 ===
    analyzer = SectorAnalyzer('tushare.db')
    output_manager = OutputManager('output/lead_lag_comparison')

    # === 2. 设置分析参数 ===
    params = {
        'start_date': '20240101',
        'end_date': '20241231',
        'level': 'L1',
        'period': 'daily',
        'max_lag': 5,
        'min_correlation': 0.3,  # 提高阈值，关注强传导
    }

    print(f"\n分析参数:")
    print(f"  数据范围: {params['start_date']} ~ {params['end_date']}")
    print(f"  分析层级: {params['level']}")
    print(f"  分析周期: {params['period']}")
    print(f"  最大滞后: {params['max_lag']} 天")
    print(f"  相关阈值: {params['min_correlation']}")
    print()

    # === 3. 原始收益率传导关系 ===
    print("=" * 70)
    print("【方法1】原始收益率传导关系（包含市场效应）")
    print("=" * 70)

    lead_lag_raw = analyzer.calculate_lead_lag(
        start_date=params['start_date'],
        end_date=params['end_date'],
        max_lag=params['max_lag'],
        level=params['level'],
        period=params['period'],
        min_correlation=params['min_correlation']
    )

    print(f"\n✓ 找到 {len(lead_lag_raw)} 对传导关系")

    if len(lead_lag_raw) > 0:
        print("\n前10对传导关系:")
        print(lead_lag_raw[['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation', 'p_value']].head(10).to_string(index=False))

        # 统计：哪些板块最常作为"领涨板块"
        lead_counts = lead_lag_raw['sector_lead_name'].value_counts().head(10)
        print("\n最常见的领涨板块 Top 10:")
        for sector, count in lead_counts.items():
            print(f"  {sector}: {count} 次")

        # 保存
        output_manager.save_dataframe(lead_lag_raw, 'lead_lag_raw', format='csv')

    # === 4. 超额收益率传导关系 ===
    print("\n" + "=" * 70)
    print("【方法2】超额收益率传导关系（剔除市场效应）")
    print("=" * 70)

    lead_lag_excess = analyzer.calculate_lead_lag_excess(
        start_date=params['start_date'],
        end_date=params['end_date'],
        max_lag=params['max_lag'],
        level=params['level'],
        period=params['period'],
        min_correlation=params['min_correlation'],
        market_index='000300.SH'  # 沪深300作为市场基准
    )

    print(f"\n✓ 找到 {len(lead_lag_excess)} 对传导关系")

    if len(lead_lag_excess) > 0:
        print("\n前10对传导关系:")
        print(lead_lag_excess[['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation', 'p_value']].head(10).to_string(index=False))

        # 统计：哪些板块最常作为"领涨板块"
        lead_counts = lead_lag_excess['sector_lead_name'].value_counts().head(10)
        print("\n最常见的领涨板块 Top 10:")
        for sector, count in lead_counts.items():
            print(f"  {sector}: {count} 次")

        # 保存
        output_manager.save_dataframe(lead_lag_excess, 'lead_lag_excess', format='csv')

    # === 5. 对比分析 ===
    print("\n" + "=" * 70)
    print("【对比分析】")
    print("=" * 70)

    if len(lead_lag_raw) > 0 and len(lead_lag_excess) > 0:
        # 原始收益中金融板块的占比
        raw_financial = lead_lag_raw[
            lead_lag_raw['sector_lead_name'].isin(['银行', '非银金融'])
        ]
        raw_financial_pct = len(raw_financial) / len(lead_lag_raw) * 100

        # 超额收益中金融板块的占比
        excess_financial = lead_lag_excess[
            lead_lag_excess['sector_lead_name'].isin(['银行', '非银金融'])
        ]
        excess_financial_pct = len(excess_financial) / len(lead_lag_excess) * 100

        print(f"\n金融板块作为领涨板块的占比:")
        print(f"  原始收益: {raw_financial_pct:.1f}% ({len(raw_financial)}/{len(lead_lag_raw)})")
        print(f"  超额收益: {excess_financial_pct:.1f}% ({len(excess_financial)}/{len(lead_lag_excess)})")
        print(f"  差异: {raw_financial_pct - excess_financial_pct:.1f}% (正值表示原始收益中金融板块占比更高)")

        # 识别只在超额收益中出现的传导关系（真实产业传导）
        raw_pairs = set(lead_lag_raw.apply(
            lambda x: f"{x['sector_lead']}->{x['sector_lag']}", axis=1
        ))
        excess_pairs = set(lead_lag_excess.apply(
            lambda x: f"{x['sector_lead']}->{x['sector_lag']}", axis=1
        ))

        unique_excess = excess_pairs - raw_pairs
        print(f"\n只在超额收益中发现的传导关系: {len(unique_excess)} 对")
        print("（这些可能是真实的产业链传导，而非市场效应）")

        if len(unique_excess) > 0:
            # 找出这些关系的详情
            unique_df = lead_lag_excess[
                lead_lag_excess.apply(
                    lambda x: f"{x['sector_lead']}->{x['sector_lag']}" in unique_excess,
                    axis=1
                )
            ].head(10)

            print("\n示例（前10对）:")
            print(unique_df[['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation']].to_string(index=False))

    # === 6. 生成对比报告 ===
    print("\n" + "=" * 70)
    print("正在生成对比报告...")
    print("=" * 70)

    results = {
        'lead_lag_raw': lead_lag_raw,
        'lead_lag_excess': lead_lag_excess,
    }

    # 生成简要报告
    report_path = f"{output_manager.output_dir}/comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 板块传导关系对比分析报告\n\n")
        f.write(f"**分析周期**: {params['start_date']} ~ {params['end_date']}\n\n")

        f.write("## 方法对比\n\n")
        f.write("| 方法 | 传导关系数量 | 说明 |\n")
        f.write("|------|-------------|------|\n")
        f.write(f"| 原始收益率 | {len(lead_lag_raw)} | 包含市场整体效应 |\n")
        f.write(f"| 超额收益率 | {len(lead_lag_excess)} | 剔除市场效应，关注板块间真实传导 |\n\n")

        if len(lead_lag_raw) > 0:
            f.write("## 原始收益率传导关系 Top 10\n\n")
            f.write("| 排名 | 领涨板块 | 跟随板块 | 滞后天数 | 相关系数 |\n")
            f.write("|------|----------|----------|----------|----------|\n")
            for i, row in lead_lag_raw.head(10).iterrows():
                f.write(f"| {i+1} | {row['sector_lead_name']} | {row['sector_lag_name']} | "
                       f"{row['lag_days']} | {row['correlation']:.3f} |\n")
            f.write("\n")

        if len(lead_lag_excess) > 0:
            f.write("## 超额收益率传导关系 Top 10\n\n")
            f.write("| 排名 | 领涨板块 | 跟随板块 | 滞后天数 | 相关系数 |\n")
            f.write("|------|----------|----------|----------|----------|\n")
            for i, row in lead_lag_excess.head(10).iterrows():
                f.write(f"| {i+1} | {row['sector_lead_name']} | {row['sector_lag_name']} | "
                       f"{row['lag_days']} | {row['correlation']:.3f} |\n")
            f.write("\n")

    print(f"✓ 对比报告已生成: {report_path}")

    # === 7. 总结 ===
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"输出目录: {output_manager.output_dir}")
    print("\n数据文件:")
    print(f"  - lead_lag_raw.csv (原始收益传导关系)")
    print(f"  - lead_lag_excess.csv (超额收益传导关系)")
    print(f"  - comparison_report.md (对比分析报告)")
    print()

    # 清理
    analyzer.close()


if __name__ == '__main__':
    main()
