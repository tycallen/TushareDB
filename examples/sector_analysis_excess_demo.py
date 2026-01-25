"""
申万行业板块关系分析示例 - 超额收益版本

使用超额收益方法剔除市场效应，识别真实的板块间传导关系
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
    print("申万行业板块关系分析 - 超额收益方法")
    print("=" * 70)

    # === 1. 初始化 ===
    analyzer = SectorAnalyzer('tushare.db')
    output_manager = OutputManager('output/excess_analysis')

    # === 2. 设置分析参数 ===
    params = {
        'start_date': '20240101',
        'end_date': '20241231',
        'level': 'L1',
        'period': 'daily',
        'min_correlation': 0.2,  # 传导关系阈值
        'max_lag': 5,
        'market_index': '000300.SH',  # 市场基准（会自动fallback到板块等权）
    }

    print(f"\n分析参数:")
    print(f"  数据范围: {params['start_date']} ~ {params['end_date']}")
    print(f"  分析层级: {params['level']}")
    print(f"  分析周期: {params['period']}")
    print(f"  相关阈值: {params['min_correlation']}")
    print(f"  最大滞后: {params['max_lag']} 天")
    print(f"  市场基准: {params['market_index']} (自动fallback到L1板块等权)")
    print()

    # === 3. 板块涨跌幅统计 ===
    print("=" * 70)
    print("【步骤1】计算板块涨跌幅")
    print("=" * 70)

    returns_df = analyzer.calculate_sector_returns(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period']
    )

    print(f"\n✓ 板块涨跌数据: {len(returns_df)} 条")
    print(f"  板块数量: {returns_df['sector_code'].nunique()}")
    print(f"  交易日数: {returns_df['trade_date'].nunique()}")

    # 保存
    output_manager.save_dataframe(returns_df, 'sector_returns', format='csv')

    # 统计
    avg_returns = returns_df.groupby('sector_code')['return'].mean().sort_values(ascending=False)
    print("\n板块平均涨跌幅 Top 10:")
    for sector, ret in avg_returns.head(10).items():
        sector_name = returns_df[returns_df['sector_code'] == sector]['sector_name'].iloc[0]
        print(f"  {sector_name}({sector}): {ret:.2f}%")

    # === 4. 相关性分析 ===
    print("\n" + "=" * 70)
    print("【步骤2】相关性分析")
    print("=" * 70)

    correlation_df = analyzer.calculate_correlation_with_pvalue(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period']
    )

    print(f"\n✓ 相关性数据: {len(correlation_df)} 对")
    print("\n相关性最强的板块对 Top 10:")
    top_corr = correlation_df.nlargest(10, 'correlation')
    for idx, row in top_corr.iterrows():
        print(f"  {row['sector_a_name']} ↔ {row['sector_b_name']}: {row['correlation']:.3f}")

    # 保存
    output_manager.save_dataframe(correlation_df, 'correlation_detail', format='csv')

    # === 5. 传导关系分析（超额收益方法）===
    print("\n" + "=" * 70)
    print("【步骤3】传导关系分析（超额收益方法，剔除市场效应）")
    print("=" * 70)

    lead_lag_df = analyzer.calculate_lead_lag_excess(
        start_date=params['start_date'],
        end_date=params['end_date'],
        max_lag=params['max_lag'],
        level=params['level'],
        period=params['period'],
        min_correlation=params['min_correlation'],
        market_index=params['market_index']
    )

    print(f"\n✓ 找到 {len(lead_lag_df)} 对超额收益传导关系")

    if len(lead_lag_df) > 0:
        print("\n传导性最强的板块对 Top 10:")
        top_lead_lag = lead_lag_df.nlargest(10, 'correlation')
        for idx, row in top_lead_lag.iterrows():
            print(f"  {row['sector_lead_name']} → {row['sector_lag_name']}: "
                  f"{row['lag_days']}天滞后, r={row['correlation']:.3f}")

        # 统计：哪些板块最常作为"领涨板块"
        lead_counts = lead_lag_df['sector_lead_name'].value_counts().head(10)
        print("\n最常见的领涨板块 Top 10:")
        for sector, count in lead_counts.items():
            print(f"  {sector}: {count} 次")

        # 保存
        output_manager.save_dataframe(lead_lag_df, 'lead_lag_excess', format='csv')
    else:
        print("\n未发现显著的超额收益传导关系")

    # === 6. 联动强度分析（Beta系数）===
    print("\n" + "=" * 70)
    print("【步骤4】联动强度分析（Beta系数）")
    print("=" * 70)

    linkage_df = analyzer.calculate_linkage_strength(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period']
    )

    print(f"\n✓ 联动关系: {len(linkage_df)} 对")
    print("\n联动最强的板块对 Top 10 (按R²排序):")
    top_linkage = linkage_df.nlargest(10, 'r_squared')
    for idx, row in top_linkage.iterrows():
        print(f"  {row['sector_a_name']} → {row['sector_b_name']}: "
              f"Beta={row['beta']:.3f}, R²={row['r_squared']:.3f}")

    # 保存
    output_manager.save_dataframe(linkage_df, 'linkage_strength', format='csv')

    # === 7. 生成分析报告 ===
    print("\n" + "=" * 70)
    print("【步骤5】生成分析报告")
    print("=" * 70)

    results = {
        'returns': returns_df,
        'lead_lag': lead_lag_df,
        'linkage': linkage_df,
    }

    # 注：不传入correlation，因为reporter期待矩阵格式而非detail格式

    # 添加方法说明到参数中
    params['method'] = '超额收益法（剔除市场效应）'

    report_path = output_manager.generate_report(
        results=results,
        metadata=params
    )

    print(f"\n✓ 分析报告已生成: {report_path}")

    # === 8. 总结 ===
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"输出目录: {output_manager.output_dir}")
    print("\n数据文件:")
    print(f"  - sector_returns.csv (板块涨跌数据)")
    print(f"  - correlation_detail.csv (相关性详情)")
    print(f"  - lead_lag_excess.csv (超额收益传导关系)")
    print(f"  - linkage_strength.csv (联动强度)")
    print(f"\n报告文件:")
    print(f"  - report.md (分析报告)")
    print("\n方法说明:")
    print(f"  本次分析使用超额收益方法，通过计算 板块收益-市场收益 来")
    print(f"  剔除市场整体波动的影响，识别真实的板块间产业链传导关系。")
    print()

    # 清理
    analyzer.close()


if __name__ == '__main__':
    main()
