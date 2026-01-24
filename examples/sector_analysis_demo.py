"""
申万行业板块关系分析示例

演示完整的板块分析流程：
1. 计算板块涨跌幅
2. 相关性分析
3. 传导关系分析
4. 联动强度分析
5. 生成报告
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

    # === 1. 初始化 ===
    print("=" * 60)
    print("申万行业板块关系分析")
    print("=" * 60)

    # 创建分析器（使用本地数据库）
    analyzer = SectorAnalyzer('tushare.db')

    # 创建输出管理器
    output_manager = OutputManager()  # 默认保存到 output/YYYY-MM-DD_analysis/

    # === 2. 设置分析参数 ===
    params = {
        'start_date': '20240101',
        'end_date': '20241231',
        'level': 'L1',          # 申万一级行业
        'period': 'daily',      # 日线分析
    }

    print(f"\n分析参数:")
    print(f"  数据范围: {params['start_date']} ~ {params['end_date']}")
    print(f"  分析层级: {params['level']}")
    print(f"  分析周期: {params['period']}")
    print()

    # === 3. 计算板块涨跌幅 ===
    print("正在计算板块涨跌幅...")
    returns_df = analyzer.calculate_sector_returns(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period']
    )
    print(f"✓ 完成: {len(returns_df)} 条记录")

    # 保存涨跌幅数据
    output_manager.save_dataframe(returns_df, 'sector_returns', format='csv')

    # === 4. 相关性分析 ===
    print("\n正在计算相关性矩阵...")
    corr_matrix = analyzer.calculate_correlation_matrix(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period']
    )
    print(f"✓ 完成: {corr_matrix.shape[0]}×{corr_matrix.shape[1]} 矩阵")

    # 保存相关性矩阵
    output_manager.save_dataframe(corr_matrix, 'correlation_matrix', format='csv')

    # 计算带p值的相关性
    print("\n正在计算相关性详情（含p值）...")
    corr_detail = analyzer.calculate_correlation_with_pvalue(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period'],
        min_correlation=0.5  # 只看相关性>0.5的
    )
    print(f"✓ 找到 {len(corr_detail)} 对高相关板块")

    if len(corr_detail) > 0:
        output_manager.save_dataframe(corr_detail, 'correlation_detail', format='csv')

    # === 5. 传导关系分析 ===
    print("\n正在计算传导关系（滞后相关性）...")
    lead_lag_df = analyzer.calculate_lead_lag(
        start_date=params['start_date'],
        end_date=params['end_date'],
        max_lag=None,  # 自适应窗口（日线默认5天）
        level=params['level'],
        period=params['period'],
        min_correlation=0.3
    )
    print(f"✓ 找到 {len(lead_lag_df)} 对传导关系")

    if len(lead_lag_df) > 0:
        output_manager.save_dataframe(lead_lag_df, 'lead_lag_relationships', format='csv')

    # === 6. 联动强度分析 ===
    print("\n正在计算联动强度（Beta系数）...")
    linkage_df = analyzer.calculate_linkage_strength(
        start_date=params['start_date'],
        end_date=params['end_date'],
        level=params['level'],
        period=params['period'],
        min_r_squared=0.3
    )
    print(f"✓ 找到 {len(linkage_df)} 对联动关系")

    if len(linkage_df) > 0:
        output_manager.save_dataframe(linkage_df, 'linkage_strength', format='csv')

    # === 7. 生成报告 ===
    print("\n正在生成分析报告...")

    # 整理结果字典
    results = {
        'returns': returns_df,
        'correlation': corr_matrix,
    }

    if len(lead_lag_df) > 0:
        results['lead_lag'] = lead_lag_df

    if len(linkage_df) > 0:
        results['linkage'] = linkage_df

    # 生成报告
    report_path = output_manager.generate_report(results, metadata=params)
    print(f"✓ 报告已生成: {report_path}")

    # === 8. 总结 ===
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"输出目录: {output_manager.output_dir}")
    print("=" * 60)

    # 清理
    analyzer.close()


if __name__ == '__main__':
    main()
