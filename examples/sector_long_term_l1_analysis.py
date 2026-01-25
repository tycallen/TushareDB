"""
长期多时期板块传导分析 - L1一级板块版本

对比L1板块在不同年度的传导关系变化
"""

import sys
import os

# 复用L3分析脚本的逻辑，只修改配置
sys.path.insert(0, os.path.dirname(__file__))

from sector_long_term_l3_analysis import main as base_main
from sector_long_term_l3_analysis import *

def main():
    """主函数 - L1版本"""

    print("=" * 70)
    print("长期多时期板块传导分析 - L1一级板块")
    print("=" * 70)

    # === 配置参数 ===
    config = {
        'level': 'L1',  # L1一级板块
        'min_correlation': 0.2,
        'min_sector_stocks': 0,  # L1不需要过滤
        'periods': [
            {'name': '2020年', 'start': '20200101', 'end': '20201231'},
            {'name': '2021年', 'start': '20210101', 'end': '20211231'},
            {'name': '2022年', 'start': '20220101', 'end': '20221231'},
            {'name': '2023年', 'start': '20230101', 'end': '20231231'},
            {'name': '2024年', 'start': '20240101', 'end': '20241231'},
        ]
    }

    print(f"\n分析配置:")
    print(f"  板块层级: {config['level']} (31个一级行业)")
    print(f"  相关阈值: {config['min_correlation']}")
    print(f"  分析时期: {len(config['periods'])} 个年度 (2020-2024)")
    print()

    # === 初始化 ===
    analyzer = SectorAnalyzer('tushare.db')
    output_manager = OutputManager(f"output/long_term_{config['level']}_analysis")

    # === 分析各时期 ===
    period_results = []
    for period in config['periods']:
        result = analyze_period(
            analyzer=analyzer,
            start_date=period['start'],
            end_date=period['end'],
            level=config['level'],
            min_correlation=config['min_correlation'],
            period_name=period['name'],
            min_sector_stocks=config['min_sector_stocks']
        )
        period_results.append(result)

        # 保存单期结果
        if len(result['lead_lag']) > 0:
            filename = f"lead_lag_{period['name'].replace('年', '')}"
            output_manager.save_dataframe(result['lead_lag'], filename, format='csv')

    # === 跨期对比 ===
    stability_df = compare_periods(period_results)

    # 保存稳定性分析结果
    output_manager.save_dataframe(stability_df, 'stability_analysis', format='csv')

    # === 生成对比报告 ===
    report_path = generate_comparison_report(
        period_results=period_results,
        stability_df=stability_df,
        output_dir=output_manager.output_dir,
        level=config['level']
    )

    # === 总结 ===
    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}")
    print(f"输出目录: {output_manager.output_dir}")
    print(f"\n数据文件:")
    for period in config['periods']:
        filename = f"lead_lag_{period['name'].replace('年', '')}.csv"
        print(f"  - {filename} ({period['name']}传导关系)")
    print(f"  - stability_analysis.csv (稳定性分析)")
    print(f"\n报告文件:")
    print(f"  - long_term_comparison_report_{config['level']}.md")
    print()

    # 清理
    analyzer.close()


if __name__ == '__main__':
    main()
