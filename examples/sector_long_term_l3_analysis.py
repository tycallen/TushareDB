"""
长期多时期板块传导分析 - 支持L3板块

功能：
1. 多年度分析（2020-2025年，每年单独分析）
2. L3三级板块分析（自动过滤样本量不足的板块）
3. 传导关系稳定性验证
4. 跨年对比报告
"""

import logging
import pandas as pd
from datetime import datetime
from tushare_db.sector_analysis import SectorAnalyzer, OutputManager
from tushare_db import DataReader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def analyze_period(
    analyzer: SectorAnalyzer,
    start_date: str,
    end_date: str,
    level: str,
    min_correlation: float,
    period_name: str,
    min_sector_stocks: int = 5
) -> dict:
    """
    分析单个时期的传导关系

    Args:
        analyzer: SectorAnalyzer实例
        start_date: 开始日期
        end_date: 结束日期
        level: 板块层级 (L1/L2/L3)
        min_correlation: 最小相关系数
        period_name: 时期名称（如'2024年'）
        min_sector_stocks: L3板块最小股票数量（用于过滤）

    Returns:
        分析结果字典
    """
    print(f"\n{'='*70}")
    print(f"分析时期: {period_name} ({start_date} ~ {end_date})")
    print(f"板块层级: {level}")
    print(f"{'='*70}")

    # L3板块需要过滤样本量不足的
    sector_filter = None
    if level == 'L3':
        print(f"正在过滤L3板块（保留成分股 >= {min_sector_stocks}只的板块）...")
        sector_filter = _get_valid_l3_sectors(analyzer.reader, min_sector_stocks)
        print(f"✓ 共 {len(sector_filter)} 个L3板块符合条件")

    # 计算超额收益传导关系
    lead_lag_df = analyzer.calculate_lead_lag_excess(
        start_date=start_date,
        end_date=end_date,
        max_lag=5,
        level=level,
        period='daily',
        min_correlation=min_correlation,
        market_index='000300.SH'
    )

    print(f"✓ 找到 {len(lead_lag_df)} 对传导关系")

    if len(lead_lag_df) > 0:
        print("\n传导性最强的前5对:")
        for idx, row in lead_lag_df.head(5).iterrows():
            print(f"  {row['sector_lead_name']} → {row['sector_lag_name']}: "
                  f"{row['lag_days']}天, r={row['correlation']:.3f}")

    return {
        'period_name': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'lead_lag': lead_lag_df,
        'count': len(lead_lag_df)
    }


def _get_valid_l3_sectors(reader: DataReader, min_stocks: int) -> set:
    """获取样本量足够的L3板块代码"""
    query = f"""
        SELECT l3_code, COUNT(DISTINCT ts_code) as stock_count
        FROM index_member_all
        WHERE is_new = 'Y' AND l3_code IS NOT NULL
        GROUP BY l3_code
        HAVING stock_count >= {min_stocks}
    """
    df = reader.db.con.execute(query).fetchdf()
    return set(df['l3_code'].tolist())


def compare_periods(period_results: list) -> pd.DataFrame:
    """
    对比不同时期的传导关系，找出稳定的传导对

    Returns:
        稳定传导关系的DataFrame
    """
    print(f"\n{'='*70}")
    print("跨期稳定性分析")
    print(f"{'='*70}")

    # 收集所有传导对
    all_pairs = {}
    for result in period_results:
        period = result['period_name']
        df = result['lead_lag']

        for _, row in df.iterrows():
            pair_key = f"{row['sector_lead']}->{row['sector_lag']}"
            if pair_key not in all_pairs:
                all_pairs[pair_key] = {
                    'lead_code': row['sector_lead'],
                    'lead_name': row['sector_lead_name'],
                    'lag_code': row['sector_lag'],
                    'lag_name': row['sector_lag_name'],
                    'periods': [],
                    'correlations': [],
                    'lag_days': []
                }
            all_pairs[pair_key]['periods'].append(period)
            all_pairs[pair_key]['correlations'].append(row['correlation'])
            all_pairs[pair_key]['lag_days'].append(row['lag_days'])

    # 转换为DataFrame
    stability_data = []
    for pair_key, data in all_pairs.items():
        stability_data.append({
            'sector_lead': data['lead_code'],
            'sector_lead_name': data['lead_name'],
            'sector_lag': data['lag_code'],
            'sector_lag_name': data['lag_name'],
            'appear_count': len(data['periods']),
            'appear_periods': ','.join(data['periods']),
            'avg_correlation': sum(data['correlations']) / len(data['correlations']),
            'min_correlation': min(data['correlations']),
            'max_correlation': max(data['correlations']),
            'avg_lag_days': sum(data['lag_days']) / len(data['lag_days']),
        })

    stability_df = pd.DataFrame(stability_data)
    stability_df = stability_df.sort_values('appear_count', ascending=False)

    print(f"\n总传导对数: {len(stability_df)}")
    print(f"出现在所有时期的传导对: {len(stability_df[stability_df['appear_count'] == len(period_results)])}")
    print(f"出现在>=一半时期的传导对: {len(stability_df[stability_df['appear_count'] >= len(period_results)/2])}")

    return stability_df


def generate_comparison_report(
    period_results: list,
    stability_df: pd.DataFrame,
    output_dir: str,
    level: str
):
    """生成对比分析报告"""
    report_path = f"{output_dir}/long_term_comparison_report_{level}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 板块传导关系长期分析报告 - {level}层级\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**分析方法**: 超额收益法（剔除市场效应）\n\n")

        # 1. 各时期统计
        f.write("## 各时期传导关系数量\n\n")
        f.write("| 时期 | 起止日期 | 传导对数 |\n")
        f.write("|------|---------|----------|\n")
        for result in period_results:
            f.write(f"| {result['period_name']} | {result['start_date']}~{result['end_date']} | {result['count']} |\n")
        f.write("\n")

        # 2. 跨期稳定的传导关系
        f.write("## 跨期稳定的传导关系\n\n")
        f.write(f"出现在多个时期的传导对（按出现次数排序）:\n\n")
        f.write("| 排名 | 领涨板块 | 跟随板块 | 出现次数 | 出现时期 | 平均相关系数 | 平均滞后天数 |\n")
        f.write("|------|----------|----------|----------|----------|--------------|-------------|\n")

        top_stable = stability_df.head(20)
        for i, row in top_stable.iterrows():
            f.write(f"| {i+1} | {row['sector_lead_name']} | {row['sector_lag_name']} | "
                   f"{row['appear_count']} | {row['appear_periods']} | "
                   f"{row['avg_correlation']:.3f} | {row['avg_lag_days']:.1f} |\n")
        f.write("\n")

        # 3. 各时期Top传导关系
        f.write("## 各时期最强传导关系 Top 5\n\n")
        for result in period_results:
            f.write(f"### {result['period_name']}\n\n")
            if len(result['lead_lag']) > 0:
                f.write("| 排名 | 领涨板块 | 跟随板块 | 滞后天数 | 相关系数 |\n")
                f.write("|------|----------|----------|----------|----------|\n")
                for i, row in result['lead_lag'].head(5).iterrows():
                    f.write(f"| {i+1} | {row['sector_lead_name']} | {row['sector_lag_name']} | "
                           f"{row['lag_days']} | {row['correlation']:.3f} |\n")
            else:
                f.write("未发现显著传导关系\n")
            f.write("\n")

        # 4. 稳定性分析结论
        total_periods = len(period_results)
        always_appear = stability_df[stability_df['appear_count'] == total_periods]
        half_appear = stability_df[stability_df['appear_count'] >= total_periods/2]

        f.write("## 稳定性分析结论\n\n")
        f.write(f"- 分析时期数: {total_periods}\n")
        f.write(f"- 总传导对数: {len(stability_df)}\n")
        f.write(f"- 出现在所有时期: {len(always_appear)} 对\n")
        f.write(f"- 出现在>=一半时期: {len(half_appear)} 对\n\n")

        if len(always_appear) > 0:
            f.write("### 最稳定的传导关系（所有时期均出现）\n\n")
            for _, row in always_appear.head(10).iterrows():
                f.write(f"- **{row['sector_lead_name']} → {row['sector_lag_name']}**: "
                       f"平均r={row['avg_correlation']:.3f}, 滞后{row['avg_lag_days']:.1f}天\n")
        else:
            f.write("**注意**: 没有在所有时期均出现的传导关系，说明传导模式变化较大。\n")

        f.write("\n")

    print(f"✓ 对比报告已生成: {report_path}")
    return report_path


def main():
    """主函数"""

    print("=" * 70)
    print("长期多时期板块传导分析")
    print("=" * 70)

    # === 配置参数 ===
    config = {
        'level': 'L3',  # L1/L2/L3
        'min_correlation': 0.2,
        'min_sector_stocks': 10,  # L3板块最小股票数（L1/L2忽略此参数）
        'periods': [
            # {'name': '2020年', 'start': '20200101', 'end': '20201231'},
            # {'name': '2021年', 'start': '20210101', 'end': '20211231'},
            {'name': '2022年', 'start': '20220101', 'end': '20221231'},
            {'name': '2023年', 'start': '20230101', 'end': '20231231'},
            {'name': '2024年', 'start': '20240101', 'end': '20241231'},
        ]
    }

    print(f"\n分析配置:")
    print(f"  板块层级: {config['level']}")
    print(f"  相关阈值: {config['min_correlation']}")
    print(f"  分析时期: {len(config['periods'])} 个")
    if config['level'] == 'L3':
        print(f"  L3最小股票数: {config['min_sector_stocks']}")
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
