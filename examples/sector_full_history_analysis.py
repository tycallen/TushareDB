"""
板块传导关系全历史分析（2000-2026）

本脚本对全部历史数据进行分段分析，识别跨越多个市场周期的稳定传导关系。

市场分期：
- 2000-2005: 股权分置改革前
- 2006-2008: 牛市及金融危机
- 2009-2012: 四万亿刺激及后续
- 2013-2015: 创业板牛市及股灾
- 2016-2018: 供给侧改革
- 2019-2021: 疫情冲击及恢复
- 2022-2024: 经济转型期
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader
from tushare_db.sector_analysis import SectorAnalyzer, OutputManager

def main():
    """执行全历史分析"""

    # 初始化
    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))
    analyzer = SectorAnalyzer(db_path)

    # 定义分析时期（按市场周期划分）
    periods = [
        {
            'name': '2000-2005年',
            'start': '20000101',
            'end': '20051231',
            'description': '股权分置改革前'
        },
        {
            'name': '2006-2008年',
            'start': '20060101',
            'end': '20081231',
            'description': '牛市及金融危机'
        },
        {
            'name': '2009-2012年',
            'start': '20090101',
            'end': '20121231',
            'description': '四万亿刺激及后续'
        },
        {
            'name': '2013-2015年',
            'start': '20130101',
            'end': '20151231',
            'description': '创业板牛市及股灾'
        },
        {
            'name': '2016-2018年',
            'start': '20160101',
            'end': '20181231',
            'description': '供给侧改革'
        },
        {
            'name': '2019-2021年',
            'start': '20190101',
            'end': '20211231',
            'description': '疫情冲击及恢复'
        },
        {
            'name': '2022-2024年',
            'start': '20220101',
            'end': '20241231',
            'description': '经济转型期'
        }
    ]

    # 分析参数
    level = 'L1'  # 先分析L1，便于观察长期趋势
    period_type = 'daily'
    min_correlation = 0.20
    max_lag = 5

    print("=" * 80)
    print(f"板块传导关系全历史分析 - {level}层级")
    print("=" * 80)
    print(f"分析方法: 超额收益法（剔除市场效应）")
    print(f"最小相关系数: {min_correlation}")
    print(f"最大滞后期: {max_lag}天")
    print(f"分析时期数: {len(periods)}")
    print()

    # 存储每个时期的结果
    all_periods_results = []

    # 遍历每个时期
    for idx, period in enumerate(periods, 1):
        print(f"[{idx}/{len(periods)}] 分析时期: {period['name']} ({period['description']})")
        print(f"  日期范围: {period['start']} ~ {period['end']}")

        try:
            # 计算传导关系
            lead_lag_df = analyzer.calculate_lead_lag_excess(
                start_date=period['start'],
                end_date=period['end'],
                max_lag=max_lag,
                level=level,
                period=period_type,
                min_correlation=min_correlation
            )

            # 添加时期标识
            lead_lag_df['period_name'] = period['name']
            lead_lag_df['period_start'] = period['start']
            lead_lag_df['period_end'] = period['end']

            print(f"  找到传导对: {len(lead_lag_df)}")

            if len(lead_lag_df) > 0:
                # 显示最强的3个传导关系
                top_3 = lead_lag_df.nlargest(3, 'correlation')
                print("  Top 3 传导关系:")
                for _, row in top_3.iterrows():
                    lead_name = row.get('sector_lead_name', row['sector_lead'])
                    lag_name = row.get('sector_lag_name', row['sector_lag'])
                    print(f"    {lead_name} → {lag_name}: "
                          f"r={row['correlation']:.3f}, lag={row['lag_days']}天")
            else:
                print("  未找到显著传导关系")

            all_periods_results.append(lead_lag_df)
            print()

        except Exception as e:
            print(f"  ⚠️  分析失败: {str(e)}")
            print()
            continue

    # 合并所有时期的结果
    if not all_periods_results:
        print("所有时期均未找到传导关系，分析结束。")
        return

    combined_df = pd.concat(all_periods_results, ignore_index=True)

    # 稳定性分析
    print("=" * 80)
    print("稳定性分析")
    print("=" * 80)

    # 创建传导对的唯一标识
    combined_df['pair_id'] = (
        combined_df['sector_lead'].astype(str) + '_' +
        combined_df['sector_lag'].astype(str)
    )

    # 统计每个传导对出现的次数
    pair_counts = combined_df.groupby('pair_id').agg({
        'period_name': lambda x: list(x),
        'sector_lead': 'first',
        'sector_lag': 'first',
        'sector_lead_name': 'first',
        'sector_lag_name': 'first',
        'correlation': 'mean',
        'lag_days': 'mean'
    }).reset_index()

    pair_counts['appearance_count'] = pair_counts['period_name'].apply(len)
    pair_counts = pair_counts.sort_values('appearance_count', ascending=False)

    total_periods = len(periods)
    stable_all = pair_counts[pair_counts['appearance_count'] == total_periods]
    stable_half = pair_counts[pair_counts['appearance_count'] >= total_periods // 2]

    print(f"\n分析时期数: {total_periods}")
    print(f"总传导对数: {len(pair_counts)}")
    print(f"出现在所有时期: {len(stable_all)} 对")
    print(f"出现在>=一半时期: {len(stable_half)} 对")
    print()

    # 显示跨时期最稳定的传导关系
    if len(stable_all) > 0:
        print("\n🔥 跨越全部时期的超稳定传导关系：")
        print("-" * 80)
        for idx, row in stable_all.head(20).iterrows():
            lead_name = row.get('sector_lead_name', row['sector_lead'])
            lag_name = row.get('sector_lag_name', row['sector_lag'])
            periods_str = ', '.join(row['period_name'])
            print(f"  {lead_name} → {lag_name}")
            print(f"    平均相关系数: {row['correlation']:.3f}")
            print(f"    平均滞后天数: {row['lag_days']:.1f}")
            print(f"    出现时期: {periods_str}")
            print()
    else:
        print("\n⚠️  没有在所有时期均出现的传导关系")

    if len(stable_half) > 0:
        print(f"\n📊 出现在 ≥{total_periods // 2} 个时期的稳定传导关系（前20个）：")
        print("-" * 80)
        display_df = stable_half.head(20).copy()
        display_df['lead_display'] = display_df.apply(
            lambda x: f"{x.get('sector_lead_name', x['sector_lead'])}", axis=1
        )
        display_df['lag_display'] = display_df.apply(
            lambda x: f"{x.get('sector_lag_name', x['sector_lag'])}", axis=1
        )
        display_df['periods_display'] = display_df['period_name'].apply(
            lambda x: ', '.join(x)
        )

        for idx, row in display_df.iterrows():
            print(f"  {row['lead_display']} → {row['lag_display']}")
            print(f"    出现次数: {row['appearance_count']}/{total_periods}")
            print(f"    平均相关系数: {row['correlation']:.3f}")
            print(f"    平均滞后天数: {row['lag_days']:.1f}")
            print(f"    时期: {row['periods_display']}")
            print()

    # 保存详细结果
    output_dir = project_root / "output" / "full_history_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始数据
    combined_df.to_csv(output_dir / f"full_history_{level}_all_periods.csv",
                       index=False, encoding='utf-8-sig')

    # 保存稳定性分析结果
    pair_counts.to_csv(output_dir / f"full_history_{level}_stability.csv",
                       index=False, encoding='utf-8-sig')

    # 生成Markdown报告
    generate_report(
        periods=periods,
        combined_df=combined_df,
        pair_counts=pair_counts,
        stable_all=stable_all,
        stable_half=stable_half,
        level=level,
        min_correlation=min_correlation,
        output_dir=output_dir
    )

    print(f"\n✅ 分析完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   - 原始数据: full_history_{level}_all_periods.csv")
    print(f"   - 稳定性数据: full_history_{level}_stability.csv")
    print(f"   - 分析报告: full_history_report_{level}.md")


def generate_report(periods, combined_df, pair_counts, stable_all, stable_half,
                   level, min_correlation, output_dir):
    """生成Markdown报告"""

    report_lines = []
    report_lines.append(f"# 板块传导关系全历史分析报告 - {level}层级")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"**分析方法**: 超额收益法（剔除市场效应）")
    report_lines.append(f"**最小相关系数**: {min_correlation}")
    report_lines.append("")

    # 1. 各时期传导关系数量
    report_lines.append("## 各时期传导关系数量")
    report_lines.append("")
    report_lines.append("| 时期 | 起止日期 | 市场特征 | 传导对数 |")
    report_lines.append("|------|---------|---------|----------|")

    for period in periods:
        period_data = combined_df[combined_df['period_name'] == period['name']]
        count = len(period_data)
        date_range = f"{period['start'][:4]}/{period['start'][4:6]}/{period['start'][6:]}~{period['end'][:4]}/{period['end'][4:6]}/{period['end'][6:]}"
        report_lines.append(f"| {period['name']} | {date_range} | {period['description']} | {count} |")

    report_lines.append("")

    # 2. 稳定性统计
    total_periods = len(periods)
    report_lines.append("## 稳定性分析")
    report_lines.append("")
    report_lines.append(f"- **分析时期数**: {total_periods}")
    report_lines.append(f"- **总传导对数**: {len(pair_counts)}")
    report_lines.append(f"- **出现在所有时期**: {len(stable_all)} 对")
    report_lines.append(f"- **出现在≥一半时期**: {len(stable_half)} 对")
    report_lines.append("")

    # 3. 超稳定传导关系
    if len(stable_all) > 0:
        report_lines.append("## 超稳定传导关系（所有时期均出现）")
        report_lines.append("")
        report_lines.append("| 排名 | 领涨板块 | 跟随板块 | 出现次数 | 平均相关系数 | 平均滞后天数 |")
        report_lines.append("|------|----------|----------|----------|--------------|-------------|")

        for idx, row in stable_all.head(20).iterrows():
            lead_name = row.get('sector_lead_name', row['sector_lead'])
            lag_name = row.get('sector_lag_name', row['sector_lag'])
            report_lines.append(
                f"| {idx+1} | {lead_name} | {lag_name} | "
                f"{row['appearance_count']}/{total_periods} | "
                f"{row['correlation']:.3f} | {row['lag_days']:.1f} |"
            )
        report_lines.append("")
    else:
        report_lines.append("## 超稳定传导关系")
        report_lines.append("")
        report_lines.append("**⚠️  没有在所有时期均出现的传导关系**，说明市场结构变化巨大。")
        report_lines.append("")

    # 4. 较稳定传导关系
    if len(stable_half) > 0:
        report_lines.append(f"## 较稳定传导关系（出现在≥{total_periods // 2}个时期）")
        report_lines.append("")
        report_lines.append("| 排名 | 领涨板块 | 跟随板块 | 出现次数 | 出现时期 | 平均相关系数 | 平均滞后天数 |")
        report_lines.append("|------|----------|----------|----------|----------|--------------|-------------|")

        for idx, row in stable_half.head(30).iterrows():
            lead_name = row.get('sector_lead_name', row['sector_lead'])
            lag_name = row.get('sector_lag_name', row['sector_lag'])
            periods_str = ','.join(row['period_name'])
            report_lines.append(
                f"| {idx+1} | {lead_name} | {lag_name} | "
                f"{row['appearance_count']}/{total_periods} | {periods_str} | "
                f"{row['correlation']:.3f} | {row['lag_days']:.1f} |"
            )
        report_lines.append("")

    # 5. 各时期最强传导关系
    report_lines.append("## 各时期最强传导关系 Top 5")
    report_lines.append("")

    for period in periods:
        period_data = combined_df[combined_df['period_name'] == period['name']]

        if len(period_data) == 0:
            continue

        report_lines.append(f"### {period['name']} ({period['description']})")
        report_lines.append("")
        report_lines.append("| 排名 | 领涨板块 | 跟随板块 | 滞后天数 | 相关系数 |")
        report_lines.append("|------|----------|----------|----------|----------|")

        top_5 = period_data.nlargest(5, 'correlation')
        for idx, row in top_5.iterrows():
            lead_name = row.get('sector_lead_name', row['sector_lead'])
            lag_name = row.get('sector_lag_name', row['sector_lag'])
            report_lines.append(
                f"| {idx+1} | {lead_name} | {lag_name} | "
                f"{row['lag_days']} | {row['correlation']:.3f} |"
            )
        report_lines.append("")

    # 6. 结论
    report_lines.append("## 分析结论")
    report_lines.append("")

    if len(stable_all) == 0:
        report_lines.append("1. **长期稳定性极低**：没有传导关系在全部7个时期（26年）均出现，")
        report_lines.append("   说明A股市场结构经历了巨大变化，传导模式随市场周期剧烈波动。")
        report_lines.append("")

    if len(stable_half) > 0:
        report_lines.append(f"2. **部分模式有中期稳定性**：{len(stable_half)}个传导对出现在≥{total_periods // 2}个时期，")
        report_lines.append("   这些关系可能反映了较为稳定的产业链联系或资金轮动模式。")
        report_lines.append("")

    # 分析各时期传导数量变化
    period_counts = []
    for period in periods:
        period_data = combined_df[combined_df['period_name'] == period['name']]
        period_counts.append(len(period_data))

    report_lines.append("3. **市场周期特征明显**：")
    for i, (period, count) in enumerate(zip(periods, period_counts)):
        if count > 0:
            report_lines.append(f"   - {period['name']}: {count}对传导关系（{period['description']}）")
    report_lines.append("")

    report_lines.append("4. **投资启示**：")
    report_lines.append("   - 板块传导关系具有明显的时代特征，不应盲目使用历史模式")
    report_lines.append("   - 需要结合当前市场环境和产业政策，动态调整分析框架")
    report_lines.append("   - 稳定性较高的传导关系（出现≥4个时期）可作为重点关注对象")
    report_lines.append("")

    # 写入文件
    report_path = output_dir / f"full_history_report_{level}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


if __name__ == '__main__':
    main()
