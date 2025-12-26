#!/usr/bin/env python
"""
行业宽度计算演示
用实际数据展示计算过程
"""
from tushare_db import DataReader
import pandas as pd


def demo_calculation():
    """演示行业宽度的计算过程"""
    print("=" * 70)
    print("行业宽度计算过程演示")
    print("=" * 70)

    reader = DataReader()

    try:
        # 使用最近一个交易日的数据
        trade_date = '20251216'

        print(f"\n📅 交易日期: {trade_date}")
        print("=" * 70)

        # 步骤1：获取股票行业映射
        print("\n【步骤1】获取股票行业映射")
        print("-" * 70)

        stock_query = """
            SELECT ts_code, name, industry
            FROM stock_basic
            WHERE list_status = 'L'
              AND industry IS NOT NULL
              AND industry != ''
            LIMIT 10
        """
        stock_sample = reader.query(stock_query)
        print("\n示例数据（前10只股票）:")
        print(stock_sample.to_string(index=False))

        # 步骤2：获取当日涨跌数据
        print("\n\n【步骤2】获取当日涨跌数据")
        print("-" * 70)

        price_query = f"""
            SELECT pb.ts_code, sb.name, sb.industry, pb.pct_chg
            FROM pro_bar pb
            JOIN stock_basic sb ON pb.ts_code = sb.ts_code
            WHERE pb.trade_date = '{trade_date}'
              AND pb.pct_chg IS NOT NULL
              AND sb.industry IS NOT NULL
              AND sb.industry != ''
            LIMIT 20
        """
        price_sample = reader.query(price_query)
        price_sample['涨跌'] = price_sample['pct_chg'].apply(
            lambda x: '📈上涨' if x > 0 else '📉下跌' if x < 0 else '➡️平盘'
        )
        print("\n示例数据（前20只股票的涨跌情况）:")
        print(price_sample[['ts_code', 'name', 'industry', 'pct_chg', '涨跌']].to_string(index=False))

        # 步骤3：按行业统计
        print("\n\n【步骤3】按行业统计上涨股票占比")
        print("-" * 70)

        # 获取完整数据
        full_query = f"""
            SELECT pb.ts_code, sb.industry, pb.pct_chg
            FROM pro_bar pb
            JOIN stock_basic sb ON pb.ts_code = sb.ts_code
            WHERE pb.trade_date = '{trade_date}'
              AND pb.pct_chg IS NOT NULL
              AND sb.industry IS NOT NULL
              AND sb.industry != ''
        """
        full_data = reader.query(full_query)

        # 按行业统计
        industry_stats = full_data.groupby('industry').agg(
            总股票数=('pct_chg', 'count'),
            上涨数量=('pct_chg', lambda x: (x > 0).sum()),
            下跌数量=('pct_chg', lambda x: (x < 0).sum()),
            平均涨幅=('pct_chg', 'mean')
        )

        # 计算宽度得分
        industry_stats['宽度得分'] = (
            industry_stats['上涨数量'] / industry_stats['总股票数'] * 100
        ).round(1)

        # 排序并显示
        industry_stats = industry_stats.sort_values('宽度得分', ascending=False)

        print("\n各行业详细统计:")
        print(industry_stats.to_string())

        # 步骤4：计算市场总分
        print("\n\n【步骤4】计算市场总分")
        print("-" * 70)

        total_score = industry_stats['宽度得分'].sum()
        avg_score = industry_stats['宽度得分'].mean()
        industry_count = len(industry_stats)

        print(f"\n行业数量: {industry_count}")
        print(f"市场总分: {total_score:.0f}")
        print(f"平均得分: {avg_score:.1f}")

        # 计算公式展示
        print(f"\n💡 计算公式:")
        print(f"   市场总分 = Σ(所有行业宽度得分)")
        print(f"           = {' + '.join([f'{s:.0f}' for s in industry_stats['宽度得分'].head(5).values])} + ...")
        print(f"           = {total_score:.0f}")

        # 详细案例分析
        print("\n\n【案例分析】以\"银行\"行业为例")
        print("-" * 70)

        if '银行' in industry_stats.index:
            bank_stats = industry_stats.loc['银行']
            print(f"\n银行行业统计:")
            print(f"  总股票数: {bank_stats['总股票数']:.0f}")
            print(f"  上涨数量: {bank_stats['上涨数量']:.0f}")
            print(f"  下跌数量: {bank_stats['下跌数量']:.0f}")
            print(f"  平均涨幅: {bank_stats['平均涨幅']:.2f}%")
            print(f"  宽度得分: {bank_stats['宽度得分']:.1f}")

            print(f"\n计算过程:")
            print(f"  宽度得分 = (上涨数量 / 总股票数) × 100")
            print(f"          = ({bank_stats['上涨数量']:.0f} / {bank_stats['总股票数']:.0f}) × 100")
            print(f"          = {bank_stats['宽度得分']:.1f}")

            # 获取银行股详细数据
            bank_detail_query = f"""
                SELECT pb.ts_code, sb.name, pb.pct_chg
                FROM pro_bar pb
                JOIN stock_basic sb ON pb.ts_code = sb.ts_code
                WHERE pb.trade_date = '{trade_date}'
                  AND sb.industry = '银行'
                  AND pb.pct_chg IS NOT NULL
                ORDER BY pb.pct_chg DESC
            """
            bank_detail = reader.query(bank_detail_query)
            bank_detail['状态'] = bank_detail['pct_chg'].apply(
                lambda x: '✓ 上涨' if x > 0 else '✗ 下跌'
            )

            print(f"\n银行股详细数据:")
            print(bank_detail[['name', 'pct_chg', '状态']].to_string(index=False))

        # 市场状态判断
        print("\n\n【市场状态判断】")
        print("-" * 70)

        if avg_score >= 70:
            status = "🚀 普涨行情 - 市场极强"
            suggestion = "积极做多，把握机会"
        elif avg_score >= 60:
            status = "📈 多头市场 - 市场强势"
            suggestion = "持股为主，适度加仓"
        elif avg_score >= 50:
            status = "➡️ 震荡偏强 - 市场中性偏强"
            suggestion = "适度参与，精选个股"
        elif avg_score >= 40:
            status = "📉 震荡偏弱 - 市场中性偏弱"
            suggestion = "谨慎观望，控制仓位"
        else:
            status = "❌ 普跌行情 - 市场极弱"
            suggestion = "防守为主，降低仓位"

        print(f"\n市场状态: {status}")
        print(f"平均得分: {avg_score:.1f} (中性线为50)")
        print(f"操作建议: {suggestion}")

        # Top/Bottom 分析
        print(f"\n\n【强弱行业对比】")
        print("-" * 70)

        print(f"\n🔥 最强5个行业:")
        for i, (industry, row) in enumerate(industry_stats.head(5).iterrows(), 1):
            print(f"  {i}. {industry:10s} 得分:{row['宽度得分']:5.1f}  "
                  f"上涨:{row['上涨数量']:.0f}/{row['总股票数']:.0f}  "
                  f"均涨:{row['平均涨幅']:+.2f}%")

        print(f"\n❄️ 最弱5个行业:")
        for i, (industry, row) in enumerate(industry_stats.tail(5).iterrows(), 1):
            print(f"  {i}. {industry:10s} 得分:{row['宽度得分']:5.1f}  "
                  f"上涨:{row['上涨数量']:.0f}/{row['总股票数']:.0f}  "
                  f"均涨:{row['平均涨幅']:+.2f}%")

        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        print("\n详细的计算说明请查看: 行业宽度计算说明.md")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


if __name__ == '__main__':
    demo_calculation()
