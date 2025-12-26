"""
市场宽度图测试脚本（不显示图形）
"""
import datetime
from tushare_db import DataReader
from market_width import get_industry_width
import matplotlib
matplotlib.use('Agg')  # 不显示图形，只生成数据


def test_market_width():
    """测试市场宽度计算"""
    print("=" * 60)
    print("测试市场宽度计算功能")
    print("=" * 60)

    reader = DataReader()
    try:
        # 测试参数：最近20个交易日
        end_date = '20251216'  # 使用数据库中最新的日期
        days = 20
        min_stocks = 30  # 至少30只股票的行业

        print(f"\n测试参数:")
        print(f"  结束日期: {end_date}")
        print(f"  交易日数: {days}")
        print(f"  最小行业股票数: {min_stocks}")
        print()

        # 计算市场宽度
        df = get_industry_width(reader, end_date, days, min_stocks_per_industry=min_stocks)

        # 显示结果
        print("\n" + "=" * 60)
        print("计算结果:")
        print("=" * 60)
        print(f"\n数据维度: {df.shape[0]} 天 x {df.shape[1]} 列（含总分）")
        print(f"\n包含的行业（{len(df.columns)-1}个）:")
        industries = [col for col in df.columns if col != '总分']
        for i, ind in enumerate(industries, 1):
            print(f"  {i:2d}. {ind}")

        print("\n最近5天的数据:")
        print(df.tail(5).to_string())

        print("\n总分统计:")
        print(df['总分'].describe())

        print("\n各行业平均上涨占比:")
        industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
        for industry, avg in industry_avg.items():
            print(f"  {industry}: {avg:.1f}%")

        print("\n" + "=" * 60)
        print("测试成功！")
        print("=" * 60)

        return df

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        reader.close()


if __name__ == '__main__':
    test_market_width()
