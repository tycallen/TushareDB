#!/usr/bin/env python
"""
个股资金流向（moneyflow_dc）接口测试脚本
"""
from tushare_db import DataDownloader, DataReader
import pandas as pd


def test_download_moneyflow_dc():
    """测试下载个股资金流向数据"""
    print("=" * 70)
    print("测试下载个股资金流向数据")
    print("=" * 70)

    downloader = DataDownloader()

    try:
        # 测试1：下载单日全部股票数据
        print("\n【测试1】下载单日全部股票数据")
        print("-" * 70)
        trade_date = '20241011'
        print(f"下载日期: {trade_date}")

        rows = downloader.download_moneyflow_dc(trade_date=trade_date)
        print(f"✓ 成功下载 {rows} 条数据")

        # 测试2：下载单个股票的数据范围
        print("\n【测试2】下载单个股票的数据范围")
        print("-" * 70)
        ts_code = '002149.SZ'
        start_date = '20240901'
        end_date = '20240913'
        print(f"股票代码: {ts_code}")
        print(f"日期范围: {start_date} - {end_date}")

        rows = downloader.download_moneyflow_dc(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        print(f"✓ 成功下载 {rows} 条数据")

        print("\n" + "=" * 70)
        print("下载测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_query_moneyflow_dc():
    """测试查询个股资金流向数据"""
    print("\n" * 2)
    print("=" * 70)
    print("测试查询个股资金流向数据")
    print("=" * 70)

    reader = DataReader()

    try:
        # 测试1：查询单日数据
        print("\n【测试1】查询单日所有股票的资金流向")
        print("-" * 70)
        trade_date = '20241011'
        df = reader.get_moneyflow_dc(trade_date=trade_date)

        if not df.empty:
            print(f"查询结果: {len(df)} 条数据")
            print(f"\n前5条数据:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(df.head())

            # 统计信息
            print(f"\n数据统计:")
            print(f"  主力净流入为正的股票数: {(df['net_amount'] > 0).sum()}")
            print(f"  主力净流入为负的股票数: {(df['net_amount'] < 0).sum()}")
            print(f"  主力净流入总额: {df['net_amount'].sum():.2f} 万元")

            # Top 10 主力净流入
            print(f"\n主力净流入 Top 10:")
            top10 = df.nlargest(10, 'net_amount')[['ts_code', 'name', 'pct_change', 'net_amount', 'net_amount_rate']]
            print(top10.to_string(index=False))
        else:
            print("✗ 未查询到数据")

        # 测试2：查询单个股票的历史数据
        print("\n【测试2】查询单个股票的历史资金流向")
        print("-" * 70)
        ts_code = '002149.SZ'
        start_date = '20240901'
        end_date = '20240913'

        df = reader.get_moneyflow_dc(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if not df.empty:
            print(f"股票: {ts_code}")
            print(f"日期范围: {start_date} - {end_date}")
            print(f"查询结果: {len(df)} 条数据\n")

            # 显示关键字段
            display_cols = [
                'trade_date', 'name', 'pct_change', 'close',
                'net_amount', 'net_amount_rate',
                'buy_elg_amount', 'buy_lg_amount',
                'buy_md_amount', 'buy_sm_amount'
            ]
            print(df[display_cols].to_string(index=False))

            # 统计分析
            print(f"\n数据分析:")
            print(f"  期间主力净流入总额: {df['net_amount'].sum():.2f} 万元")
            print(f"  主力净流入为正的天数: {(df['net_amount'] > 0).sum()} / {len(df)} 天")
            print(f"  主力净流入平均值: {df['net_amount'].mean():.2f} 万元")
            print(f"  主力净流入最大值: {df['net_amount'].max():.2f} 万元 ({df.loc[df['net_amount'].idxmax(), 'trade_date']})")
            print(f"  主力净流入最小值: {df['net_amount'].min():.2f} 万元 ({df.loc[df['net_amount'].idxmin(), 'trade_date']})")
        else:
            print("✗ 未查询到数据")

        # 测试3：数据字段完整性检查
        print("\n【测试3】数据字段完整性检查")
        print("-" * 70)

        expected_columns = [
            'trade_date', 'ts_code', 'name', 'pct_change', 'close',
            'net_amount', 'net_amount_rate',
            'buy_elg_amount', 'buy_elg_amount_rate',
            'buy_lg_amount', 'buy_lg_amount_rate',
            'buy_md_amount', 'buy_md_amount_rate',
            'buy_sm_amount', 'buy_sm_amount_rate'
        ]

        df_sample = reader.get_moneyflow_dc(trade_date='20241011')
        if not df_sample.empty:
            print(f"数据库中的字段:")
            for i, col in enumerate(df_sample.columns, 1):
                status = "✓" if col in expected_columns else "?"
                print(f"  {status} {i:2d}. {col}")

            missing_cols = set(expected_columns) - set(df_sample.columns)
            if missing_cols:
                print(f"\n缺少的字段: {', '.join(missing_cols)}")
            else:
                print(f"\n✓ 所有预期字段都存在")

        print("\n" + "=" * 70)
        print("查询测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


def main():
    """主函数"""
    print("\n")
    print("#" * 70)
    print("# 个股资金流向（moneyflow_dc）接口测试")
    print("#" * 70)

    # 测试下载功能
    test_download_moneyflow_dc()

    # 测试查询功能
    test_query_moneyflow_dc()

    print("\n")
    print("#" * 70)
    print("# 测试完成！")
    print("#" * 70)
    print("\n")


if __name__ == '__main__':
    main()
