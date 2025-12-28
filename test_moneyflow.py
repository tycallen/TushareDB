#!/usr/bin/env python3
"""
测试 moneyflow 接口功能
验证下载、查询和数据质量
"""
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tushare_db.downloader import DataDownloader
from src.tushare_db.reader import DataReader
from dotenv import load_dotenv


def test_moneyflow():
    """测试 moneyflow 接口"""

    # 1. 加载环境变量
    load_dotenv()
    tushare_token = os.getenv('TUSHARE_TOKEN')
    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')

    if not tushare_token:
        raise ValueError("请在 .env 文件中设置 TUSHARE_TOKEN")

    print("=" * 60)
    print("测试 moneyflow 接口")
    print(f"数据库: {db_path}")
    print("=" * 60)

    # 2. 初始化下载器和读取器
    downloader = DataDownloader(
        tushare_token=tushare_token,
        db_path=db_path,
        rate_limit_profile="standard"
    )

    reader = DataReader(db_path=db_path)

    try:
        # 3. 测试下载功能 - 下载单日数据
        print("\n【测试1】下载单日全部股票资金流向数据")
        print("-" * 60)

        test_date = '20190315'  # 使用文档中的示例日期
        rows = downloader.download_moneyflow(trade_date=test_date)
        print(f"✓ 下载成功: {rows} 条记录")

        # 4. 测试查询功能 - 查询单日数据
        print(f"\n【测试2】查询单日全部股票资金流向数据 ({test_date})")
        print("-" * 60)

        df_day = reader.get_moneyflow(trade_date=test_date)
        print(f"✓ 查询成功: {len(df_day)} 条记录")
        print(f"\n数据预览 (前5条):")
        print(df_day.head())

        # 5. 测试下载单个股票数据
        print("\n【测试3】下载单个股票资金流向数据")
        print("-" * 60)

        test_stock = '002149.SZ'
        test_start = '20190115'
        test_end = '20190315'

        rows2 = downloader.download_moneyflow(
            ts_code=test_stock,
            start_date=test_start,
            end_date=test_end
        )
        print(f"✓ 下载 {test_stock} ({test_start} ~ {test_end}): {rows2} 条记录")

        # 6. 测试查询单个股票数据
        print(f"\n【测试4】查询单个股票资金流向数据 ({test_stock})")
        print("-" * 60)

        df_stock = reader.get_moneyflow(
            ts_code=test_stock,
            start_date=test_start,
            end_date=test_end
        )
        print(f"✓ 查询成功: {len(df_stock)} 条记录")
        print(f"\n{test_stock} 数据 ({test_start} ~ {test_end}):")
        print(df_stock[['trade_date', 'ts_code', 'buy_lg_amount', 'sell_lg_amount', 'net_mf_amount']].head(10))

        # 7. 验证数据完整性
        print("\n【测试5】验证数据完整性")
        print("-" * 60)

        # 检查字段
        expected_fields = [
            'ts_code', 'trade_date',
            'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
            'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
            'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
            'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
            'net_mf_vol', 'net_mf_amount'
        ]

        missing_fields = [f for f in expected_fields if f not in df_day.columns]
        if missing_fields:
            print(f"✗ 缺失字段: {missing_fields}")
        else:
            print("✓ 所有字段完整")

        # 8. 验证数据库中的总记录数
        print("\n【测试6】验证数据库中的总记录数")
        print("-" * 60)

        total_records = downloader.db.execute_query(
            "SELECT COUNT(*) as count FROM moneyflow"
        )
        print(f"✓ 数据库中 moneyflow 表总记录数: {total_records.iloc[0]['count']}")

        # 9. 验证主键约束（UPSERT）
        print("\n【测试7】验证主键约束")
        print("-" * 60)

        # 重复下载相同数据，应该触发 UPSERT
        rows3 = downloader.download_moneyflow(trade_date=test_date)
        print(f"✓ 重复下载（UPSERT测试）: {rows3} 条")

        total_records_after = downloader.db.execute_query(
            "SELECT COUNT(*) as count FROM moneyflow"
        )
        print(f"✓ 重复下载后总记录数: {total_records_after.iloc[0]['count']}")

        if total_records.iloc[0]['count'] == total_records_after.iloc[0]['count']:
            print("✓ 主键约束正常工作（记录数未增加，说明发生了UPSERT）")
        else:
            print("✗ 警告：主键约束可能有问题（记录数增加了）")

        # 10. 数据统计分析
        print("\n【测试8】数据统计分析")
        print("-" * 60)

        stats = downloader.db.execute_query('''
            SELECT
                COUNT(DISTINCT trade_date) as total_days,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                COUNT(*) as total_records
            FROM moneyflow
        ''')

        if not stats.empty:
            row = stats.iloc[0]
            print(f"\n数据统计:")
            print(f"  - 交易日数: {row['total_days']} 天")
            print(f"  - 起始日期: {row['earliest_date']}")
            print(f"  - 最新日期: {row['latest_date']}")
            print(f"  - 总记录数: {row['total_records']:,} 条")
            if row['total_days'] > 0:
                print(f"  - 平均每天: {row['total_records'] / row['total_days']:.0f} 条")

        print("\n" + "=" * 60)
        print("✓ 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        downloader.close()
        reader.close()


if __name__ == "__main__":
    test_moneyflow()
