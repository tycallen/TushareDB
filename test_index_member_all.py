#!/usr/bin/env python3
"""
测试 index_member_all 接口功能
验证下载、查询和历史回测功能
"""
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tushare_db.downloader import DataDownloader
from src.tushare_db.reader import DataReader
from dotenv import load_dotenv


def test_index_member_all():
    """测试 index_member_all 接口"""

    # 1. 加载环境变量
    load_dotenv()
    tushare_token = os.getenv('TUSHARE_TOKEN')
    db_path = os.getenv('DUCKDB_PATH', 'data/tushare.duckdb')

    if not tushare_token:
        raise ValueError("请在 .env 文件中设置 TUSHARE_TOKEN")

    print("=" * 60)
    print("测试 index_member_all 接口")
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
        # 3. 确保 index_classify 数据存在
        print("\n【准备】检查并下载申万行业分类数据")
        print("-" * 60)

        if not downloader.db.table_exists('index_classify'):
            print("index_classify 表不存在，开始下载...")
            classify_rows = downloader.download_index_classify(src='SW2021')
            print(f"✓ 下载申万行业分类: {classify_rows} 条记录")
        else:
            print("✓ index_classify 表已存在")

        # 获取一个L3行业指数代码进行测试
        industries_df = downloader.db.execute_query(
            "SELECT index_code, industry_name FROM index_classify "
            "WHERE src='SW2021' AND level='L3' LIMIT 1"
        )

        if industries_df.empty:
            raise ValueError("没有找到申万L3行业数据")

        test_l3_code = industries_df.iloc[0]['index_code']  # 使用 index_code (如 801780.SI)
        test_l3_name = industries_df.iloc[0]['industry_name']

        print(f"✓ 使用测试行业: {test_l3_name} ({test_l3_code})")

        # 4. 测试下载功能
        print(f"\n【测试1】下载 {test_l3_name} L3行业成分股")
        print("-" * 60)

        rows = downloader.download_index_member_all(l3_code=test_l3_code)
        print(f"✓ 下载成功: {rows} 条记录")

        # 5. 测试查询功能
        print(f"\n【测试2】查询 {test_l3_name} 行业所有成分股（不限时间）")
        print("-" * 60)

        df = reader.get_index_member_all(l3_code=test_l3_code)
        print(f"✓ 查询成功: {len(df)} 条记录")
        print(f"\n数据预览 (前5条):")
        print(df.head())

        # 6. 测试历史回测功能
        print(f"\n【测试3】历史回测：查询2023-01-01时的 {test_l3_name} 行业成分股")
        print("-" * 60)

        df_historical = reader.get_index_member_all(
            l3_code=test_l3_code,
            trade_date='20230101'
        )
        print(f"✓ 查询成功: {len(df_historical)} 条记录")
        print(f"\n历史成分股 (2023-01-01):")
        print(df_historical[['ts_code', 'name', 'in_date', 'out_date']].head(10))

        # 7. 测试当前成分股（is_new='Y'）
        print("\n【测试4】查询当前最新成分股（is_new='Y'）")
        print("-" * 60)

        df_current = reader.get_index_member_all(
            l3_code=test_l3_code,
            is_new='Y'
        )
        print(f"✓ 查询成功: {len(df_current)} 条当前成分股")
        print(f"\n当前成分股 (is_new='Y'):")
        print(df_current[['ts_code', 'name', 'in_date', 'out_date']].head(10))

        # 8. 验证数据库中的记录数
        print("\n【测试5】验证数据库中的总记录数")
        print("-" * 60)

        total_records = downloader.db.execute_query(
            f"SELECT COUNT(*) as count FROM index_member_all WHERE l3_code='{test_l3_code}'"
        )
        print(f"✓ 数据库中 {test_l3_name} 行业总记录数: {total_records.iloc[0]['count']}")

        # 9. 检查主键是否生效
        print("\n【测试6】验证主键约束")
        print("-" * 60)

        # 尝试重复下载，应该触发 UPSERT
        rows2 = downloader.download_index_member_all(l3_code=test_l3_code)
        print(f"✓ 重复下载（UPSERT测试）: {rows2} 条")

        total_records_after = downloader.db.execute_query(
            f"SELECT COUNT(*) as count FROM index_member_all WHERE l3_code='{test_l3_code}'"
        )
        print(f"✓ 重复下载后总记录数: {total_records_after.iloc[0]['count']}")

        if total_records.iloc[0]['count'] == total_records_after.iloc[0]['count']:
            print("✓ 主键约束正常工作（记录数未增加，说明发生了UPSERT）")
        else:
            print("✗ 警告：主键约束可能有问题（记录数增加了）")

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
    test_index_member_all()
