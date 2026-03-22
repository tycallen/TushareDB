"""
补充 index_member_all 表的 PIT 历史数据

Tushare 的 index_member_all 接口：
- is_new='Y'：当前成员（out_date=NULL）
- is_new='N'：历史成员（out_date 有值，已被调出）

这个脚本会清空现有数据，重新完整下载（包括历史和当前），
确保 out_date 字段被正确填充，支持 Point-in-Time 回测。
"""

import os
import sys
import time

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tushare_db import DataDownloader
from tushare_db.duckdb_manager import DuckDBManager


def backfill_index_member_pit(db_path: str = None):
    """
    补充 index_member_all 的 PIT 历史数据
    """
    db_path = db_path or os.environ.get('DB_PATH', '/data10/tyc/quant/tushare.db')

    print(f"数据库路径: {db_path}")
    print("=" * 60)

    # 初始化下载器
    downloader = DataDownloader(
        db_path=db_path,
        rate_limit_profile="standard"
    )

    try:
        # 1. 清空现有 index_member_all 数据
        # （因为之前的 is_new='Y' 数据缺少 out_date，需要重新下载）
        print("1. 清空现有 index_member_all 数据...")
        if downloader.db.table_exists('index_member_all'):
            downloader.db.execute_query("DELETE FROM index_member_all")
            print("   ✓ 已清空")
        else:
            print("   ℹ 表不存在，将新建")

        # 2. 清除缓存元数据，强制重新下载
        print("\n2. 清除更新缓存...")
        downloader.db.execute_query(
            "DELETE FROM _tushare_cache_metadata WHERE table_name = 'index_member_all'"
        )
        print("   ✓ 已清除")

        # 3. 获取所有申万 L2 和 L3 行业
        print("\n3. 获取申万行业列表...")
        industries_df = downloader.db.execute_query(
            "SELECT index_code, industry_name, level FROM index_classify "
            "WHERE src = 'SW2021' AND level IN ('L2', 'L3') "
            "ORDER BY level, index_code"
        )

        if industries_df.empty:
            print("   ✗ 未找到申万行业分类，请先运行 update_index_classify()")
            return

        print(f"   ✓ 共 {len(industries_df)} 个行业（L2+L3）")

        # 4. 逐个行业下载（Y 和 N）
        print("\n4. 开始下载行业成分数据...")
        print("   （同时下载 is_new=Y 当前成员 和 is_new=N 历史成员）")
        print("-" * 60)

        total_y = 0
        total_n = 0
        success_count = 0

        for idx, row in industries_df.iterrows():
            index_code = row['index_code']
            industry_name = row['industry_name']
            level = row['level']

            code_param = {f"l{level[-1].lower()}_code": index_code}

            try:
                # 下载当前成员
                rows_y = downloader.download_index_member_all(**code_param, is_new='Y')
                total_y += rows_y

                # 下载历史成员（PIT 关键）
                rows_n = downloader.download_index_member_all(**code_param, is_new='N')
                total_n += rows_n

                if rows_y > 0 or rows_n > 0:
                    success_count += 1
                    print(f"   [{idx+1:3d}/{len(industries_df)}] {level} {industry_name:12s} "
                          f"({index_code}): Y={rows_y:4d} + N={rows_n:4d} = {rows_y+rows_n:5d}")
                else:
                    print(f"   [{idx+1:3d}/{len(industries_df)}] {level} {industry_name:12s} "
                          f"({index_code}): 无数据")

                # 速率限制：避免触发 Tushare 限制
                time.sleep(0.1)

            except Exception as e:
                print(f"   [{idx+1:3d}/{len(industries_df)}] {level} {industry_name:12s} "
                      f"({index_code}): 失败 - {e}")
                continue

        print("-" * 60)
        print(f"\n5. 下载完成统计:")
        print(f"   成功行业: {success_count}/{len(industries_df)}")
        print(f"   当前成员 (is_new=Y): {total_y:,} 条")
        print(f"   历史成员 (is_new=N): {total_n:,} 条")
        print(f"   总计: {total_y + total_n:,} 条")

        # 6. 验证 out_date 数据
        print("\n6. 验证 out_date 数据...")
        result = downloader.db.execute_query("""
            SELECT
                COUNT(*) as total,
                COUNT(out_date) as has_out_date,
                SUM(CASE WHEN out_date IS NULL THEN 1 ELSE 0 END) as no_out_date
            FROM index_member_all
        """)

        total = result.iloc[0]['total']
        has_out_date = result.iloc[0]['has_out_date']
        no_out_date = result.iloc[0]['no_out_date']

        print(f"   总记录数: {total:,}")
        print(f"   有 out_date (历史记录): {has_out_date:,} ({100*has_out_date/total:.1f}%)")
        print(f"   无 out_date (当前成员): {no_out_date:,} ({100*no_out_date/total:.1f}%)")

        # 7. 显示几个有 out_date 的样本
        print("\n7. 历史变迁样本（有 out_date 的记录）:")
        samples = downloader.db.execute_query("""
            SELECT ts_code, name, l1_name, l3_name, in_date, out_date, is_new
            FROM index_member_all
            WHERE out_date IS NOT NULL
            ORDER BY out_date DESC
            LIMIT 10
        """)
        for _, row in samples.iterrows():
            print(f"   {row['ts_code']} {row['name']:10s} | "
                  f"{row['l1_name']}/{row['l3_name']:8s} | "
                  f"{row['in_date']} → {row['out_date']}")

        # 8. 更新缓存时间戳
        downloader.db.execute_query(
            "INSERT OR REPLACE INTO _tushare_cache_metadata (table_name, last_updated_timestamp) "
            "VALUES ('index_member_all', ?)",
            [time.time()]
        )

        print("\n" + "=" * 60)
        print("✓ PIT 历史数据补充完成！")
        print("=" * 60)

    finally:
        downloader.close()


if __name__ == "__main__":
    # 检查环境变量
    if not os.environ.get('TUSHARE_TOKEN'):
        print("错误: 未设置 TUSHARE_TOKEN 环境变量")
        print("请先设置: export TUSHARE_TOKEN='your_token_here'")
        sys.exit(1)

    backfill_index_member_pit()
