#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复表主键约束

问题：某些表在创建时没有主键约束，导致 UPSERT 失败
解决：重建表，添加正确的主键约束

使用方法：
    python scripts/fix_table_pk.py [表名]

    # 修复 fina_indicator_vip 表
    python scripts/fix_table_pk.py fina_indicator_vip

    # 检查所有表的主键状态
    python scripts/fix_table_pk.py --check
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb

# 从 duckdb_manager 导入主键定义
from tushare_db.duckdb_manager import TABLE_PRIMARY_KEYS


def check_table_pk(con, table_name: str) -> bool:
    """检查表是否有主键约束"""
    result = con.execute(f"""
        SELECT constraint_type
        FROM duckdb_constraints()
        WHERE table_name = '{table_name}' AND constraint_type = 'PRIMARY KEY'
    """).fetchone()
    return result is not None


def fix_table_pk(db_path: str, table_name: str):
    """重建表以添加主键约束"""
    if table_name not in TABLE_PRIMARY_KEYS:
        print(f"错误: {table_name} 不在 TABLE_PRIMARY_KEYS 中定义")
        return False

    pk_columns = TABLE_PRIMARY_KEYS[table_name]
    pk_str = ", ".join([f'"{col}"' for col in pk_columns])

    con = duckdb.connect(db_path)

    try:
        # 检查表是否存在
        exists = con.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()[0]
        if not exists:
            print(f"表 {table_name} 不存在")
            return False

        # 检查是否已有主键
        if check_table_pk(con, table_name):
            print(f"表 {table_name} 已有主键约束，无需修复")
            return True

        # 获取当前行数
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"表 {table_name}: {count:,} 行，主键列: {pk_columns}")

        # 重建表
        print(f"正在重建表 {table_name}...")

        # 1. 创建临时表（带主键）
        con.execute(f"CREATE TABLE {table_name}_new AS SELECT * FROM {table_name}")

        # 2. 删除原表
        con.execute(f"DROP TABLE {table_name}")

        # 3. 从临时表重建，添加主键
        # 获取列定义
        columns_df = con.execute(f"PRAGMA table_info('{table_name}_new')").fetchdf()

        # 构建 CREATE TABLE 语句
        col_defs = []
        for _, row in columns_df.iterrows():
            col_name = row['name']
            col_type = row['type']
            col_defs.append(f'"{col_name}" {col_type}')

        create_sql = f"""
            CREATE TABLE {table_name} (
                {", ".join(col_defs)},
                PRIMARY KEY ({pk_str})
            )
        """
        con.execute(create_sql)

        # 4. 插入数据（去重）
        con.execute(f"""
            INSERT INTO {table_name}
            SELECT DISTINCT ON ({pk_str}) * FROM {table_name}_new
        """)

        # 5. 删除临时表
        con.execute(f"DROP TABLE {table_name}_new")

        # 验证
        new_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        has_pk = check_table_pk(con, table_name)

        print(f"✓ 重建完成: {new_count:,} 行，主键约束: {'已添加' if has_pk else '失败'}")

        if count != new_count:
            print(f"  注意: 去重后减少了 {count - new_count:,} 行重复数据")

        return True

    except Exception as e:
        print(f"错误: {e}")
        return False
    finally:
        con.close()


def check_all_tables(db_path: str):
    """检查所有表的主键状态"""
    con = duckdb.connect(db_path, read_only=True)

    try:
        # 获取所有表
        tables = con.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name NOT LIKE '_%'
        """).fetchall()

        print("表主键状态检查:")
        print("-" * 60)

        for (table_name,) in sorted(tables):
            has_pk = check_table_pk(con, table_name)
            defined_pk = TABLE_PRIMARY_KEYS.get(table_name)

            if defined_pk:
                status = "✓ 有主键" if has_pk else "✗ 缺少主键（需要修复）"
                pk_info = f"定义: {defined_pk}"
            else:
                status = "- 未定义主键" if not has_pk else "? 有主键但未在代码中定义"
                pk_info = ""

            print(f"  {table_name}: {status}")
            if pk_info and not has_pk:
                print(f"    {pk_info}")

    finally:
        con.close()


if __name__ == "__main__":
    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))

    if len(sys.argv) < 2:
        print("用法: python scripts/fix_table_pk.py [表名|--check]")
        print()
        print("示例:")
        print("  python scripts/fix_table_pk.py fina_indicator_vip  # 修复指定表")
        print("  python scripts/fix_table_pk.py --check             # 检查所有表")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--check":
        check_all_tables(db_path)
    else:
        fix_table_pk(db_path, arg)
