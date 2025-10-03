import duckdb
import argparse

def drop_table(table_name: str):
    """删除指定的表"""
    try:
        conn = duckdb.connect('/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db')
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        print(f"成功删除表 {table_name}")
    except Exception as e:
        print(f"删除表 {table_name} 时发生错误: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='删除DuckDB中的指定表')
    parser.add_argument('table_name', type=str, help='要删除的表名')
    
    args = parser.parse_args()
    drop_table(args.table_name)
