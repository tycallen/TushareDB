#!/usr/bin/env python3
"""
检查数据库Schema与API文档的一致性
"""
from tushare_db import DataReader
import json

def get_all_tables_and_columns():
    """获取数据库中所有表及其字段"""
    reader = DataReader()

    # 获取所有表名
    tables_df = reader.query("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """)

    schema = {}

    for table_name in tables_df['table_name']:
        # 获取表的字段信息
        columns_df = reader.query(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)

        schema[table_name] = {
            'columns': columns_df['column_name'].tolist(),
            'column_details': columns_df.to_dict('records')
        }

    reader.close()
    return schema

def check_reader_methods():
    """检查DataReader有哪些公开方法"""
    from tushare_db import DataReader
    import inspect

    methods = []
    for name, method in inspect.getmembers(DataReader, predicate=inspect.isfunction):
        if not name.startswith('_'):  # 排除私有方法
            sig = inspect.signature(method)
            methods.append({
                'name': name,
                'params': list(sig.parameters.keys()),
                'doc': method.__doc__.strip() if method.__doc__ else None
            })

    return methods

def check_web_endpoints():
    """检查web_server.py中的所有端点"""
    import re

    with open('/Users/allen/workspace/python/stock/Tushare-DuckDB/src/tushare_db/web_server.py', 'r') as f:
        content = f.read()

    # 查找所有的 @app.get 端点
    endpoints = re.findall(r'@app\.get\("([^"]+)"\)\s+async def (\w+)', content)

    return [{'path': path, 'function': func} for path, func in endpoints]

def main():
    print("=" * 80)
    print("数据库 Schema 与 API 一致性检查")
    print("=" * 80)
    print()

    # 1. 检查数据库表和字段
    print("📊 数据库表结构：")
    print("-" * 80)
    schema = get_all_tables_and_columns()

    for table, info in sorted(schema.items()):
        print(f"\n表名: {table}")
        print(f"  字段数: {len(info['columns'])}")
        print(f"  字段列表: {', '.join(info['columns'][:10])}")
        if len(info['columns']) > 10:
            print(f"           ... 共 {len(info['columns'])} 个字段")

    print("\n" + "=" * 80)

    # 2. 检查DataReader方法
    print("\n🔧 DataReader 公开方法：")
    print("-" * 80)
    methods = check_reader_methods()

    for method in methods:
        params_str = ', '.join(method['params'][1:])  # 排除 self
        print(f"  • {method['name']}({params_str})")
        if method['doc']:
            first_line = method['doc'].split('\n')[0]
            print(f"    → {first_line}")

    print("\n" + "=" * 80)

    # 3. 检查Web API端点
    print("\n🌐 Web API 端点：")
    print("-" * 80)
    endpoints = check_web_endpoints()

    for endpoint in endpoints:
        print(f"  • {endpoint['path']:<30} → {endpoint['function']}")

    print("\n" + "=" * 80)

    # 4. 对比分析
    print("\n🔍 一致性分析：")
    print("-" * 80)

    # 检查文档中提到的表是否存在
    documented_tables = {
        'stock_basic': '股票列表',
        'daily': '日线数据（未复权）',
        'adj_factor': '复权因子',
        'daily_basic': '每日指标',
        'trade_cal': '交易日历',
        'stock_company': '上市公司信息',
        'cyq_perf': '筹码分布',
        'stk_factor_pro': '技术因子',
        'cyq_chips': '筹码成本',
        'dc_member': '板块成分',
        'dc_index': '板块指数',
        'index_basic': '指数基本信息',
        'index_weight': '指数成分权重',
        'hs_const': '沪深港通成分',
        'fina_indicator_vip': '财务指标',
        'moneyflow_ind_dc': '行业资金流向'
    }

    print("\n✅ 文档中的表在数据库中的存在情况：")
    for table, desc in documented_tables.items():
        exists = table in schema
        status = "✓" if exists else "✗"
        print(f"  {status} {table:<25} {desc}")

    print("\n📋 数据库中但文档未提及的表：")
    undocumented = set(schema.keys()) - set(documented_tables.keys())
    for table in sorted(undocumented):
        print(f"  • {table}")

    print("\n" + "=" * 80)

    # 5. 导出详细Schema供参考
    print("\n💾 导出详细Schema到 database_schema.json")
    with open('database_schema.json', 'w', encoding='utf-8') as f:
        json.dump({
            'tables': schema,
            'reader_methods': methods,
            'web_endpoints': [e['path'] for e in endpoints]
        }, f, indent=2, ensure_ascii=False)

    print("✓ 完成！")

    # 6. 重点检查几个常用表的字段
    print("\n" + "=" * 80)
    print("🔎 常用表详细字段检查：")
    print("-" * 80)

    important_tables = ['stock_basic', 'daily', 'daily_basic', 'adj_factor', 'trade_cal']

    for table in important_tables:
        if table in schema:
            print(f"\n{table}:")
            for col_info in schema[table]['column_details'][:15]:  # 只显示前15个字段
                nullable = "NULL" if col_info['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  • {col_info['column_name']:<20} {col_info['data_type']:<15} {nullable}")
            if len(schema[table]['column_details']) > 15:
                print(f"  ... 共 {len(schema[table]['column_details'])} 个字段")

if __name__ == "__main__":
    main()
