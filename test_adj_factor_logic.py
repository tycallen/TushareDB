
import os
import pandas as pd
from tushare_db.client import TushareDBClient

# 初始化 TushareDBClient
# 确保 TUSHARE_TOKEN 环境变量已设置
token = os.getenv("TUSHARE_TOKEN")
if not token:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

client = TushareDBClient(tushare_token=token, db_path="test_adj_factor.db")

def test_adj_factor_logic():
    """
    测试复权因子逻辑是否正确。
    1. 获取未复权日线数据。
    2. 获取复权因子。
    3. 本地计算后复权收盘价。
    4. 获取接口计算的后复权日线数据。
    5. 比较两者差异。
    """
    ts_code = '000001.SZ'
    start_date = '20230101'
    end_date = '20231231'

    print("--- 步骤 0: 初始化交易日历数据 ---")
    client.get_data(
        'trade_cal',
        start_date=start_date,
        end_date=end_date
    )
    print("交易日历数据初始化完成。")

    print(f"\n--- 步骤 1: 获取 {ts_code} 在 {start_date} 到 {end_date} 的未复权日线数据 ---")
    df_daily_no_adj = client.get_data(
        'pro_bar',
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj=None,  # 明确指定不复权
        freq='D'
    )
    print(f"获取到 {len(df_daily_no_adj)} 条未复权数据。")
    # print(df_daily_no_adj.head())

    print("\n--- 清理 pro_bar 缓存 ---")
    client.duckdb_manager.execute_query("DROP TABLE IF EXISTS pro_bar;")
    print("缓存清理完成。")

    print(f"\n--- 步骤 2: 获取 {ts_code} 在同一时间段的复权因子 ---")
    df_adj_factor = client.get_data(
        'adj_factor',
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    print(f"获取到 {len(df_adj_factor)} 条复权因子数据。")
    # print(df_adj_factor.head())

    # 合并数据
    df_merged = pd.merge(df_daily_no_adj, df_adj_factor, on=['ts_code', 'trade_date'])
    
    print("\n--- 步骤 3: 本地计算后复权OHLC价格 ---")
    # 根据 Tushare pro_bar(adj='hfq') 接口的实际返回，其计算方式为：不复权价 * 复权因子
    # 这与 adj_factor 文档中描述的后复权公式（需要除以最新因子）不一致，但为了与接口保持一致，我们采用直接相乘
    df_merged['open_hfq_manual'] = df_merged['open'] * df_merged['adj_factor']
    df_merged['high_hfq_manual'] = df_merged['high'] * df_merged['adj_factor']
    df_merged['low_hfq_manual'] = df_merged['low'] * df_merged['adj_factor']
    df_merged['close_hfq_manual'] = df_merged['close'] * df_merged['adj_factor']
    print("本地计算完成。")
    # print(df_merged[['trade_date', 'open', 'high', 'low', 'close', 'adj_factor']].head())

    print(f"\n--- 步骤 4: 直接从接口获取后复权日线数据 ---")
    df_daily_hfq_api = client.get_data(
        'pro_bar',
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj='hfq',  # 直接获取后复权数据
        freq='D'
    )
    print(f"获取到 {len(df_daily_hfq_api)} 条后复权数据。")
    # print(df_daily_hfq_api[['trade_date', 'open', 'high', 'low', 'close']].head())

    print("\n--- 步骤 5: 对比验证 ---")
    # 将接口返回的后复权数据合并到主数据框
    df_final = pd.merge(
        df_merged,
        df_daily_hfq_api[['trade_date', 'open', 'high', 'low', 'close']],
        on='trade_date',
        suffixes=('', '_hfq_api')
    )

    # 计算差异
    # 由于浮点数精度问题，我们检查差异是否在一个很小的范围内
    df_final['open_diff'] = abs(df_final['open_hfq_manual'] - df_final['open_hfq_api'])
    df_final['high_diff'] = abs(df_final['high_hfq_manual'] - df_final['high_hfq_api'])
    df_final['low_diff'] = abs(df_final['low_hfq_manual'] - df_final['low_hfq_api'])
    df_final['close_diff'] = abs(df_final['close_hfq_manual'] - df_final['close_hfq_api'])

    # 设置一个很小的阈值来判断是否相等
    tolerance = 1e-2
    
    # 找出差异大于阈值的记录
    diff_records = df_final[
        (df_final['open_diff'] > tolerance) |
        (df_final['high_diff'] > tolerance) |
        (df_final['low_diff'] > tolerance) |
        (df_final['close_diff'] > tolerance)
    ]

    if diff_records.empty:
        print("\n✅ 验证通过！本地计算的后复权OHLC价格与接口返回的后复权OHLC价格完全一致。")
    else:
        print("\n❌ 验证失败！发现差异：")
        print(diff_records[[
            'trade_date',
            'adj_factor',
            'open_hfq_manual', 'open_hfq_api', 'open_diff',
            'high_hfq_manual', 'high_hfq_api', 'high_diff',
            'low_hfq_manual', 'low_hfq_api', 'low_diff',
            'close_hfq_manual', 'close_hfq_api', 'close_diff'
        ]])

    # 清理测试数据库
    os.remove("test_adj_factor.db")
    print("\n测试数据库 'test_adj_factor.db' 已清理。")


def test_production_data_is_unadjusted():
    """
    测试生产数据库中的日线数据是否是未复权的。
    """
    print("\n--- 开始测试生产数据库中的日线数据 ---")
    prod_client = TushareDBClient(tushare_token=token, db_path="tushare.db")
    
    ts_code = '000001.SZ'
    start_date = '20240101'
    end_date = '20240131'

    # 1. 从生产数据库直接读取数据
    print(f"--- 步骤 1: 从 tushare.db 读取 {ts_code} 数据 ---")
    try:
        if not prod_client.duckdb_manager.table_exists('pro_bar'):
            print("生产数据库中 pro_bar 表不存在，跳过测试。")
            return

        query = f"SELECT trade_date, close FROM pro_bar WHERE ts_code = '{ts_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}' ORDER BY trade_date"
        df_prod = prod_client.duckdb_manager.execute_query(query)
        if df_prod.empty:
            print(f"在 tushare.db 中未找到 {ts_code} 在指定时期的数据，跳过测试。")
            return
        print(f"从 tushare.db 读取到 {len(df_prod)} 条数据。")
    except Exception as e:
        print(f"从生产数据库读取数据失败: {e}")
        return
    finally:
        prod_client.close()

    # 2. 从 Tushare API 获取未复权数据作为基准
    print(f"\n--- 步骤 2: 从 Tushare API 获取 {ts_code} 未复权数据 ---")
    # 使用一个新的 client 来确保我们获取的是纯净的 API 数据
    api_client = TushareDBClient(tushare_token=token, db_path=":memory:")
    df_api_unadjusted = api_client.tushare_fetcher.fetch(
        'pro_bar',
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj=None,
        freq='D'
    )
    api_client.close()
    print(f"从 Tushare API 获取到 {len(df_api_unadjusted)} 条未复权数据。")

    # 3. 对比验证
    print("\n--- 步骤 3: 对比验证 ---")
    df_merged = pd.merge(
        df_prod,
        df_api_unadjusted[['trade_date', 'close']],
        on='trade_date',
        suffixes=('_prod', '_api')
    )

    df_merged['diff'] = abs(df_merged['close_prod'] - df_merged['close_api'])
    
    tolerance = 1e-6
    diff_records = df_merged[df_merged['diff'] > tolerance]

    if diff_records.empty:
        print("\n✅ 验证通过！生产数据库中的日线数据确认为未复权数据。")
    else:
        print("\n❌ 验证失败！生产数据库中的数据与API未复权数据存在差异：")
        print(diff_records)


if __name__ == "__main__":
    test_adj_factor_logic()
    test_production_data_is_unadjusted()
