import os
import warnings
import tushare_db
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm
from tushare_db import TushareDBClient
import logging
from typing import Optional


# 忽略来自 tushare 库内部的特定 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")

# 获取今天的日期
today = datetime.now().strftime('%Y%m%d')
# 获取前30天的日期，用于更新交易日历等
thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

# 初始化 Tushare Pro API
tushare_token = os.getenv("TUSHARE_TOKEN")
if not tushare_token:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

# 数据库路径需要根据您的实际情况调整
client = tushare_db.TushareDBClient(tushare_token=tushare_token, db_path="/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db")

def get_last_update_date(client: TushareDBClient, ts_code: str, table_name: str, date_col: str) -> Optional[str]:
    """
    获取指定股票在特定数据表中的最新更新日期。
    """
    if not client.duckdb_manager.table_exists(table_name):
        logging.warning(f"表 '{table_name}' 不存在。")
        return None

    columns = client.duckdb_manager.get_table_columns(table_name)
    if date_col not in columns:
        logging.warning(f"在表 '{table_name}' 中未找到日期列 '{date_col}'。")
        return None

    query = f"SELECT MAX({date_col}) FROM {table_name} WHERE ts_code = '{ts_code}';"
    try:
        result_df = client.duckdb_manager.execute_query(query)
        if result_df is not None and not result_df.empty and result_df.iloc[0, 0] is not None:
            latest_date = result_df.iloc[0, 0]
            if isinstance(latest_date, (date, datetime)):
                return latest_date.strftime('%Y%m%d')
            return str(latest_date)
        else:
            return None
    except Exception as e:
        logging.error(f"查询 {ts_code} 在表 {table_name} 中的最新日期时出错: {e}")
        return None

def update_trade_cal():
    """每日更新交易日历"""
    print("开始每日更新交易日历...")
    try:
        # 更新最近30天的交易日历，覆盖可能遗漏的日期
        tushare_db.api.trade_cal(client, start_date=thirty_days_ago, end_date=today)
        print("交易日历更新完成。")
    except Exception as e:
        print(f"更新交易日历时出错: {e}")

def update_stock_basic():
    """每日更新股票列表"""
    print("开始每日更新股票列表...")
    try:
        # 刷新所有状态的股票列表，确保获取最新的上市/退市信息
        tushare_db.api.stock_basic(client, list_status='L')
        tushare_db.api.stock_basic(client, list_status='D')
        tushare_db.api.stock_basic(client, list_status='P')
        print("股票列表更新完成。")
    except Exception as e:
        print(f"更新股票列表时出错: {e}")

def update_pro_bar():
    """每日更新所有股票的日线数据"""
    print("开始每日更新所有股票的日线数据...")
    all_stocks = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='P')])

    for _, stock in tqdm(all_stocks.iterrows(), total=len(all_stocks), desc="更新日线数据"):
        ts_code = stock['ts_code']
        # 获取该股票在 daily_bar 表中的最新交易日期
        last_update_date_str = get_last_update_date(client, ts_code, 'pro_bar', 'trade_date')

        start_date_to_fetch = '20000101' # 如果没有数据，从最早日期开始获取
        if last_update_date_str:
            last_date = datetime.strptime(last_update_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')

        # 确保起始日期不晚于今天
        if start_date_to_fetch > today:
            continue # 如果已经是最新的，则跳过

        try:
            tushare_db.api.pro_bar(client=client, ts_code=ts_code, asset='E', freq='D', start_date=start_date_to_fetch, end_date=today)
        except Exception as e:
            print(f"更新 {ts_code} 日线数据时出错: {e}")
    print("所有股票的日线数据更新完成。")

def update_fina_indicator_vip():
    """
    每日更新所有股票的财务指标数据（VIP接口）
    尝试获取最近8个季度的数据，以确保覆盖所有可能的延迟报告。
    """
    print("开始每日更新所有股票的财务指标数据...")
    current_date = datetime.now()
    quarters_to_fetch = []

    # 生成最近8个季度的结束日期
    for i in range(8): # 获取最近8个季度 (2年) 的数据
        year = current_date.year
        month = current_date.month

        # 确定当前季度的结束月份
        if month >= 10: # 第四季度
            quarter_end_month = 12
        elif month >= 7: # 第三季度
            quarter_end_month = 9
        elif month >= 4: # 第二季度
            quarter_end_month = 6
        else: # 第一季度
            quarter_end_month = 3

        # 构造季度结束日期字符串
        period_str = f"{year}{quarter_end_month:02d}31" if quarter_end_month in [3, 12] else f"{year}{quarter_end_month:02d}30"
        quarters_to_fetch.append(period_str)

        # 移动到上一个季度
        if quarter_end_month == 3:
            current_date = datetime(year - 1, 12, 31)
        else:
            current_date = datetime(year, quarter_end_month - 3, 1)

    # 去重并按升序排序
    quarters_to_fetch = sorted(list(set(quarters_to_fetch)))

    for period in quarters_to_fetch:
        # 只获取不晚于今天的季度数据
        if datetime.strptime(period, '%Y%m%d') > datetime.now():
            continue

        print(f"正在获取 {period} 的财务指标数据...")
        try:
            tushare_db.api.fina_indicator_vip(client, period=period)
            print(f"季度 {period} 财务指标数据更新完成。")
        except Exception as e:
            print(f"获取 {period} 财务指标数据时出错: {e}")
    print("财务指标数据更新完成。")

def update_daily_basic():
    """每日更新所有股票的每日基本面指标"""
    print("开始每日更新所有股票的每日基本面指标...")
    all_stocks = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='P')])

    for _, stock in tqdm(all_stocks.iterrows(), total=len(all_stocks), desc="更新每日基本面指标"):
        ts_code = stock['ts_code']
        # 获取该股票在 daily_basic 表中的最新交易日期
        last_update_date_str = get_last_update_date(client, ts_code, 'daily_basic', 'trade_date')

        start_date_to_fetch = '20000101' # 如果没有数据，从最早日期开始获取
        if last_update_date_str:
            last_date = datetime.strptime(last_update_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')

        if start_date_to_fetch > today:
            continue

        try:
            tushare_db.api.daily_basic(client, ts_code=ts_code, start_date=start_date_to_fetch, end_date=today)
        except Exception as e:
            print(f"更新 {ts_code} 每日基本面指标时出错: {e}")
    print("所有股票的每日基本面指标更新完成。")

def update_adj_factor_data():
    """每日更新所有股票的历史复权因子数据"""
    print("开始每日更新所有股票的历史复权因子数据...")
    all_stocks = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='P')])

    for _, stock in tqdm(all_stocks.iterrows(), total=len(all_stocks), desc="更新复权因子"):
        ts_code = stock['ts_code']
        # 获取该股票在 adj_factor 表中的最新交易日期
        last_update_date_str = get_last_update_date(client, ts_code, 'adj_factor', 'trade_date')

        start_date_to_fetch = '20000101' # 如果没有数据，从最早日期开始获取
        if last_update_date_str:
            last_date = datetime.strptime(last_update_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')

        if start_date_to_fetch > today:
            continue

        try:
            tushare_db.api.adj_factor(client=client, ts_code=ts_code, start_date=start_date_to_fetch, end_date=today)
        except Exception as e:
            print(f"更新 {ts_code} 复权因子数据时出错: {e}")
    print("所有股票的历史复权因子数据更新完成。")

def update_cyq_chips():
    """每日更新所有股票的历史筹码分布数据"""
    print("开始每日更新所有股票的历史筹码分布数据...")
    all_stocks = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks = pd.concat([all_stocks, tushare_db.api.stock_basic(client, list_status='P')])

    for _, stock in tqdm(all_stocks.iterrows(), total=len(all_stocks), desc="更新筹码分布"):
        ts_code = stock['ts_code']
        # 获取该股票在 cyq_chips 表中的最新交易日期
        last_update_date_str = get_last_update_date(client, ts_code, 'cyq_chips', 'trade_date')

        start_date_to_fetch = '20000101' # 如果没有数据，从最早日期开始获取
        if last_update_date_str:
            last_date = datetime.strptime(last_update_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')

        if start_date_to_fetch > today:
            continue

        try:
            tushare_db.api.cyq_chips(client=client, ts_code=ts_code, start_date=start_date_to_fetch, end_date=today)
        except Exception as e:
            print(f"更新 {ts_code} 筹码分布数据时出错: {e}")
    print("所有股票的历史筹码分布数据更新完成。")

def main():
    """主函数，执行所有每日更新任务"""
    print("开始每日数据更新...")
    update_trade_cal()
    update_stock_basic()
    update_pro_bar()
    update_adj_factor_data()
    update_cyq_chips()
    update_fina_indicator_vip()
    update_daily_basic()
    print("所有每日数据更新任务完成！")

if __name__ == "__main__":
    main()