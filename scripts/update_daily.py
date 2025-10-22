import os
import warnings
import tushare_db
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm
from tushare_db import TushareDBClient
import logging
from typing import Optional, Dict, List

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

def get_all_stocks_last_update_dates(client: TushareDBClient, table_name: str, date_col: str) -> Dict[str, Optional[str]]:
    """
    获取所有股票在特定数据表中的最新更新日期。
    """
    if not client.duckdb_manager.table_exists(table_name):
        logging.warning(f"表 '{table_name}' 不存在。")
        return {}

    columns = client.duckdb_manager.get_table_columns(table_name)
    if date_col not in columns:
        logging.warning(f"在表 '{table_name}' 中未找到日期列 '{date_col}'。")
        return {}

    query = f"SELECT ts_code, MAX({date_col}) AS last_date FROM {table_name} GROUP BY ts_code;"
    try:
        result_df = client.duckdb_manager.execute_query(query)
        if result_df is not None and not result_df.empty:
            # 将日期格式化为 YYYYMMDD 字符串
            result_df['last_date'] = result_df['last_date'].apply(
                lambda x: x.strftime('%Y%m%d') if isinstance(x, (date, datetime)) else (str(x) if x is not None else None)
            )
            return result_df.set_index('ts_code')['last_date'].to_dict()
        else:
            return {}
    except Exception as e:
        logging.error(f"获取所有股票在表 {table_name} 中的最新日期时出错: {e}")
        return {}

def get_common_latest_date(last_update_dates: Dict[str, Optional[str]]) -> Optional[str]:
    """
    从所有股票的最新更新日期中，找出出现次数最多的日期。
    """
    if not last_update_dates:
        return None
    
    # 过滤掉 None 值
    valid_dates = [d for d in last_update_dates.values() if d is not None]
    if not valid_dates:
        return None

    from collections import Counter
    date_counts = Counter(valid_dates)
    return date_counts.most_common(1)[0][0]

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

def _batch_update_stocks(
    api_func,
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    desc: str,
    **kwargs
):
    """
    通用批量更新股票数据的辅助函数。
    """
    if not ts_codes:
        return

    batch_size = 100  # 每次请求的股票数量，可以根据Tushare的限制调整
    num_batches = (len(ts_codes) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=desc):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(ts_codes))
        current_batch_codes = ts_codes[batch_start:batch_end]
        
        try:
            api_func(
                client=client,
                ts_code=",".join(current_batch_codes),
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
        except Exception as e:
            logging.error(f"批量更新 {desc} 时出错 (ts_codes: {current_batch_codes[0]}...): {e}")


def update_pro_bar():
    """每日更新所有股票的日线数据"""
    print("开始每日更新所有股票的日线数据...")
    all_stocks_df = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='P')])
    all_ts_codes = all_stocks_df['ts_code'].tolist()

    last_update_dates = get_all_stocks_last_update_dates(client, 'pro_bar', 'trade_date')
    common_latest_date = get_common_latest_date(last_update_dates)

    latest_group_ts_codes = []
    lagging_group_ts_codes = []
    earliest_lagging_start_date = today # 初始化为今天，确保如果所有股票都滞后，也能从最早日期开始

    for ts_code in all_ts_codes:
        last_date_str = last_update_dates.get(ts_code)
        if last_date_str is None: # 新股票或表中无数据
            lagging_group_ts_codes.append(ts_code)
            earliest_lagging_start_date = min(earliest_lagging_start_date, '20000101') # 从最早日期开始
        elif last_date_str == common_latest_date:
            latest_group_ts_codes.append(ts_code)
        else: # 滞后股票
            lagging_group_ts_codes.append(ts_code)
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            earliest_lagging_start_date = min(earliest_lagging_start_date, start_date_to_fetch)

    # 1. 更新最新组的股票
    if latest_group_ts_codes and common_latest_date:
        start_date_for_latest = (datetime.strptime(common_latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        if start_date_for_latest <= today:
            print(f"批量更新最新组日线数据 (从 {start_date_for_latest} 到 {today})...")
            _batch_update_stocks(
                tushare_db.api.pro_bar,
                latest_group_ts_codes,
                start_date_for_latest,
                today,
                "批量更新最新组日线数据",
                asset='E', freq='D'
            )

    # 2. 更新滞后组的股票
    if lagging_group_ts_codes and earliest_lagging_start_date <= today:
        print(f"批量更新滞后组日线数据 (从 {earliest_lagging_start_date} 到 {today})...")
        _batch_update_stocks(
            tushare_db.api.pro_bar,
            lagging_group_ts_codes,
            earliest_lagging_start_date,
            today,
            "批量更新滞后组日线数据",
            asset='E', freq='D'
        )
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
    all_stocks_df = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='P')])
    all_ts_codes = all_stocks_df['ts_code'].tolist()

    last_update_dates = get_all_stocks_last_update_dates(client, 'daily_basic', 'trade_date')
    common_latest_date = get_common_latest_date(last_update_dates)

    latest_group_ts_codes = []
    lagging_group_ts_codes = []
    earliest_lagging_start_date = today

    for ts_code in all_ts_codes:
        last_date_str = last_update_dates.get(ts_code)
        if last_date_str is None:
            lagging_group_ts_codes.append(ts_code)
            earliest_lagging_start_date = min(earliest_lagging_start_date, '20000101')
        elif last_date_str == common_latest_date:
            latest_group_ts_codes.append(ts_code)
        else:
            lagging_group_ts_codes.append(ts_code)
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            earliest_lagging_start_date = min(earliest_lagging_start_date, start_date_to_fetch)

    if latest_group_ts_codes and common_latest_date:
        start_date_for_latest = (datetime.strptime(common_latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        if start_date_for_latest <= today:
            print(f"批量更新最新组每日基本面指标 (从 {start_date_for_latest} 到 {today})...")
            _batch_update_stocks(
                tushare_db.api.daily_basic,
                latest_group_ts_codes,
                start_date_for_latest,
                today,
                "批量更新最新组每日基本面指标"
            )

    if lagging_group_ts_codes and earliest_lagging_start_date <= today:
        print(f"批量更新滞后组每日基本面指标 (从 {earliest_lagging_start_date} 到 {today})...")
        _batch_update_stocks(
            tushare_db.api.daily_basic,
            lagging_group_ts_codes,
            earliest_lagging_start_date,
            today,
            "批量更新滞后组每日基本面指标"
        )
    print("所有股票的每日基本面指标更新完成。")

def update_adj_factor_data():
    """每日更新所有股票的历史复权因子数据"""
    print("开始每日更新所有股票的历史复权因子数据...")
    all_stocks_df = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='P')])
    all_ts_codes = all_stocks_df['ts_code'].tolist()

    last_update_dates = get_all_stocks_last_update_dates(client, 'adj_factor', 'trade_date')
    common_latest_date = get_common_latest_date(last_update_dates)

    latest_group_ts_codes = []
    lagging_group_ts_codes = []
    earliest_lagging_start_date = today

    for ts_code in all_ts_codes:
        last_date_str = last_update_dates.get(ts_code)
        if last_date_str is None:
            lagging_group_ts_codes.append(ts_code)
            earliest_lagging_start_date = min(earliest_lagging_start_date, '20000101')
        elif last_date_str == common_latest_date:
            latest_group_ts_codes.append(ts_code)
        else:
            lagging_group_ts_codes.append(ts_code)
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            earliest_lagging_start_date = min(earliest_lagging_start_date, start_date_to_fetch)

    if latest_group_ts_codes and common_latest_date:
        start_date_for_latest = (datetime.strptime(common_latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        if start_date_for_latest <= today:
            print(f"批量更新最新组复权因子 (从 {start_date_for_latest} 到 {today})...")
            _batch_update_stocks(
                tushare_db.api.adj_factor,
                latest_group_ts_codes,
                start_date_for_latest,
                today,
                "批量更新最新组复权因子"
            )

    if lagging_group_ts_codes and earliest_lagging_start_date <= today:
        print(f"批量更新滞后组复权因子 (从 {earliest_lagging_start_date} 到 {today})...")
        _batch_update_stocks(
                tushare_db.api.adj_factor,
                lagging_group_ts_codes,
                earliest_lagging_start_date,
                today,
                "批量更新滞后组复权因子"
            )
    print("所有股票的历史复权因子数据更新完成。")

def update_cyq_chips():
    """每日更新所有股票的历史筹码分布数据"""
    print("开始每日更新所有股票的历史筹码分布数据...")
    all_stocks_df = tushare_db.api.stock_basic(client, list_status='L')
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='D')])
    all_stocks_df = pd.concat([all_stocks_df, tushare_db.api.stock_basic(client, list_status='P')])
    all_ts_codes = all_stocks_df['ts_code'].tolist()

    last_update_dates = get_all_stocks_last_update_dates(client, 'cyq_chips', 'trade_date')
    common_latest_date = get_common_latest_date(last_update_dates)

    latest_group_ts_codes = []
    lagging_group_ts_codes = []
    earliest_lagging_start_date = today

    for ts_code in all_ts_codes:
        last_date_str = last_update_dates.get(ts_code)
        if last_date_str is None:
            lagging_group_ts_codes.append(ts_code)
            earliest_lagging_start_date = min(earliest_lagging_start_date, '20000101')
        elif last_date_str == common_latest_date:
            latest_group_ts_codes.append(ts_code)
        else:
            lagging_group_ts_codes.append(ts_code)
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date_to_fetch = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            earliest_lagging_start_date = min(earliest_lagging_start_date, start_date_to_fetch)

    if latest_group_ts_codes and common_latest_date:
        start_date_for_latest = (datetime.strptime(common_latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        if start_date_for_latest <= today:
            print(f"批量更新最新组筹码分布 (从 {start_date_for_latest} 到 {today})...")
            _batch_update_stocks(
                tushare_db.api.cyq_chips,
                latest_group_ts_codes,
                start_date_for_latest,
                today,
                "批量更新最新组筹码分布"
            )

    if lagging_group_ts_codes and earliest_lagging_start_date <= today:
        print(f"批量更新滞后组筹码分布 (从 {earliest_lagging_start_date} 到 {today})...")
        _batch_update_stocks(
            tushare_db.api.cyq_chips,
            lagging_group_ts_codes,
            earliest_lagging_start_date,
            today,
            "批量更新滞后组筹码分布"
        )
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