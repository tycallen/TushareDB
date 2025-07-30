from csv import Error
import os
import logging
import warnings
from datetime import datetime, timedelta
from tushare_db import TushareDBClient

# 忽略来自 tushare 库内部的特定 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置需要更新的接口 ---
# 在这里添加或修改你希望自动更新的Tushare接口
# 参数说明:
#   - api_name: Tushare的接口名称, 同时也是数据库中的表名
#   - date_col: 用于增量更新的日期列名
#   - update_interval_days: 更新频率（天）。0表示只要有新数据就更新。
#   - fetch_all_stocks: 是否需要获取所有股票的代码列表来进行查询。
#                       对于像 'daily', 'pro_bar' 这样需要传入 ts_code 的接口，设为 True。
#                       对于像 'trade_cal' 这样不需要 ts_code 的接口，设为 False。
APIS_TO_UPDATE = [
    {
        "api_name": "trade_cal",
        "date_col": "cal_date",
        "update_interval_days": 7, # 交易日历不需要每天更新
        "fetch_all_stocks": False,
    },
    # {
    #     "api_name": "daily",
    #     "date_col": "trade_date",
    #     "update_interval_days": 0,
    #     "fetch_all_stocks": True,
    # },
    {
        "api_name": "pro_bar",
        "date_col": "trade_date",
        "update_interval_days": 0,
        "fetch_all_stocks": True,
    },
    {
        "api_name": "cyq_perf",
        "date_col": "trade_date",
        "update_interval_days": 0,
        "fetch_all_stocks": True,
    },
    # 在这里添加更多需要更新的接口...
    # 例如:
    {
        "api_name": "stk_factor_pro",
        "date_col": "trade_date",
        "update_interval_days": 0,
        "fetch_all_stocks": True,
    },
]

def main():
    """主函数，执行每日数据更新"""
    logging.info("开始执行数据更新脚本...")

    try:
        client = TushareDBClient()
        today_str = datetime.now().strftime('%Y%m%d')
        all_stocks = None

        # 检查是否需要预先获取所有股票代码
        if any(api.get("fetch_all_stocks") for api in APIS_TO_UPDATE):
            logging.info("正在获取所有股票代码列表...")
            stock_df = client.get_data("stock_basic", list_status='L')
            if stock_df.empty:
                logging.error("无法获取股票列表���脚本终止。")
                return
            all_stocks = ",".join(stock_df["ts_code"].tolist())
            logging.info(f"成功获取 {len(stock_df)} 只股票代码。")

        for api_config in APIS_TO_UPDATE:
            api_name = api_config["api_name"]
            date_col = api_config["date_col"]
            interval = api_config["update_interval_days"]
            fetch_all = api_config["fetch_all_stocks"]

            logging.info(f"--- 正在检查接口: {api_name} ---")

            latest_date_str = client.get_latest_common_date(api_name, date_col)

            if not latest_date_str:
                logging.warning(f"无法找到接口 '{api_name}' 的本地最新日期，将跳过此接口。请先执行 init_data.py 初始化。")
                continue

            latest_date = datetime.strptime(latest_date_str, '%Y%m%d')
            days_since_last_update = (datetime.now() - latest_date).days

            logging.info(f"本地最新数据日期为: {latest_date_str} ({days_since_last_update}天前)。")

            if days_since_last_update <= interval:
                logging.info(f"数据在设定的更新间隔 ({interval}天) 内，无需更新。")
                continue

            start_date = (latest_date + timedelta(days=1)).strftime('%Y%m%d')

            if start_date > today_str:
                logging.info("本地数据已是最新，无需更新。")
                continue

            logging.info(f"准备更新数据，日期范围: {start_date} -> {today_str}")

            try:
                params = {"start_date": start_date, "end_date": today_str}
                if fetch_all:
                    if not all_stocks:
                        logging.warning(f"接口 '{api_name}' 需要股票代码列表，但未能获取。跳过更新。")
                        continue
                    params["ts_code"] = all_stocks
                
                # 对于 pro_bar，我们通常需要指定更多参数
                if api_name == "pro_bar":
                    params.update({"adj": "qfq", "freq": "D", "asset": "E"})

                client.get_data(api_name, **params)
                logging.info(f"接口 '{api_name}' 数据更新成功！")

            except Error as e:
                logging.error(f"更新接口 '{api_name}' 时发生错误: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()
            logging.info("数据库��接已关闭。")
        logging.info("数据更新脚本执行完毕。")

if __name__ == "__main__":
    main()