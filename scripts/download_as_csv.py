import os
import pandas as pd
from datetime import datetime, timedelta
from tushare_db.api import trade_cal
from tushare_db.client import TushareDBClient
import logging
import tushare as ts
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def download_api_data_to_csv(
    client: TushareDBClient,
    api_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "data",
    file_name: Optional[str] = None,
    **api_params: Any
) -> None:
    """
    下载指定Tushare接口的数据，并保存为CSV文件。
    对于有单次请求数据量限制的接口（如 margin_detail），会按日期循环下载。

    Args:
        client: TushareDBClient 实例。
        api_name: Tushare接口名称 (e.g., 'margin_detail', 'daily_basic')。
        start_date: 开始日期 (YYYYMMDD)。对于按日期循环的接口，此参数用于确定循环范围。
        end_date: 结束日期 (YYYYMMDD)。对于按日期循环的接口，此参数用于确定循环范围。
        output_dir: CSV文件保存的目录。
        file_name: CSV文件的名称。如果为 None，则默认为 "{api_name}.csv"。
        **api_params: 传递给Tushare API的其他参数。
    """
    if file_name is None:
        file_name = f"{api_name}.csv"
    output_path = os.path.join(output_dir, file_name)

    logging.info(f"开始下载接口 '{api_name}' 的数据，日期范围从 {start_date} 到 {end_date}，将保存到 {output_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    all_data_frames = []
    pro = ts.pro_api(client.tushare_token)
    # 特殊处理需要按日期循环下载的接口，例如 margin_detail
    if not start_date or not end_date:
        raise ValueError("对于 'margin_detail' 接口，必须提供 start_date 和 end_date。")

    all_trade_dates_df = trade_cal(client, start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        logging.warning(f"在 {start_date} 到 {end_date} 之间没有找到交易日。")
        return

    trade_dates = all_trade_dates_df['cal_date'].tolist()
    trade_dates.sort() # 确保日期是按顺序的

    for date_str in trade_dates:
        logging.info(f"正在下载 {api_name} 在 {date_str} 的数据...")
        try:
            # 将 trade_date 和其他 api_params 传递给 get_data
            df = pro.query(api_name, trade_date=date_str, **api_params)
            if not df.empty:
                all_data_frames.append(df)
                logging.info(f"成功下载 {api_name} 在 {date_str} 的 {len(df)} 条数据。")
            else:
                logging.info(f"{api_name} 在 {date_str} 没有数据。")
        except Exception as e:
            logging.error(f"下载 {api_name} 在 {date_str} 的数据失败: {e}")
  

    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"所有 '{api_name}' 数据已成功保存到 {output_path}，共 {len(final_df)} 条记录。")
    else:
        logging.warning(f"没有下载到任何 '{api_name}' 数据。")

if __name__ == '__main__':
    # 请替换为您的Tushare token
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN") 
    if not TUSHARE_TOKEN:
        logging.error("请设置环境变量 TUSHARE_TOKEN 或在代码中提供您的 Tushare token。")
    else:
        client = TushareDBClient(tushare_token=TUSHARE_TOKEN)

        # 示例1: 下载 margin_detail 数据
        logging.info("\n--- 示例1: 下载 margin_detail 数据 ---")
        # download_api_data_to_csv(
        #     client,
        #     api_name="margin_detail",
        #     start_date="20230101",
        #     end_date="20230105",
        #     file_name="margin_detail_20230101_20230105.csv",
        #     ts_code="000001.SZ" # 可以添加其他 margin_detail 接口参数
        # )

        # 示例2: 下载 daily_basic 数据 (假设 daily_basic 可以直接通过 start_date 和 end_date 获取)
        logging.info("\n--- 示例2: 下载 daily_basic 数据 ---")
        download_api_data_to_csv(
            client,
            api_name="margin",
            start_date="20200101",
            end_date="20230105",
            file_name="margin_total_20200101_20251017.csv"
        )

        client.close()