
import os
import warnings
import tushare_db
from datetime import datetime, timedelta

# 忽略来自 tushare 库内部的特定 FutureWarning
# Tushare 库使用了即将被废弃的 pandas 功能，此代码可以在不修改库源码的情况下抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")


# 初始化 Tushare Pro API
# 请确保您已经设置了 TUSHARE_TOKEN 环境变量
tushare_token = os.getenv("TUSHARE_TOKEN")
if not tushare_token:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

pro = tushare_db.api.pro_api(tushare_token=tushare_token)

def get_latest_trade_date():
    """获取最近的交易日"""
    today = datetime.now()
    # Tushare 交易日历查询范围有限，我们从今天往前找最近的一个交易日
    for i in range(30): # 最多回看30天
        date_to_check = today - timedelta(days=i)
        date_str = date_to_check.strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='', start_date=date_str, end_date=date_str)
        if not cal.empty and cal.iloc[0]['is_open'] == 1:
            print(f"找到最近的交易日: {date_str}")
            return date_str
    raise ValueError("在过去30天内未找到交易日。")

def update_daily_data(trade_date: str):
    """获取指定日期的所有股票的日线数据"""
    print(f"开始更新 {trade_date} 的日线数据...")
    try:
        pro.daily(trade_date=trade_date)
        print(f"{trade_date} 的日线数据更新完成。")
    except Exception as e:
        print(f"更新 {trade_date} 日线数据时出错: {e}")

def main():
    """主函数，执行每日数据更新"""
    print("开始每日数据更新...")
    try:
        latest_trade_date = get_latest_trade_date()
        update_daily_data(latest_trade_date)
        print("每日数据更新任务完成！")
    except Exception as e:
        print(f"每日更新失败: {e}")

if __name__ == "__main__":
    main()
