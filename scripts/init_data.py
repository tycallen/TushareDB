
import os
import warnings
import tushare_db
from datetime import datetime, timedelta

# 忽略来自 tushare 库内部的特定 FutureWarning
# Tushare 库使用了即将被废弃的 pandas 功能，此代码可以在不修改库源码的情况下抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")


# 初始化 Tushare Pro API
# 请确保您已经设置了 TUSHARE_TOKEN 环境变量
# 例如: export TUSHARE_TOKEN='your_token_here'
tushare_token = os.getenv("TUSHARE_TOKEN")
if not tushare_token:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

client = tushare_db.TushareDBClient(tushare_token=tushare_token, db_path="/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db")

def init_trade_cal():
    """初始化交易日历"""
    print("开始初始化交易日历...")
    tushare_db.api.trade_cal(client, start_date='19900101', end_date='20301231')

    print("交易日历初始化完成。")

def init_stock_basic():
    """初始化股票列表"""
    print("开始初始化股票列表...")
    tushare_db.api.stock_basic(list_status='L')
    tushare_db.api.stock_basic(list_status='D')
    tushare_db.api.stock_basic(list_status='P')
    print("股票列表初始化完成。")

def init_pro_bar():
    """初始化所有股票的历史日线数据"""
    print("开始初始化所有股票的历史日线数据...")
    # 获取所有股票代码
    all_stocks = tushare_db.api.stock_basic(list_status='L')
    for ts_code in all_stocks['ts_code']:
        print(f"正在获取 {ts_code} 的历史数据...")
        try:
            tushare_db.api.pro_bar(ts_code=ts_code, asset='E', freq='D', start_date='19900101')
        except Exception as e:
            print(f"获取 {ts_code} 数据时出错: {e}")
    print("所有股票的历史日线数据初始化完成。")


def init_fina_indicator_vip():
    """
    初始化所有股票的财务指标数据（VIP接口）
    从2000年至今，按季度获取。
    """
    print("开始初始化所有股票的财务指标数据...")
    current_year = datetime.now().year
    for year in range(2000, current_year + 1):
        for quarter in ['0331', '0630', '0930', '1231']:
            period = f"{year}{quarter}"
            # 如果计算出的报告期在未来，则跳过
            if datetime.strptime(period, '%Y%m%d') > datetime.now():
                continue

            print(f"正在获取 {period} 的财务指标数据...")
            try:
                client.api.fina_indicator_vip(period=period)
            except Exception as e:
                print(f"获取 {period} 财务指标数据时出错: {e}")
    print("财务指标数据初始化完成。")


def init_index_basic():
    """初始化所有指数的基本信息"""
    print("开始初始化所有指数的基本信息...")
    markets = ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH']
    for market in markets:
        print(f"正在获取 {market} 的指数基本信息...")
        try:
            d = tushare_db.api.index_basic(client, market=market)
            print(d.head())
        except Exception as e:
            print(f"获取 {market} 指数基本信息时出错: {e}")
    print("所有指数的基本信息初始化完成。")


def init_index_weight():
    """初始化主要指数的权重"""
    print("开始初始化主要指数的权重...")
    today = datetime.today()
    # 常见指数列表
    common_indices = [
        '000001.SH',  # 上证指数
        '000300.SH',  # 沪深300
        '000016.SH',  # 上证50
        '399001.SZ',  # 深圳成指
        '000905.SH',  # 中证500
        '399006.SZ',  # 创业板指
        '000852.SH',  # 中证1000
        '399303.SZ',  # 国证2000
        '399191.SZ',  # 中小板综合
        '000688.SH',  # 科创50
        # '399102.SZ', # 创业板综合
    ]

    for index_code in common_indices:
        print(f"正在获取指数 {index_code} 的权重...")
        # 获取过去240个月的数据
        for i in range(240):
            target_date = today - timedelta(days=i * 30)
            year = target_date.year
            month = target_date.month
            print(f"  - 获取 {year}年{month}月 的数据...")
            try:
                d = tushare_db.api.index_weight(client, index_code=index_code, year=year, month=month)
                print(d.head())
            except Exception as e:
                print(f"获取 {index_code} 在 {year}-{month} 的权重数据时出错: {e}")
    print("主要指数的权重初始化完成。")


def main():
    """主函数，执行所有初始化任务"""
    print("开始数据初始化...")
    # init_trade_cal()
    # init_index_basic()
    init_index_weight()
    # init_stock_basic()
    # init_pro_bar()
    # init_fina_indicator_vip()
    # client.get_all_stock_qfq_daily_bar(start_date='20000101', end_date=datetime.now().strftime('%Y%m%d'))
    print("所有数据初始化任务完成！")

if __name__ == "__main__":
    main()
