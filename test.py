from tushare_db import TushareDBClient, stock_basic, StockBasic, trade_cal, TradeCal, hs_const, HsConst, stock_company, StockCompany, pro_bar, ProBar
from tushare_db.client import TushareDBClientError
import pandas as pd

def test_init():
    try:
        client = TushareDBClient()
        
        # base init 
        # client.initialize_basic_data()

        # download stock basic data
        client.get_all_stock_qfq_daily_bar(start_date="20000101", end_date="20250628")

    except TushareDBClientError as e:
        print(f"Successfully caught expected error: {e}")
        
def pivot_tushare_data(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """
    将Tushare的长格式数据转换为vectorbt所需的宽格式。

    Args:
        df (pd.DataFrame): 包含 'trade_date' 和 'ts_code' 的长格式DataFrame。
        value_column (str): 需要作为值的列名 (例如 'close', 'open')。

    Returns:
        pd.DataFrame: 转换后的宽格式DataFrame。
    """
    pivot_df = df.pivot(index='trade_date', columns='ts_code', values=value_column)
    return pivot_df

def test_vectorbt():
    import vectorbt as vbt
    import pandas as pd
    import numpy as np

    TS_CODE = '000002.SH'
    START_DATE = '20220101'
    END_DATE = '20250628'
    client = TushareDBClient()
    # 使用 pro_bar 接口获取数据
    df = pro_bar(
        client,
        # asset='E', # E: 指数, I: 指数
        start_date=START_DATE,
        end_date=END_DATE,
        freq='D' # D: 日线
    )
    # return
    # 数据预处理，使其符合 vectorbt 的格式要求
    # 1. 将 'trade_date' 转换为 DatetimeIndex
    # 2. 按日期升序排序
    # 3. 重命名列名以匹配 vectorbt 的标准 (首字母大写)
    # 将 trade_date 字符串转换为 datetime 对象
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 按日期和代码排序，这是一个好习惯
    long_df = df.sort_values(by=['trade_date', 'ts_code'])

    # vectorbt 推荐使用首字母大写的列名
    long_df = long_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'vol': 'Volume'
    })

    print(f"数据获取成功，共 {len(long_df)} 条记录。")
    print("数据预览:")
    print(long_df.head())

    close_df = pivot_tushare_data(long_df, 'Close')
    open_df = pivot_tushare_data(long_df, 'Open')
    high_df = pivot_tushare_data(long_df, 'High')
    low_df = pivot_tushare_data(long_df, 'Low')
    volume_df = pivot_tushare_data(long_df, 'Volume')

    # ==============================================================================
    # 步骤 3: 定义策略参数并生成信号
    # ==============================================================================
    # 定义短期和长期移动均线的窗口期
    FAST_WINDOW = 10
    SLOW_WINDOW = 30

    print(f"\n策略参数: 短期均线 = {FAST_WINDOW} 天, 长期均线 = {SLOW_WINDOW} 天")

    # 使用 vectorbt 的 vbt.MA.run() 一次性计算两条均线
    # vbt.MA.run() 会返回一个包含两条均线数据的 DataFrame
    price = close_df
    fast_ma = vbt.MA.run(price, 10)
    slow_ma = vbt.MA.run(price, 50)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    print("交易信号已生成。")


    # ==============================================================================
    # 步骤 4: 执行向量化回测
    # ==============================================================================
    # 使用 vbt.Portfolio.from_signals() 函数执行回测
    # - price: 价格数据，用于计算盈亏
    # - entries: 买入信号
    # - exits: 卖出信号
    # - freq: 数据频率，用于计算年化指标
    # - init_cash: 初始资金
    # - fees: 手续费 (例如 0.1% = 0.001)
    # - slippage: 滑点 (例如 0.1% = 0.001)
    print("\n正在执行回测...")

    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        freq='D',          # 数据频率为'Daily'
        init_cash=100000,  # 初始资金为 100,000
        fees=0.001,        # 设置0.1%的交易手续费
        slippage=0.001     # 设置0.1%的交易滑点
    )

    # ==============================================================================
    # 步骤 5: 分析和可视化回测结果
    # ==============================================================================
    print("\n回测完成，生成统计结果...")

    # 打印详细的统计数据
    # .stats() 方法提供了非常全面的性能指标
    print("\n--- 回测性能统计 ---")
    print(pf.stats())

    # 可视化结果
    # .plot() 会生成一个包含多个子图的交互式图表
    print("\n正在生成可视化图表... (图表将在您的浏览器或IDE中打开)")
    fig = pf.plot(title=f'双均线策略回测 - {TS_CODE}')
    fig.show()

    # 您还可以单独查看交易记录
    print("\n--- 交易记录 ---")
    print(pf.trades.records_readable)    

if __name__ == "__main__":
    test_vectorbt()
    # test_init()
