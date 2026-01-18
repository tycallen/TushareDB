import os
import sys
import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tushare_db.reader import DataReader

load_dotenv()

def analyze_stock(stock_name):
    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')
    # 确保路径存在，如果环境变量没设好，默认可能是 tushare.db
    if not os.path.exists(db_path):
        db_path = 'tushare.db'
        
    reader = DataReader(db_path=db_path)
    
    # 调试：查看数据库中的表
    print("数据库中的表:", reader.db.execute_query("SHOW TABLES").values.flatten().tolist())
    
    try:
        # 1. 查找股票代码
        print(f"正在查找 {stock_name} 的代码...")
        # 直接查表比较麻烦，reader可能有接口，或者直接用execute_query
        df_stock = reader.db.execute_query(
            "SELECT ts_code, name, list_date FROM stock_basic WHERE name = ?", 
            [stock_name]
        )
        
        if df_stock.empty:
            print(f"未找到股票：{stock_name}")
            return
            
        ts_code = df_stock.iloc[0]['ts_code']
        list_date = df_stock.iloc[0]['list_date']
        print(f"找到股票: {ts_code} - {stock_name}, 上市日期: {list_date}")
        
        # 2. 获取日线数据
        print(f"正在获取历史数据...")
        # 不需要复权，因为我们要比较的是当天的开盘价和昨收价，这是原始价格行为
        # 不过复权后的涨跌幅和原始的一样。为了准确比较 open 和 pre_close，用不复权数据比较直观。
        df = reader.get_stock_daily(ts_code=ts_code, start_date='20100101')
        
        if df.empty:
            print("未获取到交易数据")
            return
            
        total_days = len(df)
        print(f"总交易天数: {total_days}")
        
        # 3. 筛选高开的数据
        # 高开：Open > Pre_Close
        # 也可以定义幅度，比如 高开 > 0.5%
        
        # 简单的 Open > Pre_Close
        high_open_df = df[df['open'] > df['pre_close']].copy()
        high_open_count = len(high_open_df)
        high_open_rate = high_open_count / total_days * 100
        
        print(f"\n【高开统计 (Open > Pre_Close)】")
        print(f"高开天数: {high_open_count}")
        print(f"高开概率: {high_open_rate:.2f}%")
        
        if high_open_count == 0:
            return

        # 4. 分析高开后的走势
        
        # 情况A：高开高走（收盘 > 开盘） -> 也就是日内是涨的实心阳线（或假阴真阳，不，Close>Open就是红K线）
        up_after_open = high_open_df[high_open_df['close'] > high_open_df['open']]
        prob_up_after_open = len(up_after_open) / high_open_count * 100
        
        # 情况B：高开低走（收盘 < 开盘） -> 绿K线
        down_after_open = high_open_df[high_open_df['close'] < high_open_df['open']]
        prob_down_after_open = len(down_after_open) / high_open_count * 100
        
        # 情况C：最终收涨（收盘 > 昨收） -> 缺口没完全回补或者回补后又拉起
        close_up = high_open_df[high_open_df['close'] > high_open_df['pre_close']]
        prob_close_up = len(close_up) / high_open_count * 100
        
        # 情况D：最终收跌（收盘 < 昨收） -> 高开低走且跌破昨收
        close_down = high_open_df[high_open_df['close'] < high_open_df['pre_close']]
        prob_close_down = len(close_down) / high_open_count * 100
        
        print(f"\n【高开后当日走势概率】")
        print(f"1. 收盘 > 开盘 (高开高走/收阳线): {prob_up_after_open:.2f}%")
        print(f"2. 收盘 < 开盘 (高开低走/收阴线): {prob_down_after_open:.2f}%")
        print(f"   (注：剩余 {100 - prob_up_after_open - prob_down_after_open:.2f}% 为收盘价等于开盘价)")
        print("-" * 30)
        print(f"3. 最终收涨 (Close > Pre_Close): {prob_close_up:.2f}%")
        print(f"4. 最终收跌 (Close < Pre_Close): {prob_close_down:.2f}%")
        
        # 5. 进阶统计：平均表现
        high_open_df['intraday_chg'] = (high_open_df['close'] - high_open_df['open']) / high_open_df['open'] * 100
        avg_intraday_chg = high_open_df['intraday_chg'].mean()
        
        print(f"\n【收益率统计】")
        print(f"高开后，买入(Open)持有至收盘(Close)的平均收益率: {avg_intraday_chg:.3f}%")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        reader.close()

if __name__ == "__main__":
    analyze_stock("冰川网络")
