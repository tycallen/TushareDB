
import os
import sys
import time
import multiprocessing
import random
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tushare_db.reader import DataReader

def query_task(process_id, db_path):
    """模拟查询任务"""
    try:
        start_time = time.time()
        print(f"[Process {process_id}] Started at {datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        # 初始化 DataReader (会自动使用 read_only=True)
        reader = DataReader(db_path=db_path)
        
        # 随机选择一个股票进行查询，模拟不同的负载
        # 这里为了确保有数据，我们固定查几个大盘股
        targets = ['000001.SZ', '600000.SH', '000002.SZ', '600036.SH', '000300.SH']
        target = random.choice(targets)
        
        # 模拟频繁查询
        for i in range(5):
            df = reader.get_stock_daily(ts_code=target, start_date='20100101')
            # 简单计算一下，消耗点CPU
            _ = df['close'].mean()
            # 稍微sleep一下，模拟网络IO或处理时间
            time.sleep(random.uniform(0.1, 0.3))
            
        reader.close()
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"[Process {process_id}] Finished in {duration:.2f}s (Success)")
        return True
    except Exception as e:
        print(f"[Process {process_id}] Failed: {e}")
        return False

def test_concurrent_read():
    db_path = 'tushare.db'  # 确保这里指向正确的数据库路径
    
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    print(f"Starting concurrent read test on {db_path}...")
    print("Mode: Read-Only (Concurrency Expected)")
    
    process_count = 10
    processes = []
    
    start_time = time.time()
    
    # 启动多个进程
    for i in range(process_count):
        p = multiprocessing.Process(target=query_task, args=(i, db_path))
        processes.append(p)
        p.start()
        
    # 等待所有进程结束
    for p in processes:
        p.join()
        
    total_duration = time.time() - start_time
    print(f"\nAll {process_count} processes completed in {total_duration:.2f}s")
    print("If you see no errors above, concurrent read works!")

if __name__ == "__main__":
    test_concurrent_read()
