
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from tushare_db import DataReader

def check_strategy_feasibility():
    """
    Verify if Tushare-DuckDB has enough data to support the strategy:
    1. 连续三天上涨，但总涨幅不超过5%，获利筹码大于62%
    2. 连续2天换手率排名前80%，获利筹码大于70%
    3. 排除ST，60天内至少3次涨停，获利筹码大于80%
    """
    print("\n=== Strategy Feasibility Verification ===\n")
    
    reader = DataReader()
    
    try:
        # 1. Check Table Existence
        required_tables = ['daily', 'cyq_perf', 'stock_basic', 'daily_basic']
        print("[Check 1] Verifying required tables...")
        for table in required_tables:
            if reader.table_exists(table):
                print(f"  - {table}: OK")
            else:
                print(f"  - {table}: MISSING! (Strategy cannot run without this)")
                return

        # 2. Get latest available date
        print("\n[Check 2] Getting latest available data date...")
        latest_date_df = reader.query("SELECT MAX(trade_date) as max_date FROM daily")
        if latest_date_df.empty or latest_date_df.iloc[0]['max_date'] is None:
            print("  No data in daily. Please download data first.")
            return
        
        target_date = latest_date_df.iloc[0]['max_date']
        print(f"  Target Date for verification: {target_date}")
        
        # 3. Strategy Implementation Demo
        print("\n[Check 3] Running Strategy Logic on Target Date...")
        
        # --- Condition 1 ---
        # 1. 连续三天上涨 (Three days rising)
        # 2. 总涨幅不超过5% (Total pct_chg <= 5%)
        # 3. 获利筹码大于62% (Winner rate > 62%)
        
        print("\n  --> Testing Condition 1: 3-day rise, sum(pct_chg)<=5, winner_rate>62")
        
        # We need T, T-1, T-2. Let's find previous 2 trading days.
        dates_df = reader.query(
            f"SELECT cal_date FROM trade_cal WHERE cal_date <= '{target_date}' AND is_open=1 ORDER BY cal_date DESC LIMIT 3"
        )
        if len(dates_df) < 3:
            print("  Not enough trading days history.")
            return
            
        date_list = dates_df['cal_date'].tolist()
        t0, t1, t2 = date_list[0], date_list[1], date_list[2] # t0 is target_date (latest)
        print(f"      Checking dates: {t2} -> {t1} -> {t0}")
        
        sql_cond1 = f"""
        WITH price_data AS (
            SELECT ts_code, trade_date, pct_chg 
            FROM daily 
            WHERE trade_date IN ('{t0}', '{t1}', '{t2}')
        ),
        chips_data AS (
            SELECT ts_code, winner_rate 
            FROM cyq_perf 
            WHERE trade_date = '{t0}'
        ),
        streak_check AS (
            SELECT 
                ts_code,
                COUNT(*) as days_count,
                SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) as rise_days,
                SUM(pct_chg) as total_rise
            FROM price_data
            GROUP BY ts_code
            HAVING days_count = 3
        )
        SELECT 
            s.ts_code, s.total_rise, c.winner_rate
        FROM streak_check s
        JOIN chips_data c ON s.ts_code = c.ts_code
        WHERE s.rise_days = 3          -- 3 days rising
          AND s.total_rise <= 5        -- Total rise <= 5%
          AND c.winner_rate > 62       -- Winner rate > 62%
        LIMIT 5
        """
        res1 = reader.query(sql_cond1)
        print(f"      Matches found: {len(res1)} (Showing top 5)")
        if not res1.empty:
            print(res1)
        
        # --- Condition 2 ---
        # 1. 连续2天换手率排名前80% (2 days turnover rank top 80%)
        # 2. 获利筹码大于70% (Winner rate > 70%)
        
        print("\n  --> Testing Condition 2: 2-day turnover rank top 80%, winner_rate>70")
        # Note: "Rank top 80%" usually means exclude the bottom 20% (smallest turnover).
        # We can implement this by calculating the 20th percentile of turnover_rate for each day.
        
        # Calculate threshold for t0 and t1
        t0_threshold = reader.query(f"SELECT percentile_cont(0.2) WITHIN GROUP (ORDER BY turnover_rate) as p20 FROM daily WHERE trade_date='{t0}'").iloc[0]['p20']
        t1_threshold = reader.query(f"SELECT percentile_cont(0.2) WITHIN GROUP (ORDER BY turnover_rate) as p20 FROM daily WHERE trade_date='{t1}'").iloc[0]['p20']
        
        # If turnover_rate is None/NaN, treat as 0.
        import numpy as np
        if pd.isna(t0_threshold) or np.isnan(t0_threshold):
            t0_threshold = 0.0
        if pd.isna(t1_threshold) or np.isnan(t1_threshold):
            t1_threshold = 0.0
        
        print(f"      Turnover Thresholds (Top 80% means > P20): {t0}={t0_threshold:.2f}, {t1}={t1_threshold:.2f}")
        
        sql_cond2 = f"""
        WITH turnover_check AS (
            SELECT ts_code
            FROM daily
            WHERE (trade_date = '{t0}' AND turnover_rate >= {t0_threshold})
               OR (trade_date = '{t1}' AND turnover_rate >= {t1_threshold})
            GROUP BY ts_code
            HAVING COUNT(*) = 2 -- Must meet criteria on both days
        ),
        chips_data AS (
            SELECT ts_code, winner_rate 
            FROM cyq_perf 
            WHERE trade_date = '{t0}'
        )
        SELECT t.ts_code, c.winner_rate
        FROM turnover_check t
        JOIN chips_data c ON t.ts_code = c.ts_code
        WHERE c.winner_rate > 70
        LIMIT 5
        """
        res2 = reader.query(sql_cond2)
        print(f"      Matches found: {len(res2)} (Showing top 5)")
        if not res2.empty:
            print(res2)


        # --- Condition 3 ---
        # 1. 排除ST (Exclude ST)
        # 2. 60天内至少3次涨停 (At least 3 limit ups in 60 days)
        # 3. 获利筹码大于80% (Winner rate > 80%)
        
        print("\n  --> Testing Condition 3: Exclude ST, 3 limit-ups in 60d, winner_rate>80")
        
        # Get start date for 60 day window
        start_date_60d = reader.query(
            f"SELECT cal_date FROM trade_cal WHERE cal_date <= '{target_date}' AND is_open=1 ORDER BY cal_date DESC OFFSET 60 LIMIT 1"
        )
        if start_date_60d.empty:
            print("  Not enough history for 60 days.")
        else:
            d_start = start_date_60d.iloc[0]['cal_date']
            print(f"      Evaluating window: {d_start} to {target_date}")
            
            # Simple Limit Up check: pct_chg > 9.5 (Approximation)
            sql_cond3 = f"""
            WITH limit_up_counts AS (
                SELECT ts_code, SUM(CASE WHEN pct_chg > 9.5 THEN 1 ELSE 0 END) as lu_count
                FROM daily
                WHERE trade_date BETWEEN '{d_start}' AND '{target_date}'
                GROUP BY ts_code
                HAVING lu_count >= 3
            ),
            st_check AS (
                SELECT ts_code, name
                FROM stock_basic
                WHERE name NOT LIKE '%ST%'
            ),
            chips_data AS (
                SELECT ts_code, winner_rate 
                FROM cyq_perf 
                WHERE trade_date = '{target_date}'
            )
            SELECT l.ts_code, s.name, l.lu_count, c.winner_rate
            FROM limit_up_counts l
            JOIN st_check s ON l.ts_code = s.ts_code
            JOIN chips_data c ON l.ts_code = c.ts_code
            WHERE c.winner_rate > 80
            LIMIT 5
            """
            res3 = reader.query(sql_cond3)
            print(f"      Matches found: {len(res3)} (Showing top 5)")
            if not res3.empty:
                print(res3)
                
    except Exception as e:
        print(f"\n[ERROR] An error occurred during verification: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.close()
        print("\n=== Verification Complete ===")

if __name__ == "__main__":
    check_strategy_feasibility()
