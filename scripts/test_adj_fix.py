#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试复权参数修复
验证 qfq 和 hfq 参数是否正确实现
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tushare_db import DataReader

def test_adj_fix():
    """测试复权参数修复"""

    # 初始化 DataReader
    reader = DataReader("tushare.db")

    # 测试股票：002888.SZ（惠威科技）
    ts_code = "002888.SZ"
    test_date = "20251201"

    print("=" * 80)
    print(f"测试股票：{ts_code}")
    print(f"测试日期：{test_date}")
    print("=" * 80)

    # 1. 获取不复权数据
    df_none = reader.get_stock_daily(ts_code, test_date, test_date, adj=None)

    # 2. 获取前复权数据
    df_qfq = reader.get_stock_daily(ts_code, test_date, test_date, adj='qfq')

    # 3. 获取后复权数据
    df_hfq = reader.get_stock_daily(ts_code, test_date, test_date, adj='hfq')

    # 4. 获取复权因子
    df_adj_factor = reader.get_adj_factor(ts_code, test_date, test_date)

    if df_none.empty or df_qfq.empty or df_hfq.empty:
        print(f"❌ 未找到 {ts_code} 的数据")
        print("\n请先下载数据：")
        print(f"  python scripts/update_daily.py --stock {ts_code}")
        return

    # 提取价格
    none_open = df_none['open'].iloc[0]
    qfq_open = df_qfq['open'].iloc[0]
    hfq_open = df_hfq['open'].iloc[0]
    adj_factor = df_adj_factor['adj_factor'].iloc[0] if not df_adj_factor.empty else None

    print("\n【复权因子】")
    print(f"复权因子：{adj_factor}")

    print("\n【价格对比】")
    print(f"不复权开盘价：{none_open:.2f}元")
    print(f"前复权开盘价：{qfq_open:.2f}元")
    print(f"后复权开盘价：{hfq_open:.2f}元")

    # 验证前复权
    print("\n【前复权验证】")
    print("前复权定义：以当前价格为基准，最新日期价格不变")
    if abs(qfq_open - none_open) < 0.01:
        print(f"✅ 前复权正确：{qfq_open:.2f} ≈ {none_open:.2f}")
    else:
        print(f"❌ 前复权错误：{qfq_open:.2f} ≠ {none_open:.2f}")

    # 验证后复权
    print("\n【后复权验证】")
    print("后复权定义：价格 × 复权因子")
    if adj_factor:
        expected_hfq = none_open * adj_factor
        print(f"预期后复权价格：{none_open:.2f} × {adj_factor:.4f} = {expected_hfq:.2f}元")
        if abs(hfq_open - expected_hfq) < 0.01:
            print(f"✅ 后复权正确：{hfq_open:.2f} ≈ {expected_hfq:.2f}")
        else:
            print(f"❌ 后复权错误：{hfq_open:.2f} ≠ {expected_hfq:.2f}")

    # 测试多日数据
    print("\n" + "=" * 80)
    print("测试多日数据（最近5个交易日）")
    print("=" * 80)

    # 获取最近5个交易日
    df_none_5d = reader.get_stock_daily(ts_code, "20251125", test_date, adj=None)
    df_qfq_5d = reader.get_stock_daily(ts_code, "20251125", test_date, adj='qfq')
    df_hfq_5d = reader.get_stock_daily(ts_code, "20251125", test_date, adj='hfq')

    if not df_none_5d.empty:
        print("\n【前复权验证】最新日期价格应该不变：")
        latest_none = df_none_5d['close'].iloc[-1]
        latest_qfq = df_qfq_5d['close'].iloc[-1]
        print(f"不复权最新收盘：{latest_none:.2f}元")
        print(f"前复权最新收盘：{latest_qfq:.2f}元")
        if abs(latest_qfq - latest_none) < 0.01:
            print("✅ 前复权最新价格正确（与不复权相同）")
        else:
            print("❌ 前复权最新价格错误（应该与不复权相同）")

        # 显示历史价格对比
        print("\n【多日价格对比】")
        print(f"{'日期':<12} {'不复权':<10} {'前复权':<10} {'后复权':<10} {'复权因子':<10}")
        print("-" * 60)
        for idx in range(len(df_none_5d)):
            date = df_none_5d['trade_date'].iloc[idx]
            none_price = df_none_5d['close'].iloc[idx]
            qfq_price = df_qfq_5d['close'].iloc[idx]
            hfq_price = df_hfq_5d['close'].iloc[idx]
            factor = df_none_5d['adj_factor'].iloc[idx] if 'adj_factor' in df_none_5d.columns else 'N/A'
            print(f"{date:<12} {none_price:<10.2f} {qfq_price:<10.2f} {hfq_price:<10.2f} {factor if isinstance(factor, str) else f'{factor:.4f}':<10}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    reader.close()

if __name__ == "__main__":
    test_adj_fix()
