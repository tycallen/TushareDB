"""快速测试超额收益传导关系"""
from tushare_db.sector_analysis import SectorAnalyzer

analyzer = SectorAnalyzer('tushare.db')

print("测试1: 计算市场指数收益率")
market_returns = analyzer.calculator.calculate_market_returns(
    '20240101', '20240131', '000300.SH', 'daily'
)
print(f"市场收益率数据: {len(market_returns)} 条")
print(market_returns.head())

print("\n测试2: 原始收益传导关系")
lead_lag_raw = analyzer.calculate_lead_lag(
    '20240101', '20240131', 5, 'L1', 'daily', 0.3
)
print(f"找到 {len(lead_lag_raw)} 对")
if len(lead_lag_raw) > 0:
    print(lead_lag_raw[['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation']].head(5))

print("\n测试3: 超额收益传导关系")
lead_lag_excess = analyzer.calculate_lead_lag_excess(
    '20240101', '20240131', 5, 'L1', 'daily', 0.3, '000300.SH'
)
print(f"找到 {len(lead_lag_excess)} 对")
if len(lead_lag_excess) > 0:
    print(lead_lag_excess[['sector_lead_name', 'sector_lag_name', 'lag_days', 'correlation']].head(5))

analyzer.close()
