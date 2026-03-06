# 报告管理系统使用指南

## 概述

报告管理系统自动保存每次因子验证的结果到本地 SQLite 数据库，支持历史查询、对比分析和统计报表。

---

## 快速开始

### 1. 基础用法（自动保存）

```python
from src.tushare_db.factor_validation import EnhancedFactorFilter

# 创建增强版过滤器（自动保存结果）
filter_obj = EnhancedFactorFilter(
    db_path="tushare.db",
    auto_save=True  # 默认开启自动保存
)

# 运行验证（结果自动保存到数据库）
report = filter_obj.filter(
    factor="macd_golden_cross",
    ts_codes=['000001.SZ'],
    notes="测试笔记"  # 可选备注
)

print(report.markdown)
```

### 2. 查询历史记录

```python
# 查询特定因子的历史
history = filter_obj.get_factor_history("macd_golden_cross")
print(history[['timestamp', 'alpha_ratio', 'recommendation']])

# 查询特定股票的所有测试
stock_history = filter_obj.get_stock_history('000001.SZ')
```

### 3. 统计摘要

```python
# 获取最近30天的统计
stats = filter_obj.get_statistics(days=30)

print(f"总记录数: {stats['total_records']}")
print(f"建议保留: {stats['keep_count']}")
print(f"平均Alpha: {stats['avg_alpha_ratio']:.3f}")
```

---

## 高级功能

### 使用 ReportManager 直接操作

```python
from src.tushare_db.factor_validation.report_manager import ReportManager

# 创建管理器
manager = ReportManager()

# 手动保存报告
record_id = manager.save_report(
    report=report,
    ts_code='000001.SZ',
    stock_name='平安银行',
    notes='手动保存'
)

# 复杂查询
df = manager.query_records(
    factor_name='macd_golden_cross',
    recommendation='KEEP',
    limit=100
)

# 时间序列对比
df_time = manager.compare_factor_over_time(
    factor_name='macd_golden_cross',
    ts_code='000001.SZ'
)
```

### 导出数据

```python
# 导出到 CSV
filter_obj.export_reports("my_reports.csv", days=30)

# 生成统计报告
report = filter_obj.generate_report(days=30)
with open("report.md", "w") as f:
    f.write(report)
```

---

## 数据结构

### ValidationRecord 字段

| 字段 | 说明 | 示例 |
|------|------|------|
| timestamp | 测试时间 | 2024-03-06T10:00:00 |
| factor_name | 因子名称 | macd_golden_cross |
| ts_code | 股票代码 | 000001.SZ |
| p_actual | 真实触发概率 | 0.035 (3.5%) |
| p_random | 随机触发概率 | 0.039 (3.9%) |
| alpha_ratio | Alpha比率 | 0.91 |
| recommendation | 建议 | KEEP / DISCARD |
| is_significant | 是否显著 | True / False |
| notes | 备注 | 测试笔记 |

---

## 典型使用场景

### 场景1：追踪因子表现变化

```python
# 每周测试一次 MACD 金叉，观察 Alpha 变化
for week in range(4):
    report = filter_obj.filter(
        factor="macd_golden_cross",
        ts_codes=['000001.SZ'],
        notes=f"第{week+1}周测试"
    )
    time.sleep(1)  # 间隔一周

# 查看趋势
history = filter_obj.get_factor_history("macd_golden_cross")
print(history[['timestamp', 'alpha_ratio']])
```

### 场景2：批量因子筛选

```python
# 测试所有内置因子
factors = FactorRegistry.list_builtin()

for factor_name in factors:
    report = filter_obj.filter(
        factor=factor_name,
        ts_codes=['000001.SZ'],
        notes="批量筛选"
    )

# 查看排名
stats = filter_obj.get_statistics(days=7)
print(f"最佳因子: {stats['best_factor']}")
print(f"最佳Alpha: {stats['best_alpha']}")
```

### 场景3：多股票对比

```python
stocks = ['000001.SZ', '000002.SZ', '600000.SH']

for stock in stocks:
    report = filter_obj.filter(
        factor="rsi_oversold",
        ts_codes=[stock]
    )

# 对比不同股票的表现
for stock in stocks:
    history = filter_obj.get_stock_history(stock)
    avg_alpha = history['alpha_ratio'].mean()
    print(f"{stock}: 平均Alpha={avg_alpha:.3f}")
```

---

## 数据库位置

默认数据库路径：`~/.factor_validation/reports.db`

```bash
# 查看数据库大小
ls -lh ~/.factor_validation/reports.db

# 使用 SQLite 命令行查询
sqlite3 ~/.factor_validation/reports.db "SELECT * FROM validation_records LIMIT 10;"
```

---

## 演示脚本

```bash
# 报告管理系统演示
python demo_report_manager_simple.py

# 完整工作流程演示
python demo_complete_workflow.py

# 批量对比演示
python demo_full_comparison.py

# 因子性能分析
python analyze_factor_performance.py
```

---

## API 参考

### EnhancedFactorFilter

| 方法 | 说明 |
|------|------|
| `filter(..., save_report=True, notes="")` | 运行验证并自动保存 |
| `get_factor_history(factor_name)` | 获取因子历史 |
| `get_stock_history(ts_code)` | 获取股票历史 |
| `compare_factor_over_time(...)` | 时间序列对比 |
| `get_statistics(days)` | 统计摘要 |
| `export_reports(filepath, days)` | 导出数据 |
| `generate_report(days)` | 生成报告 |

### ReportManager

| 方法 | 说明 |
|------|------|
| `save_report(report, ts_code, ...)` | 保存报告 |
| `query_records(...)` | 查询记录 |
| `get_factor_history(factor_name)` | 因子历史 |
| `compare_factor_over_time(...)` | 时间对比 |
| `get_statistics_summary(days)` | 统计摘要 |
| `export_to_csv(filepath)` | 导出CSV |
| `export_to_excel(filepath)` | 导出Excel |

---

## 注意事项

1. **存储空间**：每条记录约 1KB，1000 条约 1MB
2. **自动清理**：使用 `manager.delete_old_records(days=90)` 清理旧数据
3. **备份**：定期备份 `~/.factor_validation/reports.db`
4. **并发**：SQLite 不支持多进程并发写入

---

## 示例输出

```
# 统计摘要示例

总记录数: 150
测试因子数: 15
测试股票数: 10

质检结果:
  建议保留: 45 (30.0%)
  建议废弃: 105 (70.0%)

Alpha Ratio 统计:
  平均值: 1.234
  中位数: 1.056
  最佳因子: kdj_golden_cross (Alpha=2.450)
  最差因子: close_gt_open (Alpha=0.650)
```
