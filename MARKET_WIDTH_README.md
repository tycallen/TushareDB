# 市场宽度图使用说明

## 简介

市场宽度图（Market Breadth）是一种技术分析工具，用于衡量市场中上涨股票的比例，帮助判断市场整体强弱和趋势的可持续性。

本实现基于 Tushare-DuckDB 项目的数据源，使用 DuckDB 进行高效数据查询和分析。

## 功能特点

- ✅ 基于本地 DuckDB 数据库，查询速度快
- ✅ 支持自定义日期范围和交易日数量
- ✅ 自动过滤小市值行业，避免噪音干扰
- ✅ 生成热力图和趋势图，直观展示市场宽度
- ✅ 提供详细的统计分析功能

## 安装依赖

```bash
pip install matplotlib seaborn pandas
```

## 快速开始

### 1. 基本使用

```python
from tushare_db import DataReader
from market_width import get_industry_width, show_industry_width
import datetime

# 初始化数据读取器
reader = DataReader()

# 获取最近100个交易日的市场宽度数据
end_date = datetime.date.today()
df = get_industry_width(reader, end_date, days=100)

# 显示可视化图表（显示最近50天）
show_industry_width(df, count=50)

# 关闭连接
reader.close()
```

### 2. 运行完整示例

```bash
# 运行主程序（会显示图形）
python market_width.py

# 运行测试脚本（不显示图形，只输出数据）
python test_market_width.py

# 运行示例集合
python market_width_example.py
```

## 核心函数说明

### get_industry_width()

获取多个交易日的行业市场宽度数据。

**参数：**
- `reader`: DataReader 实例
- `end_date`: 结束日期，格式 'YYYYMMDD' 或 datetime.date
- `days`: 向前统计的交易日数量
- `min_stocks_per_industry`: 行业至少包含的股票数量，用于过滤小行业（默认20）

**返回：**
- `pd.DataFrame`: 行业宽度数据，行为交易日，列为各行业及总分

**示例：**
```python
from tushare_db import DataReader
from market_width import get_industry_width

reader = DataReader()
df = get_industry_width(
    reader,
    end_date='20251216',  # 结束日期
    days=60,              # 统计60个交易日
    min_stocks_per_industry=30  # 至少30只股票的行业
)
reader.close()

# 查看数据
print(df.head())
print(df['总分'].describe())
```

### show_industry_width()

可视化展示市场宽度热力图和趋势图。

**参数：**
- `df`: 市场宽度数据（由 get_industry_width 返回）
- `count`: 显示的交易日数量（从最新开始），默认显示全部

**示例：**
```python
from market_width import show_industry_width

# 显示最近30天的市场宽度
show_industry_width(df, count=30)
```

## 数据解读

### 市场宽度指标

- **行业上涨占比**：某行业中上涨股票占总股票的百分比（0-100%）
  - `> 50%`：行业整体偏强
  - `< 50%`：行业整体偏弱

- **总分**：所有行业上涨占比的总和
  - 总分越高，市场整体越强
  - 总分持续上升，说明市场趋势健康
  - 总分持续下降，说明市场趋势走弱

### 热力图解读

- **颜色**：
  - 🔴 红色：上涨占比高（市场强）
  - 🔵 蓝色：上涨占比低（市场弱）

- **观察要点**：
  1. **横向**：某一天各行业的表现分化程度
  2. **纵向**：某个行业在不同时间的持续性
  3. **总分列**：市场整体强弱的趋势变化

## 使用示例

### 示例1：分析最近市场表现

```python
from tushare_db import DataReader
from market_width import get_industry_width

reader = DataReader()

# 获取最近20个交易日的数据
df = get_industry_width(reader, '20251216', days=20)

# 找出最强和最弱的交易日
print(f"市场最强交易日: {df['总分'].idxmax()} (总分: {df['总分'].max():.0f})")
print(f"市场最弱交易日: {df['总分'].idxmin()} (总分: {df['总分'].min():.0f})")

# 分析各行业平均表现
industry_avg = df.drop(columns=['总分']).mean().sort_values(ascending=False)
print("\n各行业平均上涨占比:")
for industry, avg in industry_avg.items():
    print(f"  {industry}: {avg:.1f}%")

reader.close()
```

### 示例2：找出强势行业

```python
from tushare_db import DataReader
from market_width import get_industry_width

reader = DataReader()
df = get_industry_width(reader, '20251216', days=60)

# 统计上涨占比 > 50% 的天数
strong_days = (df.drop(columns=['总分']) > 50).sum().sort_values(ascending=False)

print("行业强势天数统计（上涨占比 > 50%）:")
for industry, days_count in strong_days.head(10).items():
    pct = days_count / len(df) * 100
    print(f"  {industry}: {days_count}天 ({pct:.1f}%)")

reader.close()
```

### 示例3：趋势对比分析

```python
from tushare_db import DataReader
from market_width import get_industry_width

reader = DataReader()
df = get_industry_width(reader, '20251216', days=60)

# 对比最近10天 vs 前10天
recent_10 = df.tail(10).drop(columns=['总分']).mean()
previous_10 = df.iloc[-20:-10].drop(columns=['总分']).mean()
trend = recent_10 - previous_10

print("改善最明显的行业:")
for industry, change in trend.sort_values(ascending=False).head(5).items():
    print(f"  {industry}: +{change:.1f}%")

print("\n恶化最明显的行业:")
for industry, change in trend.sort_values(ascending=False).tail(5).items():
    print(f"  {industry}: {change:.1f}%")

reader.close()
```

## 参数调优建议

### min_stocks_per_industry（最小行业股票数）

- **默认值**：20
- **推荐设置**：
  - `10-20`：包含更多细分行业，适合详细分析
  - `30-50`：只关注主要行业，图表更清晰
  - `50+`：只关注大行业，适合快速浏览

### days（统计天数）

- **短期分析**：20-40天，适合捕捉当前市场状态
- **中期分析**：60-100天，适合观察趋势变化
- **长期分析**：120-200天，适合判断市场周期

## 技术原理

1. **数据来源**：
   - `stock_basic` 表：获取股票行业分类
   - `daily` 表：获取日线行情数据（pct_chg 涨跌幅）
   - `trade_cal` 表：获取交易日历

2. **计算方法**：
   - 对每个交易日，统计每个行业中上涨股票（pct_chg > 0）的数量
   - 计算上涨股票占该行业总股票数的百分比
   - 汇总所有行业的百分比得到总分

3. **可视化**：
   - 使用 seaborn 的热力图展示行业宽度矩阵
   - 使用发散颜色映射（红蓝配色）突出强弱对比
   - 单独展示总分列，使用不同的比例尺

## 数据要求

- ✅ 需要初始化 `stock_basic` 表（股票基本信息）
- ✅ 需要初始化 `daily` 表（日线行情数据）
- ✅ 需要初始化 `trade_cal` 表（交易日历）

如果数据库中没有这些数据，请运行：

```bash
# 初始化基础数据
python scripts/init_data.py

# 或单独初始化所需表
python -c "from tushare_db import DataDownloader; dl = DataDownloader(); dl.download_stock_basic()"
```

## 常见问题

### Q: 图表中文显示为方框或乱码？

A: 这是字体配置问题。代码已经自动配置了中文字体，会优先选择 PingFang、Heiti 等系统字体。

**快速测试**：
```bash
python test_chinese_display.py
```

**诊断工具**：
```bash
python check_chinese_font.py
```

**手动指定字体**（如果自动配置不起作用）：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang HK']  # macOS
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux
```

若仍有方块/乱码，确认系统已安装对应中文字体，并清除 matplotlib 字体缓存
（`rm -rf ~/.cache/matplotlib` 后重跑）。

### Q: 为什么有些行业不显示？

A: 默认过滤了股票数量少于20只的行业。可以调整 `min_stocks_per_industry` 参数：

```python
df = get_industry_width(reader, end_date, days=100, min_stocks_per_industry=10)
```

### Q: 如何保存图表？

A: 在调用 `show_industry_width()` 前添加：

```python
import matplotlib.pyplot as plt
plt.savefig('market_width.png', dpi=300, bbox_inches='tight')
```

### Q: 数据更新频率如何？

A: 取决于你的数据更新频率。建议每天收盘后运行日更新脚本：

```bash
python scripts/daily_update.py
```

### Q: 可以分析指数成分股吗？

A: 可以。你需要修改 `get_stock_industry_mapping()` 函数，添加指数成分股过滤：

```python
# 示例：只分析沪深300成分股
index_stocks = reader.query("""
    SELECT DISTINCT con_code as ts_code
    FROM index_weight
    WHERE index_code = '000300.SH'
""")['ts_code'].tolist()

# 在 get_stock_industry_mapping 中添加过滤
WHERE ts_code IN (...)
```

## 参考资料

- [Tushare 数据接口文档](https://tushare.pro/document/2)
- [市场宽度指标原理](https://www.investopedia.com/terms/m/market-breadth.asp)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目遵循 MIT 许可证。
