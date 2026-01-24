# 板块关系分析使用指南

## 快速开始

### 1. 运行分析

最简单的方式是运行演示脚本：

```bash
python examples/sector_analysis_demo.py
```

这会自动分析 2024 年全年的申万一级行业数据，并生成完整报告。

### 2. 查看结果

分析完成后，结果保存在 `output/YYYY-MM-DD_analysis/` 目录：

```
output/2026-01-24_analysis/
├── analysis.log          # 分析日志
├── report.md             # Markdown 分析报告（主要查看这个）
├── data/                 # 数据文件
│   ├── sector_returns.csv          # 板块涨跌幅明细（7502行）
│   ├── correlation_matrix.csv      # 相关性矩阵（31×31）
│   ├── correlation_detail.csv      # 高相关板块对（439对）
│   └── linkage_strength.csv        # 联动强度详情（870对）
└── plots/                # 图表（如果生成）
```

### 3. 查看报告

**方式一：直接查看 Markdown**

```bash
cat output/2026-01-24_analysis/report.md
```

或用任意 Markdown 阅读器打开（VS Code、Typora、GitHub 等）。

**方式二：分析数据文件**

用 Excel、pandas 或其他工具打开 CSV 文件：

```python
import pandas as pd

# 查看板块涨跌幅
returns = pd.read_csv('output/2026-01-24_analysis/data/sector_returns.csv')
print(returns.head())

# 查看联动强度 Top 10
linkage = pd.read_csv('output/2026-01-24_analysis/data/linkage_strength.csv')
print(linkage.head(10))
```

## 自定义分析

### 编程方式

```python
from tushare_db.sector_analysis import SectorAnalyzer, OutputManager

# 1. 初始化
analyzer = SectorAnalyzer('tushare.db')
output_manager = OutputManager()  # 默认保存到 output/YYYY-MM-DD_analysis/

# 2. 自定义参数
params = {
    'start_date': '20240101',    # 开始日期
    'end_date': '20240630',      # 结束日期（改为半年）
    'level': 'L2',               # 申万二级行业（更细分）
    'period': 'weekly',          # 周线数据（可选: daily/weekly/monthly）
}

# 3. 计算板块涨跌幅
returns_df = analyzer.calculate_sector_returns(
    start_date=params['start_date'],
    end_date=params['end_date'],
    level=params['level'],
    period=params['period']
)

# 4. 相关性分析
corr_matrix = analyzer.calculate_correlation_matrix(
    start_date=params['start_date'],
    end_date=params['end_date'],
    level=params['level'],
    period=params['period']
)

# 查看相关性最强的板块对
corr_detail = analyzer.calculate_correlation_with_pvalue(
    start_date=params['start_date'],
    end_date=params['end_date'],
    level=params['level'],
    period=params['period'],
    min_correlation=0.7  # 只看相关性 > 0.7 的
)
print(corr_detail.head(10))

# 5. 传导关系分析（滞后相关性）
lead_lag_df = analyzer.calculate_lead_lag(
    start_date=params['start_date'],
    end_date=params['end_date'],
    max_lag=None,  # 自适应窗口（周线默认4周）
    level=params['level'],
    period=params['period'],
    min_correlation=0.5
)
print(f"找到 {len(lead_lag_df)} 对传导关系")

# 6. 联动强度分析（Beta 系数）
linkage_df = analyzer.calculate_linkage_strength(
    start_date=params['start_date'],
    end_date=params['end_date'],
    level=params['level'],
    period=params['period'],
    min_r_squared=0.5  # 只看 R² > 0.5 的（解释力度强）
)

# 查看联动最强的板块对
print("\n联动最强的 5 对板块:")
for i, row in linkage_df.head(5).iterrows():
    print(f"{row['sector_a']} -> {row['sector_b']}: "
          f"Beta={row['beta']:.3f}, R²={row['r_squared']:.3f}")
    print(f"  含义: {row['sector_a']} 涨 1% 时，{row['sector_b']} 平均涨 {row['beta']:.3f}%")

# 7. 保存结果
output_manager.save_dataframe(returns_df, 'sector_returns', format='csv')
output_manager.save_dataframe(linkage_df, 'linkage_strength', format='excel')

# 8. 生成报告
results = {
    'returns': returns_df,
    'correlation': corr_matrix,
    'lead_lag': lead_lag_df,
    'linkage': linkage_df,
}
report_path = output_manager.generate_report(results, metadata=params)
print(f"报告已生成: {report_path}")

# 9. 清理
analyzer.close()
```

### 修改演示脚本

复制 `examples/sector_analysis_demo.py` 并修改参数：

```python
# 修改这部分参数
params = {
    'start_date': '20230101',    # 改为 2023 年
    'end_date': '20231231',
    'level': 'L2',               # 改为二级行业
    'period': 'weekly',          # 改为周线
}
```

## 报告说明

### 板块涨跌幅统计

展示各板块的平均涨跌幅排名（Top 10）。

**示例：**
- 801780.SI（银行）平均涨 0.14%
- 801080.SI（电子）平均涨 0.12%

### 相关性分析

展示板块间的同步涨跌关系（-1 到 1）。

**相关系数含义：**
- 0.9 ~ 1.0：高度正相关（同涨同跌）
- 0.7 ~ 0.9：强正相关
- 0.5 ~ 0.7：中度正相关
- -0.5 ~ -0.7：中度负相关（反向）

**示例：**
- 801880.SI ↔ 801890.SI 相关系数 0.969（几乎同步）

### 传导关系分析

展示板块间的时滞带动效应（领涨 vs 跟涨）。

**字段说明：**
- `sector_lead`：领涨板块
- `sector_lag`：跟涨板块
- `lag_days`：滞后天数（日线）或周数（周线）
- `correlation`：滞后相关系数

**示例：**
- A板块今天涨 → B板块3天后涨（lag_days=3）

### 联动强度分析

展示板块间的涨跌联动倍数（Beta 系数）。

**字段说明：**
- `sector_a`：驱动板块
- `sector_b`：联动板块
- `beta`：联动系数
- `r_squared`：解释力度（0~1，越高越可靠）

**Beta 系数含义：**
- Beta = 1.06：A涨1%，B平均涨1.06%（同向放大）
- Beta = 0.89：A涨1%，B平均涨0.89%（同向减弱）
- Beta = -0.5：A涨1%，B平均跌0.5%（反向）

**示例：**
- 801880.SI → 801890.SI: Beta=1.057, R²=0.939
  - 含义：801880 涨 1% 时，801890 平均涨 1.06%
  - 解释力度 93.9%（非常可靠）

## 参数配置

### 层级选择

- `L1`：申万一级行业（31 个，宏观）
- `L2`：申万二级行业（134 个，中观）
- `L3`：申万三级行业（228 个，微观）

### 周期选择

- `daily`：日线数据（滞后窗口 5 天）
- `weekly`：周线数据（滞后窗口 4 周）
- `monthly`：月线数据（滞后窗口 3 月）

### 阈值调整

- `min_correlation`：最小相关系数（建议 0.5~0.7）
- `min_r_squared`：最小 R²（建议 0.3~0.5）
- `max_lag`：最大滞后期（None 表示自适应）

## 常见问题

### Q1: 找不到传导关系（0 对）？

**原因：** 日线滞后 1-5 天窗口太短，市场同步性强。

**解决：**
1. 增加数据时长（至少半年）
2. 降低阈值 `min_correlation=0.2`
3. 使用周线或月线数据

### Q2: 数据量太大，运行很慢？

**优化方案：**
1. 减少时间范围（如只分析 1 个季度）
2. 使用 L1 层级（板块少）
3. 使用周线/月线数据（样本少）

### Q3: 如何理解 Beta 和相关系数的区别？

- **相关系数**：衡量同步性（-1 到 1）
- **Beta 系数**：衡量联动倍数（可以 > 1）
- **R²**：衡量 Beta 的可靠性（0 到 1）

**示例：**
- 相关系数 0.9 = 涨跌方向 90% 一致
- Beta 1.5 = A涨1%，B平均涨1.5%
- R² 0.8 = Beta 能解释 80% 的变化

### Q4: 如何找到最强的板块关系？

```python
# 1. 相关性最强（同步性）
corr_detail = analyzer.calculate_correlation_with_pvalue(...)
top_corr = corr_detail.nlargest(10, 'correlation')

# 2. 联动最强（解释力度）
linkage = analyzer.calculate_linkage_strength(...)
top_linkage = linkage.nlargest(10, 'r_squared')

# 3. 传导性最强（滞后相关）
lead_lag = analyzer.calculate_lead_lag(...)
top_lead = lead_lag.nlargest(10, 'correlation')
```

## 高级用法

### 可视化相关性矩阵

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 计算相关性矩阵
corr_matrix = analyzer.calculate_correlation_matrix(
    start_date='20240101',
    end_date='20241231',
    level='L1',
    period='daily'
)

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='RdYlGn', center=0,
            vmin=-1, vmax=1, square=True)
plt.title('申万一级行业相关性热力图')
plt.tight_layout()

# 保存
output_manager.save_plot(plt.gcf(), 'correlation_heatmap')
plt.show()
```

### 筛选特定板块

```python
# 只分析金融板块
financial_sectors = ['801780.SI', '801790.SI', '801880.SI']  # 银行、非银、保险

# 计算后筛选
corr_detail = analyzer.calculate_correlation_with_pvalue(...)
financial_corr = corr_detail[
    (corr_detail['sector_a'].isin(financial_sectors)) &
    (corr_detail['sector_b'].isin(financial_sectors))
]
```

### 时间序列分析

```python
# 滚动窗口分析相关性变化
import pandas as pd

window_size = 60  # 60 个交易日
results = []

for start in range(0, len(all_dates) - window_size, 20):
    window_dates = all_dates[start:start + window_size]
    corr = analyzer.calculate_correlation_matrix(
        start_date=window_dates[0],
        end_date=window_dates[-1],
        level='L1'
    )
    # 提取特定板块对的相关性
    results.append({
        'date': window_dates[-1],
        'correlation': corr.loc['801780.SI', '801790.SI']
    })

# 绘制相关性变化趋势
df = pd.DataFrame(results)
df.plot(x='date', y='correlation')
```

## 数据说明

### 板块代码

申万行业代码格式：`XXXXXX.SI`

**常见一级行业：**
- 801010.SI：农林牧渔
- 801030.SI：基础化工
- 801080.SI：电子
- 801780.SI：银行
- 801790.SI：非银金融

完整列表参见：`index_classify` 表

### 数据来源

- 板块成分：`index_member_all` 表（含历史变更）
- 个股行情：`daily` 表
- 板块分类：`index_classify` 表

### 无未来函数保证

所有计算均使用历史数据，确保成分股在计算日期前已纳入：

```sql
WHERE in_date <= '{trade_date}'
  AND (out_date IS NULL OR out_date > '{trade_date}')
```

## 技术细节

### 计算方法

**等权平均涨跌幅：**
```
板块涨跌幅 = Σ(成分股涨跌幅) / 成分股数量
```

**周期转换（累计收益率）：**
```
周/月涨跌幅 = (1 + 日涨幅1) × (1 + 日涨幅2) × ... - 1
```

**Beta 系数（线性回归）：**
```
sector_b = α + β × sector_a
```

### 自适应滞后窗口

| 周期 | 滞后窗口 | 说明 |
|------|---------|------|
| daily | 5 天 | 一周交易日 |
| weekly | 4 周 | 一个月 |
| monthly | 3 月 | 一个季度 |

可通过 `max_lag` 参数自定义。
