# 申万板块关系分析系统设计方案

**创建日期**: 2025-01-24
**设计目标**: 基于申万行业分类，构建板块涨跌关系分析系统，实现相关性、传导性、联动强度的多周期综合分析

---

## 1. 需求概述

### 1.1 核心需求
- 分析申万行业板块之间的涨跌关系
- 识别板块间的带动性和传导关系
- 支持多周期分析（日线/周线/月线）
- 支持多权重方法对比（等权/市值加权/官方指数）

### 1.2 数据源
- **申万行业分类**: SW2021版（484个分类，36年历史数据）
- **成分股关系**: `index_member_all` 表（含历史成分变更记录）
- **个股行情**: `daily` 表（完整日线数据）
- **无未来函数**: 通过 `in_date <= trade_date` 筛选确保时点准确性

### 1.3 分析维度
- **相关性分析**: Pearson相关系数矩阵
- **传导关系**: 滞后相关性 + 格兰杰因果检验（可选）
- **联动强度**: Beta系数（板块A涨1%，板块B平均涨多少）
- **多周期**: 日线/周线/月线独立分析

---

## 2. 系统架构

### 2.1 模块结构
```
src/tushare_db/
├── sector_analysis/          # 新增板块分析模块
│   ├── __init__.py
│   ├── calculator.py         # 板块涨跌计算器
│   ├── analyzer.py           # 关系分析器
│   ├── reporter.py           # 报告生成器
│   └── config.py             # 配置管理
```

### 2.2 三层架构

**计算层（Calculator）**
- 输入: 成分股数据、日线行情
- 输出: 板块日/周/月涨跌幅序列
- 方法: 等权平均（第一阶段），后续扩展市值加权

**分析层（Analyzer）**
- 输入: 板块涨跌幅序列
- 输出: 相关性矩阵、传导关系、联动强度
- 算法: Pearson相关、格兰杰因果、滚动相关

**报告层（Reporter）**
- 输入: 分析结果
- 输出: Markdown报告、CSV数据、可视化图表
- 支持: 控制台输出、文件导出

### 2.3 设计原则
- **单一职责**: 每个模块只做一件事
- **可配置化**: 时间范围、窗口参数通过配置文件控制
- **可测试性**: 每层独立测试，易于调试
- **按需计算**: 不缓存中间结果，保持灵活性

---

## 3. 核心接口设计

### 3.1 SectorAnalyzer类
```python
class SectorAnalyzer:
    """板块关系分析器"""

    def __init__(self, db_path: str):
        self.reader = DataReader(db_path)

    def calculate_sector_returns(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',      # L1/L2/L3
        period: str = 'daily',   # daily/weekly/monthly
        method: str = 'equal'    # equal/weighted/index
    ) -> pd.DataFrame:
        """
        计算板块涨跌幅

        Returns:
            DataFrame with columns: [sector_code, sector_name, trade_date, return, stock_count]
        """

    def calculate_correlation_matrix(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily'
    ) -> pd.DataFrame:
        """
        计算相关性矩阵

        Returns:
            相关系数矩阵（行列都是板块代码）
        """

    def calculate_lead_lag(
        self,
        start_date: str,
        end_date: str,
        max_lag: int = None,    # None表示自适应
        level: str = 'L1',
        period: str = 'daily'
    ) -> pd.DataFrame:
        """
        计算传导关系

        Returns:
            DataFrame with columns: [sector_lead, sector_lag, lag_days, correlation, p_value]
        """

    def calculate_linkage_strength(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        计算联动强度

        Returns:
            DataFrame with columns: [sector_a, sector_b, beta, r_squared]
            说明：sector_a涨1%时，sector_b平均涨beta%
        """
```

---

## 4. 核心算法

### 4.1 板块涨跌幅计算

**等权平均方法**:
```python
def _calculate_equal_weighted_return(sector_code, trade_date):
    # 1. 获取成分股（确保无未来函数）
    members = query("""
        SELECT ts_code
        FROM index_member_all
        WHERE l1_code = ?
          AND in_date <= ?
          AND (out_date IS NULL OR out_date > ?)
    """, sector_code, trade_date, trade_date)

    # 2. 获取成分股涨跌幅
    returns = query("""
        SELECT pct_chg
        FROM daily
        WHERE ts_code IN ?
          AND trade_date = ?
    """, members, trade_date)

    # 3. 计算等权平均（剔除停牌/缺失）
    sector_return = returns[returns.notna()].mean()

    return sector_return
```

**周期转换**:
```python
# 日线 → 周线/月线（累计收益率）
period_return = (1 + daily_returns).prod() - 1

# 按自然周/月分组，确保对齐交易日历
weekly_returns = daily_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
```

### 4.2 相关性分析

**Pearson相关系数**:
```python
# 同期相关性
correlation_matrix = sector_returns.pivot(
    index='trade_date',
    columns='sector_code',
    values='return'
).corr(method='pearson')

# 显著性检验
from scipy.stats import pearsonr
corr, p_value = pearsonr(sector_a_returns, sector_b_returns)
```

**滚动窗口相关性**:
```python
# 计算60日滚动相关，观察关系稳定性
rolling_corr = sector_returns.rolling(window=60).corr()
```

### 4.3 传导关系检测

**滞后相关分析**:
```python
def calculate_lead_lag_correlation(sector_a, sector_b, max_lag):
    """计算板块A领先板块B的最优滞后期"""
    results = []
    for lag in range(0, max_lag + 1):
        # sector_a在t日，sector_b在t+lag日
        corr = correlation(
            sector_a_returns[:-lag if lag > 0 else None],
            sector_b_returns[lag:]
        )
        results.append({'lag': lag, 'corr': corr})

    # 找到相关性最强的滞后期
    best_lag = max(results, key=lambda x: abs(x['corr']))
    return best_lag
```

**格兰杰因果检验（可选）**:
```python
from statsmodels.tsa.stattools import grangercausalitytests

# 检验"sector_a是否格兰杰引起sector_b"
# p_value < 0.05 表示显著的因果关系
result = grangercausalitytests(
    data[[sector_b, sector_a]],
    maxlag=max_lag
)
```

**自适应窗口**:
```python
max_lag_map = {
    'daily': 5,      # 1-5日滞后
    'weekly': 4,     # 1-4周滞后（对应5-20日）
    'monthly': 3     # 1-3月滞后（对应20-60日）
}
```

### 4.4 联动强度计算

**Beta系数（线性回归）**:
```python
from scipy.stats import linregress

# sector_b = alpha + beta * sector_a + epsilon
slope, intercept, r_value, p_value, std_err = linregress(
    sector_a_returns,
    sector_b_returns
)

# beta: sector_a涨1%，sector_b平均涨beta%
# r_squared: 解释力度（0-1，越高说明联动越强）
linkage_strength = {
    'beta': slope,
    'r_squared': r_value ** 2,
    'p_value': p_value
}
```

---

## 5. 输出管理

### 5.1 日志分级
```python
import logging

logger = logging.getLogger('sector_analysis')

# 文件：详细调试信息
file_handler.setLevel(logging.DEBUG)

# 控制台：关键进度信息
console_handler.setLevel(logging.INFO)
```

### 5.2 输出统一接口
```python
class OutputManager:
    """统一管理所有输出"""

    def save_dataframe(self, df: pd.DataFrame, name: str, format='csv')
    def save_plot(self, fig, name: str)
    def generate_report(self, results: dict)
```

### 5.3 输出目录结构
```
output/
├── 2025-01-24_analysis/
│   ├── data/
│   │   ├── sector_returns.csv
│   │   ├── correlation_matrix.csv
│   │   └── lead_lag_results.csv
│   ├── plots/
│   │   ├── correlation_heatmap.png
│   │   └── network_graph.png
│   ├── report.md
│   └── analysis.log
```

---

## 6. 可视化设计

### 6.1 相关性热力图
```python
import seaborn as sns

sns.heatmap(
    correlation_matrix,
    annot=True,          # 显示数值
    cmap='RdYlGn',       # 红-黄-绿配色
    center=0,            # 0为中心色
    vmin=-1, vmax=1
)
plt.title('申万一级行业相关性矩阵')
```

### 6.2 传导网络图
```python
import networkx as nx

# 构建有向图：A→B表示A领涨B
G = nx.DiGraph()
for row in lead_lag_results[lead_lag_results['correlation'] > 0.5]:
    G.add_edge(
        row['sector_lead'],
        row['sector_lag'],
        weight=row['correlation'],
        lag=row['lag_days']
    )

# 力导向布局
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
```

### 6.3 Markdown报告模板
```markdown
# 申万行业板块关系分析报告

## 分析周期
- 数据范围：{start_date} ~ {end_date}
- 分析层级：{level}
- 计算方法：{method}

## 核心发现
1. 相关性最强的板块对
2. 最强领涨板块
3. 最强联动关系

## 详细数据
（附表格和图表）
```

---

## 7. 技术依赖

### 7.1 必需依赖
```bash
pip install scipy statsmodels matplotlib seaborn networkx tqdm
```

### 7.2 依赖清单
```
pandas >= 1.5.0        # 数据处理
numpy >= 1.24.0        # 数值计算
scipy >= 1.10.0        # 统计分析
statsmodels >= 0.14.0  # 格兰杰因果检验
matplotlib >= 3.7.0    # 基础绘图
seaborn >= 0.12.0      # 高级可视化
networkx >= 3.0        # 网络图
tqdm >= 4.65.0         # 进度显示
```

---

## 8. 实施计划

### 8.1 第一阶段：核心功能（3-5天）
- [x] 设计评审
- [ ] 创建模块结构
- [ ] 实现Calculator（板块涨跌幅计算）
- [ ] 实现Analyzer（相关性/传导/联动）
- [ ] 实现Reporter（数据导出/报告生成）
- [ ] 编写使用示例脚本

### 8.2 第二阶段：可视化增强（2-3天）
- [ ] 相关性热力图
- [ ] 传导网络图
- [ ] 时间序列对比图
- [ ] 可视化配置优化

### 8.3 后续扩展（按需）
- [ ] 市值加权计算
- [ ] 申万指数数据下载与对比
- [ ] Web交互界面
- [ ] 实时监控功能

---

## 9. 风险与限制

### 9.1 已知限制
- 第一阶段仅实现等权平均方法
- 格兰杰因果检验为可选功能（计算量大）
- 暂不支持实时更新

### 9.2 数据质量依赖
- 依赖 `index_member_all` 的成分数据完整性
- 依赖 `daily` 表的行情数据质量
- 停牌/退市股票会影响板块涨跌计算

### 9.3 性能考虑
- 全量历史计算可能耗时较长
- 建议先用短周期数据验证
- 后续可考虑增量计算优化

---

## 10. 使用示例

```python
from tushare_db.sector_analysis import SectorAnalyzer

# 初始化分析器
analyzer = SectorAnalyzer(db_path='tushare.db')

# 计算2024年申万一级行业的日线涨跌
returns = analyzer.calculate_sector_returns(
    start_date='20240101',
    end_date='20241231',
    level='L1',
    period='daily',
    method='equal'
)

# 计算相关性矩阵
corr_matrix = analyzer.calculate_correlation_matrix(
    start_date='20240101',
    end_date='20241231',
    level='L1',
    period='daily'
)

# 分析传导关系
lead_lag = analyzer.calculate_lead_lag(
    start_date='20240101',
    end_date='20241231',
    level='L1',
    period='daily'
)

# 计算联动强度
linkage = analyzer.calculate_linkage_strength(
    start_date='20240101',
    end_date='20241231',
    level='L1'
)

# 生成报告
from tushare_db.sector_analysis import OutputManager

output = OutputManager(output_dir='output/2025-01-24_analysis')
output.save_dataframe(returns, 'sector_returns')
output.save_dataframe(corr_matrix, 'correlation_matrix')
output.generate_report({
    'returns': returns,
    'correlation': corr_matrix,
    'lead_lag': lead_lag,
    'linkage': linkage
})
```

---

**设计状态**: ✅ 已通过评审
**下一步**: 开始第一阶段实施
