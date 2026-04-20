# Design: 两融余额N日累计变化 vs 中证全指 双Y轴日线图

## 目标
绘制每日的（过去N日两融余额累计变化量）与（中证全指收盘价）的双Y轴日线图。
- 横轴：日期
- 左Y轴：两融余额N日累计变化量，单位亿元
- 右Y轴：中证全指收盘点位
- 参考横线：红线（左2000亿/右5500）、绿线（左-1000亿/右4000）、0轴（左0/右4500）

## 背景

### 数据现状
- 数据库中已有 `margin_detail` 表（个股级别融资融券明细），但缺少市场汇总数据
- 已有 `index_daily` 表，但不含 `000985.SH`（中证全指）数据，需下载
- Tushare 提供 `margin` 接口：沪深每日融资融券余额汇总

### 全A指数选择
- 使用 `000985.SH`（中证全指），中证指数公司发布的A股全市场指数，最标准

## 方案

采用方案B：新增 `margin` 数据接口 + 独立绘图脚本。

### 数据层修改

#### 1. `src/tushare_db/duckdb_manager.py`
在 `TABLE_PRIMARY_KEYS` 字典中新增：
```python
'margin': ['trade_date', 'exchange'],
```
字段说明（Tushare `margin` 接口返回）：
- `trade_date`: 交易日期
- `exchange_id`: 交易所代码（SSE/SZSE）
- `rzye`: 融资余额（万元）
- `rqye`: 融券余额（万元）
- `rzrqye`: 融资融券余额（万元）
- `rzmre`: 融资买入额（万元）
- `rzche`: 融资偿还额（万元）
- `rqmcl`: 融券卖出量（股）
- `rqyl`: 融券余量（股）
- `rqchl`: 融券偿还量（股）

#### 2. `src/tushare_db/downloader.py`
新增方法 `download_margin()`：
```python
def download_margin(
    self,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange_id: Optional[str] = None
) -> int:
    """
    下载融资融券交易汇总数据（市场级别）

    数据说明：
    - 获取沪深两市每日融资融券余额汇总
    - 单次最大获取6000条数据
    - 需要至少2000积分

    Args:
        trade_date: 交易日期 YYYYMMDD（可选）
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        exchange_id: 交易所代码 SSE/SZSE（可选）

    Returns:
        下载的行数
    """
```

同时新增 `download_index_daily` 调用支持，确保 `000985.SH` 可下载。

#### 3. `src/tushare_db/reader.py`
新增方法 `get_margin()`：
```python
def get_margin(
    self,
    exchange_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    查询融资融券余额汇总数据

    Args:
        exchange_id: 交易所代码 SSE/SZSE（可选）
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）

    Returns:
        DataFrame，包含字段：
        - trade_date: 交易日期
        - exchange_id: 交易所代码
        - rzye: 融资余额（万元）
        - rqye: 融券余额（万元）
        - rzrqye: 融资融券余额（万元）
        - rzmre: 融资买入额（万元）
        - rzche: 融资偿还额（万元）
        - rqmcl: 融券卖出量（股）
        - rqyl: 融券余量（股）
        - rqchl: 融券偿还量（股）
    """
```

#### 4. `scripts/update_daily.py`
在每日增量更新中加入：
```python
# 下载两融余额汇总
downloader.download_margin(trade_date=trade_date)
```

### 绘图脚本

#### `scripts/plot_margin_vs_alla.py`

**功能**：读取本地 DuckDB，计算两融余额N日累计变化，绘制双Y轴日线图。

**命令行参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--days` | 30 | 过去N日 |
| `--start-date` | None | 起始日期（YYYYMMDD），优先级高于 --days |
| `--end-date` | None | 结束日期（YYYYMMDD），默认今天 |
| `--db-path` | 环境变量或 `tushare.db` | 数据库路径 |
| `--output` | `reports/charts/margin_vs_alla_YYYYMMDD.png` | 输出图片路径 |
| `--no-show` | False | 不显示图表（仅保存） |

**数据查询流程**：
1. 查询 `margin` 表，获取沪深两市数据，按日期求和得每日总余额（`rzrqye`）
2. 查询 `index_daily` 表，`ts_code='000985.SH'`，获取中证全指收盘价
3. 合并两表（按日期左连接）
4. 计算N日累计变化：`margin_change = 当日余额 - N日前余额`，单位从万元转为亿元

**图表配置**：
- matplotlib 双Y轴（`twinx()`）
- 左Y轴范围：根据数据动态，但确保包含 -1000 ~ 2000 区间
- 右Y轴范围：根据全A指数数据动态
- 参考线（使用各自Y轴独立设置，matplotlib `axhline`）：
  - 红线：`axhline(y=2000, color='red', linestyle='--', alpha=0.6)` 对应左Y
  - 绿线：`axhline(y=-1000, color='green', linestyle='--', alpha=0.6)` 对应左Y
  - 0轴：`axhline(y=0, color='gray', linestyle='-', alpha=0.4)` 对应左Y
  - 右侧对应线通过设置右Y轴范围实现：右Y轴范围映射关系为 右 = 左/2 + 4500，即：
    - 左2000 -> 右5500
    - 左0 -> 右4500
    - 左-1000 -> 右4000
- 标题："两融余额{N}日累计变化 vs 中证全指"
- 图例：左上角
- 日期格式：横轴使用 `DateFormatter('%Y-%m')`

**错误处理**：
- `margin` 表不存在：提示用户运行 `download_margin()`
- `000985.SH` 数据不存在：提示用户运行 `download_index_daily(ts_code='000985.SH')`
- 数据不足N日：调整N为实际可用天数并打印警告

### Skill 文档同步

更新 `docs/skills/tushare-duckdb/SKILL.md`：
- 在 Database Schema 表格中新增 `margin` 表说明
- 在 Usage Patterns 中新增 `get_margin()` 示例

验证：`python scripts/validate_skill_sync.py`

## 依赖
- matplotlib（已安装）
- pandas（已安装）
- tushare（已有）

## 测试计划
1. 手动运行 `download_margin()` 下载数据，验证 `margin` 表写入正确
2. 手动运行 `download_index_daily(ts_code='000985.SH')` 下载全A指数数据
3. 运行 `scripts/plot_margin_vs_alla.py --days 30` 验证图表生成
4. 验证 `--start-date` / `--end-date` 参数
5. 验证 `--output` 保存路径
6. 运行 `validate_skill_sync.py` 确保 skill 文档同步
