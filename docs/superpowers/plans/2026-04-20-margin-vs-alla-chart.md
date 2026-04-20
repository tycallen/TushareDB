# 两融余额N日累计变化 vs 中证全指 双Y轴日线图 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `margin` 数据表和接口，实现独立脚本绘制两融余额N日累计变化 vs 中证全指(000985.SH)的双Y轴日线图

**Architecture:** 在现有 DataDownloader/DataReader 架构中新增 `margin` 表（市场级两融余额汇总），通过 Tushare `margin` 接口下载。绘图脚本独立运行，读取本地 DuckDB，计算N日累计变化，matplotlib 双Y轴出图。

**Tech Stack:** Python, DuckDB, Tushare, matplotlib, pandas

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tushare_db/duckdb_manager.py` | Modify | `TABLE_PRIMARY_KEYS` 新增 `margin: ['trade_date', 'exchange_id']` |
| `src/tushare_db/downloader.py` | Modify | 新增 `download_margin()` 方法 |
| `src/tushare_db/reader.py` | Modify | 新增 `get_margin()` 方法 |
| `scripts/update_daily.py` | Modify | 新增 `update_margin()` 函数并在并行任务中调用；`update_index_daily()` 中加入 `000985.SH` |
| `scripts/plot_margin_vs_alla.py` | Create | 独立绘图脚本，双Y轴日线图 |
| `docs/skills/tushare-duckdb/SKILL.md` | Modify | 新增 `margin` 表说明和 `get_margin()` 使用示例 |

---

## Task 1: 新增 margin 表主键

**Files:**
- Modify: `src/tushare_db/duckdb_manager.py`

- [ ] **Step 1: 在 `margin_detail` 下方插入 `margin` 主键定义**

  ```python
  # 融资融券
  "margin_detail": ["ts_code", "trade_date"],
  "margin": ["trade_date", "exchange_id"],  # 市场级两融余额汇总
  ```

- [ ] **Step 2: 验证无语法错误**

  Run: `python -c "from src.tushare_db.duckdb_manager import TABLE_PRIMARY_KEYS; print('margin' in TABLE_PRIMARY_KEYS); print(TABLE_PRIMARY_KEYS['margin'])"`
  Expected: `True` and `['trade_date', 'exchange_id']`

- [ ] **Step 3: Commit**

  ```bash
  git add src/tushare_db/duckdb_manager.py
  git commit -m "feat: add margin table primary keys" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 2: 新增 download_margin 方法

**Files:**
- Modify: `src/tushare_db/downloader.py`

- [ ] **Step 1: 在 `download_margin_detail` 方法下方插入 `download_margin` 方法**

  找到 `download_margin_detail` 方法的结束位置（约在 1515 行），在其后添加：

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
        logger.debug(f"下载两融余额汇总: trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}, exchange_id={exchange_id}")

        df = self.fetcher.fetch(
            'margin',
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            exchange_id=exchange_id
        )

        if df.empty:
            logger.debug(f"无两融余额汇总数据")
            return 0

        self.db.write_dataframe(df, 'margin', mode='append')
        logger.info(f"两融余额汇总数据: {len(df)} 行")
        return len(df)
  ```

- [ ] **Step 2: 验证方法可导入**

  Run: `python -c "from tushare_db import DataDownloader; print(hasattr(DataDownloader, 'download_margin'))"`
  Expected: `True`

- [ ] **Step 3: Commit**

  ```bash
  git add src/tushare_db/downloader.py
  git commit -m "feat: add download_margin method for market-level margin balance" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 3: 新增 get_margin 方法

**Files:**
- Modify: `src/tushare_db/reader.py`

- [ ] **Step 1: 在 `get_moneyflow` 方法下方插入 `get_margin` 方法**

  找到 `get_moneyflow` 方法的结束位置（约在 699 行），在其后添加：

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
            - rzye: 融资余额（元）
            - rqye: 融券余额（元）
            - rzrqye: 融资融券余额（元）
            - rzmre: 融资买入额（元）
            - rzche: 融资偿还额（元）
            - rqmcl: 融券卖出量（股）
            - rqyl: 融券余量（股）
            - rqchl: 融券偿还量（股）

        Examples:
            >>> # 获取所有交易所的两融余额
            >>> df = reader.get_margin(start_date='20240101', end_date='20241231')
            >>>
            >>> # 获取上海交易所的两融余额
            >>> df = reader.get_margin(exchange_id='SSE', start_date='20240101')
        """
        conditions = []
        params = []

        if exchange_id:
            conditions.append("exchange_id = ?")
            params.append(exchange_id)

        if start_date and end_date:
            conditions.append("trade_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        query = "SELECT * FROM margin"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY trade_date DESC, exchange_id"

        return self.db.execute_query(query, params if params else None)
  ```

- [ ] **Step 2: 验证方法可导入**

  Run: `python -c "from tushare_db import DataReader; print(hasattr(DataReader, 'get_margin'))"`
  Expected: `True`

- [ ] **Step 3: Commit**

  ```bash
  git add src/tushare_db/reader.py
  git commit -m "feat: add get_margin query method" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 4: 新增 update_margin 并更新每日任务

**Files:**
- Modify: `scripts/update_daily.py`

- [ ] **Step 1: 在 `update_margin_detail` 函数下方插入 `update_margin` 函数**

  找到 `update_margin_detail` 函数的结束位置（约在 1411 行），在其后添加：

  ```python

def update_margin(downloader: DataDownloader):
    """
    增量更新融资融券余额汇总数据 (margin)

    策略：
        1. 获取数据库中 margin 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新两融余额汇总数据 (margin)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('margin', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 融资融券数据开始于 2010年
            start_date = '20100101'
            logger.info("数据库中没有两融余额汇总历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 获取交易日
        if start_date > today:
            logger.info("无需更新")
            return

        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])

        if trading_dates_df.empty:
            logger.info("期间无交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        logger.info(f"需要更新 {len(trading_dates)} 个交易日")

        # 3. 逐日更新
        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_margin(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 两融余额汇总更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新两融余额汇总数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")
  ```

- [ ] **Step 2: 在 `update_index_daily` 的指数列表中加入 `000985.SH`**

  找到 indices 列表（约在 364 行），在列表末尾添加：

  ```python
        ('000985.SH', '中证全指'),
  ```

- [ ] **Step 3: 在并行任务列表中加入 `update_margin`**

  找到 `parallel_tasks` 列表中 `update_margin_detail` 所在行（约在 2648 行），在其后添加：

  ```python
                (update_margin, "两融余额汇总"),
  ```

- [ ] **Step 4: 验证脚本可导入**

  Run: `python -c "import sys; sys.path.insert(0, '.'); from scripts.update_daily import update_margin; print('update_margin imported')"`
  Expected: `update_margin imported`

- [ ] **Step 5: Commit**

  ```bash
  git add scripts/update_daily.py
  git commit -m "feat: add margin balance daily update and CSI All Share index (000985.SH)" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 5: 创建绘图脚本

**Files:**
- Create: `scripts/plot_margin_vs_alla.py`

- [ ] **Step 1: 创建脚本文件**

  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  """
  两融余额N日累计变化 vs 中证全指 双Y轴日线图

  使用方法:
      python scripts/plot_margin_vs_alla.py
      python scripts/plot_margin_vs_alla.py --days 60
      python scripts/plot_margin_vs_alla.py --start-date 20240101 --end-date 20241231
      python scripts/plot_margin_vs_alla.py --output my_chart.png --no-show

  环境变量:
      DB_PATH: 数据库路径（可选，默认为 tushare.db）
  """
  import argparse
  import os
  import sys
  from datetime import datetime, timedelta
  from pathlib import Path

  import matplotlib.dates as mdates
  import matplotlib.pyplot as plt
  import pandas as pd

  # 添加项目根目录到路径
  project_root = Path(__file__).parent.parent
  sys.path.insert(0, str(project_root))

  from tushare_db import DataReader

  # 中证全指代码
  ALL_A_INDEX_CODE = '000985.SH'


  def parse_args():
      parser = argparse.ArgumentParser(
          description='绘制两融余额N日累计变化 vs 中证全指双Y轴日线图'
      )
      parser.add_argument(
          '--days', type=int, default=30,
          help='过去N日（默认30）'
      )
      parser.add_argument(
          '--start-date', type=str, default=None,
          help='起始日期 YYYYMMDD（优先级高于 --days）'
      )
      parser.add_argument(
          '--end-date', type=str, default=None,
          help='结束日期 YYYYMMDD（默认今天）'
      )
      parser.add_argument(
          '--db-path', type=str, default=None,
          help='数据库路径（默认从 DB_PATH 环境变量或 tushare.db）'
      )
      parser.add_argument(
          '--output', type=str, default=None,
          help='输出图片路径（默认 reports/charts/margin_vs_alla_YYYYMMDD.png）'
      )
      parser.add_argument(
          '--no-show', action='store_true',
          help='不显示图表（仅保存）'
      )
      return parser.parse_args()


  def query_margin_data(reader: DataReader, start_date: str, end_date: str) -> pd.DataFrame:
      """查询两融余额汇总数据，返回按日期求和后的每日总余额"""
      if not reader.table_exists('margin'):
          raise RuntimeError(
              "margin 表不存在。请先运行 DataDownloader.download_margin() 下载数据：\n"
              "  from tushare_db import DataDownloader\n"
              "  downloader = DataDownloader()\n"
              "  downloader.download_margin(start_date='20200101', end_date='20241231')\n"
              "  downloader.close()\n"
              "或直接运行: python scripts/update_daily.py"
          )

      df = reader.get_margin(start_date=start_date, end_date=end_date)
      if df.empty:
          raise RuntimeError(f"未找到 {start_date} 至 {end_date} 的两融余额数据")

      # 按日期求和（SSE + SZSE）
      df_sum = df.groupby('trade_date')['rzrqye'].sum().reset_index()
      df_sum.columns = ['trade_date', 'total_margin']
      df_sum = df_sum.sort_values('trade_date').reset_index(drop=True)
      return df_sum


  def query_alla_index_data(reader: DataReader, start_date: str, end_date: str) -> pd.DataFrame:
      """查询中证全指日线数据"""
      if not reader.table_exists('index_daily'):
          raise RuntimeError("index_daily 表不存在。请先下载指数日线数据。")

      df = reader.get_index_daily(ts_code=ALL_A_INDEX_CODE, start_date=start_date, end_date=end_date)
      if df.empty:
          raise RuntimeError(
              f"未找到 {ALL_A_INDEX_CODE} 在 {start_date} 至 {end_date} 的数据。\n"
              f"请先下载：\n"
              f"  from tushare_db import DataDownloader\n"
              f"  downloader = DataDownloader()\n"
              f"  downloader.download_index_daily(ts_code='{ALL_A_INDEX_CODE}', start_date='{start_date}', end_date='{end_date}')\n"
              f"  downloader.close()"
          )

      df = df[['trade_date', 'close']].copy()
      df = df.sort_values('trade_date').reset_index(drop=True)
      return df


  def calculate_n_day_change(df: pd.DataFrame, n: int) -> pd.DataFrame:
      """计算N日累计变化量（亿元）"""
      df = df.copy()
      # 单位从元转为亿元
      df['total_margin_yi'] = df['total_margin'] / 1e8
      # N日累计变化 = 当日 - N日前
      df['margin_change'] = df['total_margin_yi'] - df['total_margin_yi'].shift(n)
      return df


  def plot_chart(df: pd.DataFrame, n: int, output_path: str = None, show: bool = True):
      """绘制双Y轴日线图"""
      # 转换日期格式
      df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

      fig, ax1 = plt.subplots(figsize=(14, 7))

      # 左Y轴：两融余额N日累计变化
      color_margin = '#1f77b4'
      ax1.plot(df['trade_date'], df['margin_change'], color=color_margin, linewidth=1.5, label=f'两融余额{n}日累计变化（亿元）')
      ax1.set_xlabel('日期', fontsize=12)
      ax1.set_ylabel('两融余额累计变化（亿元）', color=color_margin, fontsize=12)
      ax1.tick_params(axis='y', labelcolor=color_margin)

      # 右Y轴：中证全指收盘
      ax2 = ax1.twinx()
      color_index = '#ff7f0e'
      ax2.plot(df['trade_date'], df['close'], color=color_index, linewidth=1.5, label='中证全指收盘')
      ax2.set_ylabel('中证全指收盘点位', color=color_index, fontsize=12)
      ax2.tick_params(axis='y', labelcolor=color_index)

      # 参考横线（左Y轴）
      ax1.axhline(y=2000, color='red', linestyle='--', alpha=0.6, label='红线: 2000亿 / 5500')
      ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.4, label='0轴: 0 / 4500')
      ax1.axhline(y=-1000, color='green', linestyle='--', alpha=0.6, label='绿线: -1000亿 / 4000')

      # 标题
      start_str = df['trade_date'].min().strftime('%Y-%m-%d')
      end_str = df['trade_date'].max().strftime('%Y-%m-%d')
      plt.title(f'两融余额{n}日累计变化 vs 中证全指\n({start_str} ~ {end_str})', fontsize=14)

      # 日期格式
      ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
      ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
      plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

      # 合并图例（放在左上角）
      lines1, labels1 = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

      plt.tight_layout()

      # 保存
      if output_path:
          os.makedirs(os.path.dirname(output_path), exist_ok=True)
          plt.savefig(output_path, dpi=150, bbox_inches='tight')
          print(f"图表已保存: {output_path}")

      if show:
          plt.show()


  def main():
      args = parse_args()

      # 确定日期范围
      if args.end_date is None:
          end_date = datetime.now().strftime('%Y%m%d')
      else:
          end_date = args.end_date

      if args.start_date:
          start_date = args.start_date
          n = None  # 使用自定义日期范围时不计算N日
      else:
          n = args.days
          start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=n * 2)
          # 留有余量用于计算N日变化
          start_date = start_dt.strftime('%Y%m%d')

      # 数据库路径
      db_path = args.db_path or os.getenv('DB_PATH', 'tushare.db')

      print(f"查询范围: {start_date} ~ {end_date}")
      if n:
          print(f"N日参数: {n}")

      # 读取数据
      reader = DataReader(db_path=db_path)
      try:
          margin_df = query_margin_data(reader, start_date, end_date)
          alla_df = query_alla_index_data(reader, start_date, end_date)

          # 合并
          merged = pd.merge(margin_df, alla_df, on='trade_date', how='inner')
          if merged.empty:
              raise RuntimeError("合并后数据为空，请检查日期范围是否有交集")

          # 计算N日变化
          if n:
              merged = calculate_n_day_change(merged, n)
              # 去掉前N日（无法计算完整变化）
              merged = merged.iloc[n:].reset_index(drop=True)
          else:
              merged['margin_change'] = merged['total_margin'] / 1e8

          if merged.empty:
              raise RuntimeError(f"数据不足，无法计算{n}日累计变化")

          print(f"有效数据: {len(merged)} 个交易日")
          print(f"两融变化范围: {merged['margin_change'].min():.0f} ~ {merged['margin_change'].max():.0f} 亿元")
          print(f"中证全指范围: {merged['close'].min():.0f} ~ {merged['close'].max():.0f}")

          # 输出路径
          if args.output:
              output_path = args.output
          else:
              today_str = datetime.now().strftime('%Y%m%d')
              output_path = f"reports/charts/margin_vs_alla_{today_str}.png"

          # 绘图
          plot_chart(merged, n or len(merged), output_path=output_path, show=not args.no_show)

      finally:
          reader.close()


  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 2: 验证脚本语法**

  Run: `python -m py_compile scripts/plot_margin_vs_alla.py`
  Expected: 无输出（无语法错误）

- [ ] **Step 3: Commit**

  ```bash
  git add scripts/plot_margin_vs_alla.py
  git commit -m "feat: add plot_margin_vs_alla.py dual-axis chart script" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 6: 同步 SKILL.md 文档

**Files:**
- Modify: `docs/skills/tushare-duckdb/SKILL.md`

- [ ] **Step 1: 在日频时间序列表中 `margin_detail` 下方插入 `margin` 行**

  找到 `margin_detail` 所在行（约在 199 行），在其后插入：

  ```markdown
  | `margin` | margin | trade_date, exchange_id | **20100331** | 两融余额汇总（市场级别，SSE/SZSE） |
  ```

- [ ] **Step 2: 在 Usage Patterns 的 Download Data 代码块后新增 `get_margin` 示例**

  在 `### Download Data` 部分的代码块后（约在 66 行），添加：

  ```markdown
  ### Query Margin Balance

  ```python
  from tushare_db import DataReader

  reader = DataReader(db_path="tushare.db")

  # 获取所有交易所的两融余额汇总
  df = reader.get_margin(start_date='20240101', end_date='20241231')

  # 获取上海交易所的两融余额
  df = reader.get_margin(exchange_id='SSE', start_date='20240101')

  reader.close()
  ```
  ```

- [ ] **Step 3: 运行验证脚本**

  Run: `python scripts/validate_skill_sync.py`
  Expected: `所有表已在 SKILL.md 中记录 ✓` 或类似成功信息

- [ ] **Step 4: Commit**

  ```bash
  git add docs/skills/tushare-duckdb/SKILL.md
  git commit -m "docs: sync SKILL.md with margin table and get_margin usage" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Task 7: 集成测试

**Files:**
- 无新增文件，验证现有修改

- [ ] **Step 1: 下载 margin 数据**

  Run:
  ```python
  python -c "
  from tushare_db import DataDownloader
  d = DataDownloader()
  rows = d.download_margin(start_date='20250401', end_date='20250420')
  print(f'Downloaded {rows} rows')
  d.close()
  "
  ```
  Expected: `Downloaded N rows`（N > 0）

- [ ] **Step 2: 下载中证全指数据**

  Run:
  ```python
  python -c "
  from tushare_db import DataDownloader
  d = DataDownloader()
  rows = d.download_index_daily(ts_code='000985.SH', start_date='20250401', end_date='20250420')
  print(f'Downloaded {rows} rows')
  d.close()
  "
  ```
  Expected: `Downloaded N rows`（N > 0）

- [ ] **Step 3: 运行绘图脚本**

  Run: `python scripts/plot_margin_vs_alla.py --days 30 --no-show`
  Expected:
  - 输出 `查询范围: ...`
  - 输出 `有效数据: N 个交易日`
  - 输出 `图表已保存: reports/charts/margin_vs_alla_YYYYMMDD.png`
  - 图片文件存在

- [ ] **Step 4: 验证数据可查询**

  Run:
  ```python
  python -c "
  from tushare_db import DataReader
  r = DataReader()
  df = r.get_margin(start_date='20250401', end_date='20250420')
  print(f'Queried {len(df)} rows')
  print(df[['trade_date', 'exchange_id', 'rzrqye']].head())
  r.close()
  "
  ```
  Expected: 有数据行和列输出

- [ ] **Step 5: Final commit**

  ```bash
  git add -A
  git commit -m "test: verify margin data and chart generation" -m "Generated with [Claude Code](https://claude.ai/code)" -m "via [Happy](https://happy.engineering)" -m "Co-Authored-By: Claude <noreply@anthropic.com>" -m "Co-Authored-By: Happy <yesreply@happy.engineering>"
  ```

---

## Self-Review Checklist

- [ ] **Spec coverage:**
  - `margin` 表主键定义 -> Task 1
  - `download_margin()` -> Task 2
  - `get_margin()` -> Task 3
  - `update_daily.py` 集成 -> Task 4
  - `000985.SH` 加入 index_daily -> Task 4
  - 绘图脚本 -> Task 5
  - SKILL.md 同步 -> Task 6
  - 测试计划 -> Task 7

- [ ] **Placeholder scan:** 无 TBD/TODO/"implement later"
- [ ] **Type consistency:**
  - `TABLE_PRIMARY_KEYS['margin']` = `['trade_date', 'exchange_id']`
  - `download_margin(exchange_id=...)` 与 `get_margin(exchange_id=...)` 参数名一致
  - `exchange_id` 值为 `SSE`/`SZSE`
