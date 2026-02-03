# DuckDB 回测引擎架构设计文档

## 1. 概述

本文档描述了一个基于 DuckDB 数据源的量化回测引擎架构设计。该引擎支持事件驱动和向量化两种回测模式，提供完整的数据管理、交易模拟、策略接口和绩效分析功能。

### 1.1 数据源概览

数据库路径: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`

| 表名 | 说明 | 记录数 | 关键字段 |
|------|------|--------|----------|
| `daily` | 日线行情 | 16,486,059 | ts_code, trade_date, open/high/low/close, vol, amount |
| `adj_factor` | 复权因子 | - | ts_code, trade_date, adj_factor |
| `stk_factor_pro` | 因子数据(含复权价) | - | 250+ 技术指标字段，含 qfq/hfq 价格 |
| `daily_basic` | 每日指标 | - | pe, pb, ps, turnover_rate, total_mv, circ_mv |
| `stock_basic` | 股票基本信息 | 5,795 | ts_code, name, industry, list_date, list_status |
| `trade_cal` | 交易日历 | - | cal_date, is_open, pretrade_date |
| `index_daily` | 指数行情 | - | 上证指数等基准数据 |
| `moneyflow` | 资金流向 | - | 主力资金流入流出 |

数据时间范围: **2000-01-04 至 2026-01-30** (6,321 个交易日)

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Backtest Engine                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  DataLoader │────▶│ EventEngine │────▶│  Strategy   │                   │
│  └─────────────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                    │                   │                          │
│         │                    ▼                   ▼                          │
│         │           ┌─────────────┐     ┌─────────────┐                    │
│         │           │OrderManager │◀───▶│PositionMgr │                    │
│         │           └──────┬──────┘     └──────┬──────┘                    │
│         │                  │                   │                           │
│         │                  └────────┬──────────┘                           │
│         │                           ▼                                      │
│         │                  ┌─────────────────┐                             │
│         └─────────────────▶│PerformanceAnalyzer│                           │
│                            └─────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 核心模块职责

| 模块 | 职责 | 关键方法 |
|------|------|----------|
| **DataLoader** | 数据加载与预处理 | `load_stocks()`, `get_adj_prices()`, `get_calendar()` |
| **EventEngine** | 事件分发与回测循环 | `run()`, `dispatch_event()`, `register_handler()` |
| **Strategy** | 策略逻辑实现 | `on_bar()`, `on_order_filled()`, `generate_signals()` |
| **OrderManager** | 订单生命周期管理 | `submit_order()`, `cancel_order()`, `match_order()` |
| **PositionManager** | 持仓与资金管理 | `update_position()`, `get_holdings()`, `get_nav()` |
| **PerformanceAnalyzer** | 绩效统计与归因 | `calculate_returns()`, `risk_metrics()`, `attribution()` |

---

## 3. 数据管理模块 (DataLoader)

### 3.1 类设计

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import date
import duckdb
import pandas as pd
import numpy as np

@dataclass
class MarketData:
    """单个时间点的市场数据快照"""
    ts_code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    adj_factor: float = 1.0

    # 可选指标
    turnover_rate: Optional[float] = None
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    total_mv: Optional[float] = None


@dataclass
class BarData:
    """K线数据容器"""
    datetime: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    open_interest: float = 0.0

    # 复权价格
    adj_close: Optional[float] = None


class DataLoader:
    """
    高效的DuckDB数据加载器

    特性:
    - 延迟加载：按需加载数据
    - 批量查询：减少数据库往返
    - 内存缓存：避免重复查询
    - 复权处理：支持前复权/后复权
    """

    def __init__(self, db_path: str, cache_size_mb: int = 1024):
        """
        初始化数据加载器

        Args:
            db_path: DuckDB数据库路径
            cache_size_mb: 内存缓存大小(MB)
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

        # 配置DuckDB性能参数
        self.conn.execute(f"SET memory_limit='{cache_size_mb}MB'")
        self.conn.execute("SET threads TO 4")

        # 数据缓存
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._adj_factor_cache: Dict[str, pd.DataFrame] = {}
        self._calendar_cache: Optional[pd.DatetimeIndex] = None
        self._stock_info_cache: Optional[pd.DataFrame] = None

    def get_trading_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = 'SSE'
    ) -> pd.DatetimeIndex:
        """
        获取交易日历

        Args:
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            exchange: 交易所代码 (SSE/SZSE)

        Returns:
            交易日的DatetimeIndex
        """
        query = """
        SELECT cal_date
        FROM trade_cal
        WHERE exchange = ?
          AND is_open = 1
          AND cal_date BETWEEN ? AND ?
        ORDER BY cal_date
        """
        df = self.conn.execute(query, [exchange, start_date, end_date]).fetchdf()
        return pd.to_datetime(df['cal_date'])

    def load_stock_data(
        self,
        ts_codes: Union[str, List[str]],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        adj: str = 'qfq'  # qfq: 前复权, hfq: 后复权, None: 不复权
    ) -> pd.DataFrame:
        """
        加载股票数据

        Args:
            ts_codes: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
            adj: 复权方式

        Returns:
            DataFrame with MultiIndex (trade_date, ts_code)
        """
        if isinstance(ts_codes, str):
            ts_codes = [ts_codes]

        # 默认字段
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'vol', 'amount']

        # 使用 stk_factor_pro 表获取复权数据
        if adj in ('qfq', 'hfq'):
            price_suffix = f'_{adj}'
            adj_fields = [f'{f}{price_suffix}' if f in ('open', 'high', 'low', 'close') else f
                         for f in fields]
            select_clause = ', '.join([f'd.{f}' for f in adj_fields])

            query = f"""
            SELECT
                d.ts_code,
                d.trade_date,
                {select_clause},
                d.adj_factor
            FROM stk_factor_pro d
            WHERE d.ts_code IN ({','.join(['?']*len(ts_codes))})
              AND d.trade_date BETWEEN ? AND ?
            ORDER BY d.ts_code, d.trade_date
            """
        else:
            # 不复权，使用原始 daily 表
            select_clause = ', '.join([f'd.{f}' for f in fields])
            query = f"""
            SELECT
                d.ts_code,
                d.trade_date,
                {select_clause}
            FROM daily d
            WHERE d.ts_code IN ({','.join(['?']*len(ts_codes))})
              AND d.trade_date BETWEEN ? AND ?
            ORDER BY d.ts_code, d.trade_date
            """

        params = ts_codes + [start_date, end_date]
        df = self.conn.execute(query, params).fetchdf()

        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 重命名复权列为标准名称
        if adj in ('qfq', 'hfq'):
            rename_map = {f'{f}_{adj}': f for f in ['open', 'high', 'low', 'close']}
            df = df.rename(columns=rename_map)

        # 设置多重索引
        df = df.set_index(['trade_date', 'ts_code'])

        return df

    def load_universe(
        self,
        date: str,
        filters: Optional[Dict] = None
    ) -> List[str]:
        """
        获取指定日期的股票池

        Args:
            date: 日期
            filters: 过滤条件 {
                'market': ['主板', '创业板'],
                'list_status': 'L',
                'min_list_days': 252,
                'exclude_st': True
            }
        """
        filters = filters or {}

        conditions = ["sb.list_status = 'L'"]
        params = []

        # 上市时间过滤
        if filters.get('min_list_days'):
            conditions.append(f"sb.list_date <= ?")
            # 计算 N 个交易日前的日期
            min_date = pd.to_datetime(date) - pd.Timedelta(days=int(filters['min_list_days'] * 1.5))
            params.append(min_date.strftime('%Y%m%d'))

        # 市场过滤
        if filters.get('market'):
            markets = filters['market']
            conditions.append(f"sb.market IN ({','.join(['?']*len(markets))})")
            params.extend(markets)

        # ST 过滤
        if filters.get('exclude_st', True):
            conditions.append("sb.name NOT LIKE '%ST%'")

        query = f"""
        SELECT sb.ts_code
        FROM stock_basic sb
        WHERE {' AND '.join(conditions)}
        """

        df = self.conn.execute(query, params).fetchdf()
        return df['ts_code'].tolist()

    def load_benchmark(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """加载基准指数数据"""
        query = """
        SELECT trade_date, close, pct_chg
        FROM index_daily
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df = self.conn.execute(query, [index_code, start_date, end_date]).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        return df

    def preload_data(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        批量预加载数据到内存

        优化策略：
        1. 使用 DuckDB 的并行查询
        2. 按股票代码分区存储
        3. 使用 Arrow 格式减少内存拷贝
        """
        # 批量查询
        query = """
        SELECT
            d.ts_code,
            d.trade_date,
            d.open_qfq as open,
            d.high_qfq as high,
            d.low_qfq as low,
            d.close_qfq as close,
            d.vol as volume,
            d.amount,
            d.adj_factor,
            d.turnover_rate,
            d.pe_ttm,
            d.pb,
            d.total_mv
        FROM stk_factor_pro d
        WHERE d.ts_code IN (SELECT UNNEST(?::VARCHAR[]))
          AND d.trade_date BETWEEN ? AND ?
        ORDER BY d.ts_code, d.trade_date
        """

        df = self.conn.execute(query, [ts_codes, start_date, end_date]).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 按股票分组存储
        data_dict = {}
        for ts_code, group in df.groupby('ts_code'):
            data_dict[ts_code] = group.set_index('trade_date').copy()

        self._price_cache.update(data_dict)
        return data_dict

    def get_bar(self, ts_code: str, trade_date: pd.Timestamp) -> Optional[BarData]:
        """获取单根K线数据"""
        if ts_code not in self._price_cache:
            return None

        df = self._price_cache[ts_code]
        if trade_date not in df.index:
            return None

        row = df.loc[trade_date]
        return BarData(
            datetime=trade_date,
            symbol=ts_code,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            amount=row['amount'],
            adj_close=row['close']
        )

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
```

### 3.2 复权处理

```python
class AdjustmentHandler:
    """复权处理器"""

    @staticmethod
    def forward_adjust(
        prices: pd.DataFrame,
        adj_factors: pd.Series,
        base_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        前复权处理

        前复权以最新价格为基准，向前调整历史价格
        公式: 前复权价 = 原始价 × (当日复权因子 / 最新复权因子)

        Args:
            prices: 原始价格 DataFrame
            adj_factors: 复权因子 Series
            base_date: 基准日期，默认为最新日期
        """
        if base_date is None:
            base_factor = adj_factors.iloc[-1]
        else:
            base_factor = adj_factors.loc[base_date]

        adj_ratio = adj_factors / base_factor

        price_cols = ['open', 'high', 'low', 'close']
        adjusted = prices.copy()
        for col in price_cols:
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * adj_ratio

        return adjusted

    @staticmethod
    def backward_adjust(
        prices: pd.DataFrame,
        adj_factors: pd.Series
    ) -> pd.DataFrame:
        """
        后复权处理

        后复权以上市首日价格为基准，向后调整价格
        公式: 后复权价 = 原始价 × 当日复权因子
        """
        price_cols = ['open', 'high', 'low', 'close']
        adjusted = prices.copy()
        for col in price_cols:
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * adj_factors

        return adjusted
```

### 3.3 内存数据结构

```python
from collections import OrderedDict
from typing import Iterator

class TimeSeriesBuffer:
    """
    时间序列数据缓冲区

    特性:
    - 固定窗口大小，自动淘汰旧数据
    - O(1) 的插入和查询
    - 支持切片访问
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._data: OrderedDict = OrderedDict()

    def append(self, timestamp: pd.Timestamp, value: any):
        """添加数据点"""
        self._data[timestamp] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def get(self, timestamp: pd.Timestamp) -> any:
        """获取指定时间的数据"""
        return self._data.get(timestamp)

    def last(self, n: int = 1) -> list:
        """获取最近 n 条数据"""
        items = list(self._data.items())[-n:]
        return [v for _, v in items]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame.from_dict(self._data, orient='index')

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator:
        return iter(self._data.items())


class MarketDataManager:
    """
    市场数据管理器

    管理多只股票的实时行情数据
    """

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self._buffers: Dict[str, TimeSeriesBuffer] = {}
        self._current_time: Optional[pd.Timestamp] = None

    def update(self, ts_code: str, bar: BarData):
        """更新股票行情"""
        if ts_code not in self._buffers:
            self._buffers[ts_code] = TimeSeriesBuffer(self.lookback_window)

        self._buffers[ts_code].append(bar.datetime, bar)
        self._current_time = bar.datetime

    def get_history(
        self,
        ts_code: str,
        field: str = 'close',
        periods: int = 20
    ) -> pd.Series:
        """获取历史数据序列"""
        if ts_code not in self._buffers:
            return pd.Series()

        buffer = self._buffers[ts_code]
        data = buffer.last(periods)

        return pd.Series(
            [getattr(bar, field) for bar in data],
            index=[bar.datetime for bar in data]
        )

    def get_latest(self, ts_code: str) -> Optional[BarData]:
        """获取最新行情"""
        if ts_code not in self._buffers:
            return None
        return self._buffers[ts_code].last(1)[0] if len(self._buffers[ts_code]) > 0 else None

    def get_snapshot(self) -> Dict[str, BarData]:
        """获取所有股票的最新快照"""
        return {
            ts_code: buffer.last(1)[0]
            for ts_code, buffer in self._buffers.items()
            if len(buffer) > 0
        }
```

---

## 4. 事件引擎 (EventEngine)

### 4.1 事件类型定义

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable
import queue
from threading import Thread
from datetime import datetime

class EventType(Enum):
    """事件类型枚举"""
    # 行情事件
    BAR = auto()            # K线数据
    TICK = auto()           # Tick数据

    # 交易事件
    ORDER = auto()          # 订单事件
    FILL = auto()           # 成交事件
    POSITION = auto()       # 持仓变化

    # 系统事件
    TIMER = auto()          # 定时器
    ENGINE_START = auto()   # 引擎启动
    ENGINE_STOP = auto()    # 引擎停止

    # 策略事件
    SIGNAL = auto()         # 信号事件
    REBALANCE = auto()      # 调仓事件


@dataclass
class Event:
    """事件基类"""
    type: EventType
    data: Any
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BarEvent(Event):
    """K线事件"""
    type: EventType = EventType.BAR
    data: BarData = None


@dataclass
class OrderEvent(Event):
    """订单事件"""
    type: EventType = EventType.ORDER
    data: 'Order' = None


@dataclass
class FillEvent(Event):
    """成交事件"""
    type: EventType = EventType.FILL
    data: 'Fill' = None
```

### 4.2 事件引擎实现

```python
from typing import List, Dict, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EventEngine:
    """
    事件驱动引擎

    职责:
    1. 事件队列管理
    2. 事件分发
    3. 回测循环控制
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False

    def register(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.name}")

    def unregister(self, event_type: EventType, handler: Callable):
        """注销事件处理器"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def put(self, event: Event):
        """将事件放入队列"""
        self._queue.put(event)

    def dispatch(self, event: Event):
        """分发事件到对应的处理器"""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error handling {event.type.name}: {e}")
                raise

    def process(self):
        """处理队列中的一个事件"""
        if not self._queue.empty():
            event = self._queue.get()
            self.dispatch(event)
            return True
        return False


class BacktestEngine:
    """
    回测引擎主类

    支持两种模式:
    1. 事件驱动模式：适合复杂策略，支持订单管理
    2. 向量化模式：适合简单策略，高性能
    """

    def __init__(
        self,
        data_loader: DataLoader,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.0003,  # 万三
        slippage: float = 0.001,          # 0.1%
        stamp_duty: float = 0.001         # 印花税 0.1% (卖出)
    ):
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.stamp_duty = stamp_duty

        # 核心组件
        self.event_engine = EventEngine()
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.strategy: Optional[BaseStrategy] = None
        self.analyzer: Optional[PerformanceAnalyzer] = None

        # 回测状态
        self._current_date: Optional[pd.Timestamp] = None
        self._trading_dates: List[pd.Timestamp] = []
        self._is_running: bool = False

    def set_strategy(self, strategy: 'BaseStrategy'):
        """设置策略"""
        self.strategy = strategy
        strategy.set_engine(self)

    def run(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[List[str]] = None,
        benchmark: str = '000001.SH',
        mode: str = 'event'  # 'event' or 'vectorized'
    ) -> 'BacktestResult':
        """
        运行回测

        Args:
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            universe: 股票池，None表示全市场
            benchmark: 基准指数代码
            mode: 回测模式
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")

        # 1. 初始化
        self._initialize(start_date, end_date, universe, benchmark)

        # 2. 运行回测
        if mode == 'event':
            self._run_event_driven()
        else:
            self._run_vectorized()

        # 3. 生成报告
        result = self._generate_result()

        logger.info("Backtest completed")
        return result

    def _initialize(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[List[str]],
        benchmark: str
    ):
        """初始化回测环境"""
        # 获取交易日历
        self._trading_dates = self.data_loader.get_trading_calendar(
            start_date, end_date
        ).tolist()

        # 获取股票池
        if universe is None:
            universe = self.data_loader.load_universe(start_date)

        # 预加载数据
        self.data_loader.preload_data(universe, start_date, end_date)

        # 初始化组件
        self.order_manager = OrderManager(self)
        self.position_manager = PositionManager(
            self.initial_capital,
            commission_rate=self.commission_rate,
            stamp_duty=self.stamp_duty
        )
        self.analyzer = PerformanceAnalyzer(benchmark)

        # 加载基准数据
        self._benchmark_data = self.data_loader.load_benchmark(
            benchmark, start_date, end_date
        )

        # 注册事件处理器
        self._register_handlers()

        # 初始化策略
        if self.strategy:
            self.strategy.on_init()

    def _register_handlers(self):
        """注册事件处理器"""
        self.event_engine.register(EventType.BAR, self._on_bar)
        self.event_engine.register(EventType.ORDER, self._on_order)
        self.event_engine.register(EventType.FILL, self._on_fill)

    def _run_event_driven(self):
        """事件驱动回测"""
        self._is_running = True
        market_data_mgr = MarketDataManager()

        for trade_date in self._trading_dates:
            self._current_date = trade_date

            # 获取当日所有股票的行情
            for ts_code, df in self.data_loader._price_cache.items():
                if trade_date in df.index:
                    bar = self.data_loader.get_bar(ts_code, trade_date)
                    if bar:
                        market_data_mgr.update(ts_code, bar)

                        # 发送K线事件
                        event = BarEvent(data=bar, timestamp=trade_date)
                        self.event_engine.put(event)

            # 处理所有事件
            while self.event_engine.process():
                pass

            # 日终处理
            self._on_day_end(trade_date, market_data_mgr)

        self._is_running = False

    def _on_bar(self, event: BarEvent):
        """处理K线事件"""
        if self.strategy:
            self.strategy.on_bar(event.data)

    def _on_order(self, event: OrderEvent):
        """处理订单事件"""
        order = event.data
        self.order_manager.process_order(order)

    def _on_fill(self, event: FillEvent):
        """处理成交事件"""
        fill = event.data
        self.position_manager.update_position(fill)

        if self.strategy:
            self.strategy.on_order_filled(fill)

    def _on_day_end(self, trade_date: pd.Timestamp, market_data: MarketDataManager):
        """日终处理"""
        # 更新持仓市值
        snapshot = market_data.get_snapshot()
        self.position_manager.mark_to_market(snapshot)

        # 记录净值
        nav = self.position_manager.get_nav()
        self.analyzer.record_nav(trade_date, nav)

    def _run_vectorized(self):
        """向量化回测"""
        # 向量化模式下，策略需要返回信号矩阵
        if not hasattr(self.strategy, 'generate_signals'):
            raise ValueError("Strategy must implement generate_signals for vectorized mode")

        # 获取所有价格数据
        prices = pd.DataFrame({
            ts_code: df['close']
            for ts_code, df in self.data_loader._price_cache.items()
        })

        # 生成信号
        entries, exits = self.strategy.generate_signals(prices)

        # 使用向量化方式计算收益
        # ... 省略具体实现

    def _generate_result(self) -> 'BacktestResult':
        """生成回测结果"""
        return self.analyzer.generate_report(
            self.position_manager.get_trade_history(),
            self._benchmark_data
        )

    # 供策略调用的接口
    def buy(
        self,
        ts_code: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = 'MARKET'
    ) -> str:
        """买入"""
        order = Order(
            ts_code=ts_code,
            direction=OrderDirection.BUY,
            volume=volume,
            price=price,
            order_type=OrderType[order_type],
            timestamp=self._current_date
        )
        return self.order_manager.submit_order(order)

    def sell(
        self,
        ts_code: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = 'MARKET'
    ) -> str:
        """卖出"""
        order = Order(
            ts_code=ts_code,
            direction=OrderDirection.SELL,
            volume=volume,
            price=price,
            order_type=OrderType[order_type],
            timestamp=self._current_date
        )
        return self.order_manager.submit_order(order)

    def get_position(self, ts_code: str) -> int:
        """获取持仓"""
        return self.position_manager.get_position(ts_code)

    def get_cash(self) -> float:
        """获取可用资金"""
        return self.position_manager.cash

    def get_nav(self) -> float:
        """获取净值"""
        return self.position_manager.get_nav()
```

---

## 5. 交易模拟模块

### 5.1 订单管理 (OrderManager)

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import uuid

class OrderType(Enum):
    """订单类型"""
    MARKET = auto()       # 市价单
    LIMIT = auto()        # 限价单
    STOP = auto()         # 止损单
    STOP_LIMIT = auto()   # 止损限价单
    MOC = auto()          # 收盘价成交


class OrderDirection(Enum):
    """买卖方向"""
    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    """订单状态"""
    PENDING = auto()      # 待提交
    SUBMITTED = auto()    # 已提交
    PARTIAL = auto()      # 部分成交
    FILLED = auto()       # 全部成交
    CANCELLED = auto()    # 已撤销
    REJECTED = auto()     # 已拒绝


@dataclass
class Order:
    """订单"""
    ts_code: str
    direction: OrderDirection
    volume: int
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = None

    # 订单状态
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: int = 0
    filled_amount: float = 0.0
    avg_price: float = 0.0

    # 止损止盈
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Fill:
    """成交记录"""
    order_id: str
    ts_code: str
    direction: OrderDirection
    volume: int
    price: float
    amount: float
    commission: float
    slippage_cost: float
    timestamp: datetime

    @property
    def total_cost(self) -> float:
        """总成本（含手续费和滑点）"""
        return self.amount + self.commission + self.slippage_cost


class OrderManager:
    """
    订单管理器

    职责:
    1. 订单生命周期管理
    2. 订单撮合模拟
    3. 风控检查
    """

    def __init__(self, engine: 'BacktestEngine'):
        self.engine = engine
        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: Dict[str, Order] = {}
        self._cancelled_orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []

    def submit_order(self, order: Order) -> str:
        """
        提交订单

        Returns:
            订单ID
        """
        # 风控检查
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order.order_id

        order.status = OrderStatus.SUBMITTED
        self._pending_orders[order.order_id] = order

        # 市价单立即撮合
        if order.order_type == OrderType.MARKET:
            self._match_order(order)

        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if order_id in self._pending_orders:
            order = self._pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self._cancelled_orders[order_id] = order
            return True
        return False

    def process_order(self, order: Order):
        """处理订单（用于限价单等需要等待的订单）"""
        if order.order_type == OrderType.LIMIT:
            self._match_limit_order(order)
        elif order.order_type == OrderType.STOP:
            self._check_stop_order(order)

    def _validate_order(self, order: Order) -> bool:
        """订单验证"""
        # 检查股票代码
        if not order.ts_code:
            return False

        # 检查数量（必须是100的整数倍）
        if order.volume <= 0 or order.volume % 100 != 0:
            return False

        # 检查卖出时持仓是否足够
        if order.direction == OrderDirection.SELL:
            position = self.engine.position_manager.get_position(order.ts_code)
            if position < order.volume:
                return False

        # 检查买入时资金是否足够
        if order.direction == OrderDirection.BUY:
            # 估算所需资金
            price = order.price or self._get_current_price(order.ts_code)
            if price:
                required = price * order.volume * (1 + self.engine.commission_rate + self.engine.slippage)
                if self.engine.position_manager.cash < required:
                    return False

        return True

    def _match_order(self, order: Order):
        """撮合订单"""
        # 获取成交价格
        fill_price = self._get_fill_price(order)
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            return

        # 计算滑点
        slippage_cost = self._calculate_slippage(order, fill_price)

        # 计算手续费
        amount = fill_price * order.volume
        commission = self._calculate_commission(order, amount)

        # 创建成交记录
        fill = Fill(
            order_id=order.order_id,
            ts_code=order.ts_code,
            direction=order.direction,
            volume=order.volume,
            price=fill_price,
            amount=amount,
            commission=commission,
            slippage_cost=slippage_cost,
            timestamp=order.timestamp
        )

        # 更新订单状态
        order.filled_volume = order.volume
        order.filled_amount = amount
        order.avg_price = fill_price
        order.status = OrderStatus.FILLED

        # 移动到已成交订单
        if order.order_id in self._pending_orders:
            self._pending_orders.pop(order.order_id)
        self._filled_orders[order.order_id] = order
        self._fills.append(fill)

        # 发送成交事件
        event = FillEvent(data=fill, timestamp=order.timestamp)
        self.engine.event_engine.put(event)

    def _get_fill_price(self, order: Order) -> Optional[float]:
        """获取成交价格"""
        bar = self.engine.data_loader.get_bar(order.ts_code, order.timestamp)
        if bar is None:
            return None

        if order.order_type == OrderType.MARKET:
            # 市价单使用开盘价（假设在开盘时下单）
            return bar.open
        elif order.order_type == OrderType.LIMIT:
            # 限价单检查是否可成交
            if order.direction == OrderDirection.BUY:
                if bar.low <= order.price:
                    return min(order.price, bar.open)
            else:
                if bar.high >= order.price:
                    return max(order.price, bar.open)
        elif order.order_type == OrderType.MOC:
            # 收盘价成交
            return bar.close

        return None

    def _calculate_slippage(self, order: Order, price: float) -> float:
        """计算滑点成本"""
        slippage_ratio = self.engine.slippage

        if order.direction == OrderDirection.BUY:
            # 买入时滑点向上
            slippage = price * order.volume * slippage_ratio
        else:
            # 卖出时滑点向下
            slippage = price * order.volume * slippage_ratio

        return slippage

    def _calculate_commission(self, order: Order, amount: float) -> float:
        """计算手续费"""
        # 佣金
        commission = amount * self.engine.commission_rate
        commission = max(commission, 5.0)  # 最低5元

        # 印花税（仅卖出时收取）
        if order.direction == OrderDirection.SELL:
            commission += amount * self.engine.stamp_duty

        return commission

    def _get_current_price(self, ts_code: str) -> Optional[float]:
        """获取当前价格"""
        bar = self.engine.data_loader.get_bar(ts_code, self.engine._current_date)
        return bar.close if bar else None

    def get_fills(self) -> List[Fill]:
        """获取所有成交记录"""
        return self._fills.copy()
```

### 5.2 持仓管理 (PositionManager)

```python
@dataclass
class Position:
    """持仓信息"""
    ts_code: str
    volume: int = 0
    cost_basis: float = 0.0      # 成本基础
    avg_price: float = 0.0       # 平均成本
    market_value: float = 0.0    # 市值
    unrealized_pnl: float = 0.0  # 未实现盈亏
    realized_pnl: float = 0.0    # 已实现盈亏


class PositionManager:
    """
    持仓管理器

    职责:
    1. 持仓跟踪
    2. 资金管理
    3. 盈亏计算
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty

        self.cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trade_history: List[Dict] = []
        self._nav_history: List[Tuple[datetime, float]] = []

    def update_position(self, fill: Fill):
        """根据成交更新持仓"""
        ts_code = fill.ts_code

        if ts_code not in self._positions:
            self._positions[ts_code] = Position(ts_code=ts_code)

        pos = self._positions[ts_code]

        if fill.direction == OrderDirection.BUY:
            # 买入：更新持仓和成本
            new_cost = pos.cost_basis + fill.total_cost
            new_volume = pos.volume + fill.volume

            pos.volume = new_volume
            pos.cost_basis = new_cost
            pos.avg_price = new_cost / new_volume if new_volume > 0 else 0

            self.cash -= fill.total_cost

        else:
            # 卖出：计算已实现盈亏
            sell_ratio = fill.volume / pos.volume
            cost_of_sold = pos.cost_basis * sell_ratio

            # 卖出收入（扣除手续费和印花税）
            sell_proceeds = fill.amount - fill.commission - fill.slippage_cost

            # 已实现盈亏
            realized_pnl = sell_proceeds - cost_of_sold
            pos.realized_pnl += realized_pnl

            # 更新持仓
            pos.volume -= fill.volume
            pos.cost_basis -= cost_of_sold

            if pos.volume == 0:
                pos.avg_price = 0

            self.cash += sell_proceeds

        # 记录交易
        self._trade_history.append({
            'timestamp': fill.timestamp,
            'ts_code': ts_code,
            'direction': fill.direction.name,
            'volume': fill.volume,
            'price': fill.price,
            'amount': fill.amount,
            'commission': fill.commission,
            'realized_pnl': pos.realized_pnl
        })

    def mark_to_market(self, market_snapshot: Dict[str, BarData]):
        """按市价重估持仓"""
        for ts_code, pos in self._positions.items():
            if pos.volume > 0 and ts_code in market_snapshot:
                bar = market_snapshot[ts_code]
                pos.market_value = bar.close * pos.volume
                pos.unrealized_pnl = pos.market_value - pos.cost_basis

    def get_position(self, ts_code: str) -> int:
        """获取持仓数量"""
        if ts_code in self._positions:
            return self._positions[ts_code].volume
        return 0

    def get_holdings(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return {k: v for k, v in self._positions.items() if v.volume > 0}

    def get_nav(self) -> float:
        """获取净资产"""
        total_market_value = sum(
            pos.market_value for pos in self._positions.values() if pos.volume > 0
        )
        return self.cash + total_market_value

    def get_total_value(self) -> float:
        """获取总资产（同get_nav）"""
        return self.get_nav()

    def get_cash_ratio(self) -> float:
        """获取现金占比"""
        nav = self.get_nav()
        return self.cash / nav if nav > 0 else 1.0

    def get_trade_history(self) -> pd.DataFrame:
        """获取交易历史"""
        return pd.DataFrame(self._trade_history)

    def record_nav(self, timestamp: datetime, nav: float):
        """记录净值"""
        self._nav_history.append((timestamp, nav))

    def get_nav_history(self) -> pd.Series:
        """获取净值历史"""
        if not self._nav_history:
            return pd.Series()
        dates, navs = zip(*self._nav_history)
        return pd.Series(navs, index=dates)
```

### 5.3 成交模型

```python
class FillModel:
    """
    成交模型基类

    模拟真实市场的成交行为
    """

    def get_fill_price(
        self,
        order: Order,
        bar: BarData
    ) -> Tuple[Optional[float], int]:
        """
        获取成交价格和数量

        Returns:
            (成交价格, 成交数量)，价格为None表示无法成交
        """
        raise NotImplementedError


class SimpleFillModel(FillModel):
    """简单成交模型：全部成交，无部分成交"""

    def get_fill_price(
        self,
        order: Order,
        bar: BarData
    ) -> Tuple[Optional[float], int]:
        if order.order_type == OrderType.MARKET:
            return bar.open, order.volume
        elif order.order_type == OrderType.LIMIT:
            if order.direction == OrderDirection.BUY:
                if bar.low <= order.price:
                    return min(order.price, bar.open), order.volume
            else:
                if bar.high >= order.price:
                    return max(order.price, bar.open), order.volume
        elif order.order_type == OrderType.MOC:
            return bar.close, order.volume

        return None, 0


class VolumeWeightedFillModel(FillModel):
    """
    成交量加权成交模型

    考虑成交量限制和市场冲击
    """

    def __init__(
        self,
        volume_limit_ratio: float = 0.1,  # 最大可成交比例
        price_impact_coef: float = 0.1     # 价格冲击系数
    ):
        self.volume_limit_ratio = volume_limit_ratio
        self.price_impact_coef = price_impact_coef

    def get_fill_price(
        self,
        order: Order,
        bar: BarData
    ) -> Tuple[Optional[float], int]:
        # 限制成交量
        max_volume = int(bar.volume * self.volume_limit_ratio)
        fill_volume = min(order.volume, max_volume)

        if fill_volume == 0:
            return None, 0

        # 计算市场冲击
        participation_rate = fill_volume / bar.volume
        price_impact = participation_rate * self.price_impact_coef

        # VWAP 估计（使用当日高低收价的加权平均）
        vwap = (bar.high + bar.low + bar.close * 2) / 4

        if order.direction == OrderDirection.BUY:
            fill_price = vwap * (1 + price_impact)
            # 不能超过当日最高价
            fill_price = min(fill_price, bar.high)
        else:
            fill_price = vwap * (1 - price_impact)
            # 不能低于当日最低价
            fill_price = max(fill_price, bar.low)

        return fill_price, fill_volume


class SlippageModel:
    """滑点模型"""

    @staticmethod
    def fixed_slippage(price: float, slippage_pct: float, direction: OrderDirection) -> float:
        """固定百分比滑点"""
        if direction == OrderDirection.BUY:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    @staticmethod
    def volume_based_slippage(
        price: float,
        order_volume: int,
        bar_volume: float,
        base_slippage: float = 0.001
    ) -> float:
        """基于成交量的滑点"""
        participation = order_volume / bar_volume
        slippage = base_slippage * (1 + participation * 10)
        return slippage
```

---

## 6. 策略接口设计

### 6.1 策略基类

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    策略基类

    所有策略必须继承此类并实现相应方法
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.engine: Optional['BacktestEngine'] = None
        self._params: Dict = {}

    def set_engine(self, engine: 'BacktestEngine'):
        """设置回测引擎引用"""
        self.engine = engine

    def set_params(self, **kwargs):
        """设置策略参数"""
        self._params.update(kwargs)

    def get_param(self, key: str, default=None):
        """获取策略参数"""
        return self._params.get(key, default)

    # ========== 生命周期方法 ==========

    def on_init(self):
        """
        策略初始化
        在回测开始前调用，用于初始化指标计算器等
        """
        pass

    def on_start(self):
        """策略启动"""
        pass

    def on_stop(self):
        """策略停止"""
        pass

    # ========== 事件处理方法 ==========

    @abstractmethod
    def on_bar(self, bar: BarData):
        """
        K线数据回调

        每根K线推送时调用，这是策略的核心入口

        Args:
            bar: K线数据
        """
        pass

    def on_order_filled(self, fill: Fill):
        """
        订单成交回调

        Args:
            fill: 成交信息
        """
        pass

    def on_order_rejected(self, order: Order):
        """订单拒绝回调"""
        pass

    def on_order_cancelled(self, order: Order):
        """订单撤销回调"""
        pass

    # ========== 向量化接口 ==========

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成交易信号（向量化模式）

        Args:
            data: 包含 OHLCV 数据的字典

        Returns:
            entries: 入场信号 DataFrame (True/False)
            exits: 出场信号 DataFrame (True/False)
        """
        raise NotImplementedError("向量化模式需要实现 generate_signals 方法")

    # ========== 便捷方法 ==========

    def buy(
        self,
        ts_code: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = 'MARKET'
    ) -> str:
        """买入"""
        return self.engine.buy(ts_code, volume, price, order_type)

    def sell(
        self,
        ts_code: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = 'MARKET'
    ) -> str:
        """卖出"""
        return self.engine.sell(ts_code, volume, price, order_type)

    def get_position(self, ts_code: str) -> int:
        """获取持仓"""
        return self.engine.get_position(ts_code)

    def get_cash(self) -> float:
        """获取现金"""
        return self.engine.get_cash()

    def get_nav(self) -> float:
        """获取净值"""
        return self.engine.get_nav()


class SignalStrategy(BaseStrategy):
    """
    信号型策略基类

    简化的策略模板，只需实现信号生成逻辑
    """

    def __init__(
        self,
        name: str = "SignalStrategy",
        position_size: float = 0.1,  # 每个信号的仓位比例
        max_positions: int = 10       # 最大持仓数
    ):
        super().__init__(name)
        self.position_size = position_size
        self.max_positions = max_positions
        self._signals: Dict[str, int] = {}  # ts_code -> signal (1=buy, -1=sell, 0=hold)

    @abstractmethod
    def calculate_signal(self, ts_code: str, bar: BarData) -> int:
        """
        计算交易信号

        Args:
            ts_code: 股票代码
            bar: K线数据

        Returns:
            1: 买入信号
            -1: 卖出信号
            0: 无信号
        """
        pass

    def on_bar(self, bar: BarData):
        """处理K线数据"""
        signal = self.calculate_signal(bar.symbol, bar)
        self._signals[bar.symbol] = signal

        current_position = self.get_position(bar.symbol)

        if signal == 1 and current_position == 0:
            # 买入信号且无持仓
            available_cash = self.get_cash() * self.position_size
            volume = int(available_cash / bar.close / 100) * 100
            if volume >= 100:
                self.buy(bar.symbol, volume)

        elif signal == -1 and current_position > 0:
            # 卖出信号且有持仓
            self.sell(bar.symbol, current_position)
```

### 6.2 策略接口规范

```python
class IStrategy(ABC):
    """策略接口规范"""

    @abstractmethod
    def on_bar(self, bar: BarData) -> None:
        """
        K线数据推送回调

        触发时机: 每根K线完成时
        用途: 策略主逻辑，信号生成，下单决策

        Args:
            bar: K线数据，包含 OHLCV 和时间戳
        """
        pass

    @abstractmethod
    def on_order_filled(self, fill: Fill) -> None:
        """
        订单成交回调

        触发时机: 订单成交后
        用途: 更新策略内部状态，调整风控参数

        Args:
            fill: 成交信息，包含价格、数量、费用
        """
        pass

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量生成信号（向量化模式）

        Args:
            data: 市场数据字典，键为字段名（'open', 'close'等）
                  值为 DataFrame，index为日期，columns为股票代码

        Returns:
            entries: 买入信号矩阵 (bool DataFrame)
            exits: 卖出信号矩阵 (bool DataFrame)

        Note:
            - 信号应该基于历史数据生成，避免未来函数
            - 使用 shift(1) 确保使用前一日数据决策
        """
        pass
```

---

## 7. 绩效分析模块 (PerformanceAnalyzer)

### 7.1 收益率计算

```python
class PerformanceAnalyzer:
    """
    绩效分析器

    计算各类绩效指标和风险指标
    """

    def __init__(self, benchmark_code: str = '000001.SH'):
        self.benchmark_code = benchmark_code
        self._nav_history: List[Tuple[datetime, float]] = []
        self._benchmark_returns: Optional[pd.Series] = None

    def record_nav(self, timestamp: datetime, nav: float):
        """记录每日净值"""
        self._nav_history.append((timestamp, nav))

    def set_benchmark(self, benchmark_returns: pd.Series):
        """设置基准收益率"""
        self._benchmark_returns = benchmark_returns

    def calculate_returns(self) -> pd.DataFrame:
        """
        计算收益率序列

        Returns:
            DataFrame with columns: nav, daily_return, cumulative_return
        """
        if not self._nav_history:
            return pd.DataFrame()

        dates, navs = zip(*self._nav_history)
        nav_series = pd.Series(navs, index=dates, name='nav')

        df = pd.DataFrame({'nav': nav_series})
        df['daily_return'] = df['nav'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        return df

    def calculate_risk_metrics(self, risk_free_rate: float = 0.03) -> Dict:
        """
        计算风险指标

        Args:
            risk_free_rate: 年化无风险利率

        Returns:
            风险指标字典
        """
        returns_df = self.calculate_returns()
        if returns_df.empty:
            return {}

        daily_returns = returns_df['daily_return'].dropna()

        # 年化因子
        ann_factor = 252

        # 总收益率
        total_return = returns_df['cumulative_return'].iloc[-1]

        # 年化收益率
        n_days = len(daily_returns)
        annual_return = (1 + total_return) ** (ann_factor / n_days) - 1

        # 年化波动率
        annual_volatility = daily_returns.std() * np.sqrt(ann_factor)

        # 夏普比率
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min()

        # 最大回撤持续期
        dd_start = drawdown.idxmin()
        dd_end = cumulative[dd_start:].idxmax() if dd_start < cumulative.index[-1] else cumulative.index[-1]
        max_dd_duration = (dd_end - dd_start).days

        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # 胜率
        win_rate = (daily_returns > 0).mean()

        # 盈亏比
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'trading_days': n_days
        }

    def calculate_alpha_beta(self) -> Dict:
        """计算Alpha和Beta"""
        if self._benchmark_returns is None:
            return {'alpha': None, 'beta': None}

        returns_df = self.calculate_returns()
        strategy_returns = returns_df['daily_return'].dropna()

        # 对齐日期
        common_dates = strategy_returns.index.intersection(self._benchmark_returns.index)
        strategy = strategy_returns.loc[common_dates]
        benchmark = self._benchmark_returns.loc[common_dates]

        # 计算Beta
        covariance = np.cov(strategy, benchmark)[0, 1]
        variance = np.var(benchmark)
        beta = covariance / variance if variance > 0 else 0

        # 计算Alpha (年化)
        strategy_annual = strategy.mean() * 252
        benchmark_annual = benchmark.mean() * 252
        alpha = strategy_annual - beta * benchmark_annual

        # R-squared
        correlation = np.corrcoef(strategy, benchmark)[0, 1]
        r_squared = correlation ** 2

        # Information Ratio
        excess_returns = strategy - benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

    def calculate_monthly_returns(self) -> pd.DataFrame:
        """计算月度收益率表"""
        returns_df = self.calculate_returns()
        if returns_df.empty:
            return pd.DataFrame()

        daily_returns = returns_df['daily_return']

        # 转换为月度收益
        monthly = (1 + daily_returns).resample('M').prod() - 1

        # 创建年月矩阵
        monthly_matrix = monthly.groupby([monthly.index.year, monthly.index.month]).first()
        monthly_matrix = monthly_matrix.unstack()
        monthly_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # 添加年度收益
        yearly = (1 + daily_returns).resample('Y').prod() - 1
        monthly_matrix['Year'] = yearly.values

        return monthly_matrix

    def generate_report(
        self,
        trade_history: pd.DataFrame,
        benchmark_data: pd.DataFrame
    ) -> 'BacktestResult':
        """生成完整报告"""
        # 设置基准收益率
        if 'pct_chg' in benchmark_data.columns:
            self._benchmark_returns = benchmark_data['pct_chg'] / 100
        elif 'close' in benchmark_data.columns:
            self._benchmark_returns = benchmark_data['close'].pct_change()

        # 计算各项指标
        returns_df = self.calculate_returns()
        risk_metrics = self.calculate_risk_metrics()
        alpha_beta = self.calculate_alpha_beta()
        monthly_returns = self.calculate_monthly_returns()

        # 交易统计
        trade_stats = self._calculate_trade_stats(trade_history)

        return BacktestResult(
            nav_history=returns_df,
            risk_metrics=risk_metrics,
            alpha_beta=alpha_beta,
            monthly_returns=monthly_returns,
            trade_stats=trade_stats,
            trade_history=trade_history
        )

    def _calculate_trade_stats(self, trade_history: pd.DataFrame) -> Dict:
        """计算交易统计"""
        if trade_history.empty:
            return {}

        total_trades = len(trade_history)

        # 分离买卖
        buys = trade_history[trade_history['direction'] == 'BUY']
        sells = trade_history[trade_history['direction'] == 'SELL']

        # 如果有已实现盈亏
        if 'realized_pnl' in trade_history.columns:
            profitable = sells[sells['realized_pnl'] > 0]
            win_rate = len(profitable) / len(sells) if len(sells) > 0 else 0
        else:
            win_rate = None

        return {
            'total_trades': total_trades,
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'win_rate': win_rate,
            'avg_trade_amount': trade_history['amount'].mean() if 'amount' in trade_history else None,
            'total_commission': trade_history['commission'].sum() if 'commission' in trade_history else None
        }


@dataclass
class BacktestResult:
    """回测结果"""
    nav_history: pd.DataFrame
    risk_metrics: Dict
    alpha_beta: Dict
    monthly_returns: pd.DataFrame
    trade_stats: Dict
    trade_history: pd.DataFrame

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            "=" * 60,
            "回测结果摘要",
            "=" * 60,
            "",
            "【收益指标】",
            f"  总收益率: {self.risk_metrics.get('total_return', 0):.2%}",
            f"  年化收益率: {self.risk_metrics.get('annual_return', 0):.2%}",
            "",
            "【风险指标】",
            f"  年化波动率: {self.risk_metrics.get('annual_volatility', 0):.2%}",
            f"  最大回撤: {self.risk_metrics.get('max_drawdown', 0):.2%}",
            f"  夏普比率: {self.risk_metrics.get('sharpe_ratio', 0):.3f}",
            f"  卡尔玛比率: {self.risk_metrics.get('calmar_ratio', 0):.3f}",
            f"  索提诺比率: {self.risk_metrics.get('sortino_ratio', 0):.3f}",
            "",
            "【Alpha/Beta分析】",
            f"  Alpha: {self.alpha_beta.get('alpha', 0):.4f}",
            f"  Beta: {self.alpha_beta.get('beta', 0):.4f}",
            f"  信息比率: {self.alpha_beta.get('information_ratio', 0):.3f}",
            "",
            "【交易统计】",
            f"  总交易次数: {self.trade_stats.get('total_trades', 0)}",
            f"  胜率: {self.trade_stats.get('win_rate', 0):.2%}" if self.trade_stats.get('win_rate') else "  胜率: N/A",
            "=" * 60
        ]
        return "\n".join(lines)

    def to_html(self) -> str:
        """生成HTML报告"""
        # 省略具体实现，返回HTML字符串
        pass

    def plot(self):
        """绘制绩效图表"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 净值曲线
        ax1 = axes[0, 0]
        self.nav_history['nav'].plot(ax=ax1)
        ax1.set_title('Net Asset Value')
        ax1.set_ylabel('NAV')

        # 2. 回撤曲线
        ax2 = axes[0, 1]
        cumulative = (1 + self.nav_history['daily_return']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative / rolling_max - 1
        drawdown.plot(ax=ax2, color='red')
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')

        # 3. 月度收益热力图
        ax3 = axes[1, 0]
        if not self.monthly_returns.empty:
            import seaborn as sns
            monthly_data = self.monthly_returns.drop('Year', axis=1, errors='ignore')
            sns.heatmap(monthly_data * 100, annot=True, fmt='.1f',
                       cmap='RdYlGn', center=0, ax=ax3)
            ax3.set_title('Monthly Returns (%)')

        # 4. 收益分布
        ax4 = axes[1, 1]
        self.nav_history['daily_return'].hist(bins=50, ax=ax4)
        ax4.axvline(0, color='red', linestyle='--')
        ax4.set_title('Daily Returns Distribution')
        ax4.set_xlabel('Daily Return')

        plt.tight_layout()
        return fig
```

### 7.2 绩效归因分析

```python
class PerformanceAttribution:
    """
    绩效归因分析

    包括：
    1. Brinson 归因（资产配置 vs 个股选择）
    2. 因子归因
    """

    def __init__(self):
        pass

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> Dict:
        """
        Brinson 归因分析

        将超额收益分解为：
        - 资产配置效应 (Allocation Effect)
        - 个股选择效应 (Selection Effect)
        - 交互效应 (Interaction Effect)

        Args:
            portfolio_weights: 组合权重 (日期 x 股票)
            portfolio_returns: 组合中各股票收益率
            benchmark_weights: 基准权重
            benchmark_returns: 基准中各股票收益率
        """
        # 对齐数据
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        common_stocks = portfolio_weights.columns.intersection(benchmark_weights.columns)

        p_weights = portfolio_weights.loc[common_dates, common_stocks]
        p_returns = portfolio_returns.loc[common_dates, common_stocks]
        b_weights = benchmark_weights.loc[common_dates, common_stocks]
        b_returns = benchmark_returns.loc[common_dates, common_stocks]

        # 计算各效应
        allocation_effect = ((p_weights - b_weights) * b_returns).sum(axis=1)
        selection_effect = (b_weights * (p_returns - b_returns)).sum(axis=1)
        interaction_effect = ((p_weights - b_weights) * (p_returns - b_returns)).sum(axis=1)

        total_effect = allocation_effect + selection_effect + interaction_effect

        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_effect': total_effect,
            'cumulative_allocation': (1 + allocation_effect).cumprod() - 1,
            'cumulative_selection': (1 + selection_effect).cumprod() - 1,
            'cumulative_interaction': (1 + interaction_effect).cumprod() - 1
        }

    def factor_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: float = 0.03
    ) -> Dict:
        """
        因子归因分析

        使用多因子回归分解收益来源

        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率 DataFrame (日期 x 因子)
            risk_free_rate: 无风险利率（年化）
        """
        from sklearn.linear_model import LinearRegression

        # 对齐数据
        common_idx = returns.index.intersection(factor_returns.index)
        y = returns.loc[common_idx].values
        X = factor_returns.loc[common_idx].values

        # 日度无风险利率
        rf_daily = risk_free_rate / 252
        y_excess = y - rf_daily

        # 多因子回归
        model = LinearRegression()
        model.fit(X, y_excess)

        # 计算各因子贡献
        factor_exposures = dict(zip(factor_returns.columns, model.coef_))
        alpha = model.intercept_ * 252  # 年化Alpha

        # 预测收益
        predicted = model.predict(X)

        # R-squared
        ss_res = np.sum((y_excess - predicted) ** 2)
        ss_tot = np.sum((y_excess - y_excess.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        return {
            'alpha': alpha,
            'factor_exposures': factor_exposures,
            'r_squared': r_squared,
            'residual_return': y_excess - predicted
        }
```

---

## 8. 示例策略实现

### 8.1 双均线策略

```python
class DualMAStrategy(BaseStrategy):
    """
    双均线策略

    规则:
    - 短期均线上穿长期均线：买入
    - 短期均线下穿长期均线：卖出
    """

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        position_size: float = 0.1
    ):
        super().__init__(name=f"DualMA({short_period},{long_period})")
        self.short_period = short_period
        self.long_period = long_period
        self.position_size = position_size

        # 历史数据缓存
        self._price_history: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData):
        """处理K线数据"""
        ts_code = bar.symbol

        # 更新价格历史
        if ts_code not in self._price_history:
            self._price_history[ts_code] = []

        self._price_history[ts_code].append(bar.close)

        # 保留足够的历史数据
        if len(self._price_history[ts_code]) > self.long_period + 10:
            self._price_history[ts_code] = self._price_history[ts_code][-(self.long_period + 10):]

        # 计算均线
        prices = self._price_history[ts_code]
        if len(prices) < self.long_period:
            return

        short_ma = np.mean(prices[-self.short_period:])
        long_ma = np.mean(prices[-self.long_period:])

        # 计算前一日均线（用于判断金叉/死叉）
        prev_short_ma = np.mean(prices[-self.short_period-1:-1])
        prev_long_ma = np.mean(prices[-self.long_period-1:-1])

        # 获取当前持仓
        current_position = self.get_position(ts_code)

        # 金叉买入
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            if current_position == 0:
                available_cash = self.get_cash() * self.position_size
                volume = int(available_cash / bar.close / 100) * 100
                if volume >= 100:
                    self.buy(ts_code, volume)

        # 死叉卖出
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            if current_position > 0:
                self.sell(ts_code, current_position)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """向量化信号生成"""
        close = data['close']

        # 计算均线
        short_ma = close.rolling(self.short_period).mean()
        long_ma = close.rolling(self.long_period).mean()

        # 金叉：短均线上穿长均线
        entries = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))

        # 死叉：短均线下穿长均线
        exits = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

        return entries, exits
```

### 8.2 动量策略

```python
class MomentumStrategy(BaseStrategy):
    """
    动量策略

    规则:
    - 选取过去N日收益率最高的K只股票
    - 等权重持有，定期调仓
    """

    def __init__(
        self,
        lookback_period: int = 20,
        top_k: int = 10,
        rebalance_freq: int = 5,  # 每5天调仓
        holding_period: int = 5
    ):
        super().__init__(name=f"Momentum({lookback_period},{top_k})")
        self.lookback_period = lookback_period
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.holding_period = holding_period

        self._bar_count = 0
        self._price_history: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData):
        """处理K线数据"""
        ts_code = bar.symbol

        # 更新价格历史
        if ts_code not in self._price_history:
            self._price_history[ts_code] = []
        self._price_history[ts_code].append(bar.close)

        # 计数
        self._bar_count += 1

    def on_day_end(self):
        """日终处理 - 判断是否调仓"""
        if self._bar_count % self.rebalance_freq != 0:
            return

        # 计算动量
        momentum_scores = {}
        for ts_code, prices in self._price_history.items():
            if len(prices) >= self.lookback_period:
                momentum = (prices[-1] - prices[-self.lookback_period]) / prices[-self.lookback_period]
                momentum_scores[ts_code] = momentum

        if not momentum_scores:
            return

        # 选取动量最高的K只股票
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        target_stocks = [s[0] for s in sorted_stocks[:self.top_k]]

        # 调仓
        self._rebalance(target_stocks)

    def _rebalance(self, target_stocks: List[str]):
        """调仓到目标持仓"""
        current_holdings = {
            ts_code for ts_code in self._price_history
            if self.get_position(ts_code) > 0
        }

        # 卖出不在目标中的股票
        for ts_code in current_holdings - set(target_stocks):
            position = self.get_position(ts_code)
            if position > 0:
                self.sell(ts_code, position)

        # 计算每只股票的目标市值
        nav = self.get_nav()
        target_value_per_stock = nav / len(target_stocks)

        # 买入目标股票
        for ts_code in target_stocks:
            current_position = self.get_position(ts_code)
            current_value = current_position * self._price_history[ts_code][-1]

            if current_value < target_value_per_stock * 0.9:  # 偏离超过10%才调整
                # 需要买入
                additional_value = target_value_per_stock - current_value
                price = self._price_history[ts_code][-1]
                volume = int(additional_value / price / 100) * 100
                if volume >= 100:
                    self.buy(ts_code, volume)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """向量化信号生成"""
        close = data['close']

        # 计算动量
        momentum = close.pct_change(self.lookback_period)

        # 每天选取排名前K的股票
        rank = momentum.rank(axis=1, ascending=False)

        # 入场信号：排名进入前K
        entries = (rank <= self.top_k) & (rank.shift(1) > self.top_k)

        # 出场信号：排名跌出前K
        exits = (rank > self.top_k) & (rank.shift(1) <= self.top_k)

        return entries, exits
```

### 8.3 多因子策略

```python
class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略

    综合多个因子进行选股:
    1. 价值因子 (PE, PB)
    2. 质量因子 (ROE)
    3. 动量因子
    4. 规模因子 (市值)

    使用因子打分法进行综合排序
    """

    def __init__(
        self,
        factors: Dict[str, float] = None,
        top_k: int = 20,
        rebalance_freq: int = 20  # 每月调仓
    ):
        super().__init__(name="MultiFactor")

        # 默认因子权重
        self.factors = factors or {
            'momentum': 0.3,      # 动量因子
            'value': 0.3,         # 价值因子 (1/PE)
            'quality': 0.2,       # 质量因子 (ROE)
            'size': 0.2           # 规模因子 (小市值偏好)
        }

        self.top_k = top_k
        self.rebalance_freq = rebalance_freq

        self._bar_count = 0
        self._factor_data: Dict[str, Dict[str, float]] = {}

    def calculate_factor_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子得分

        Args:
            data: 包含各因子原始值的DataFrame

        Returns:
            标准化后的因子得分
        """
        scores = pd.DataFrame(index=data.index)

        # 动量因子：过去20日收益率
        if 'close' in data.columns:
            momentum = data['close'].pct_change(20)
            scores['momentum'] = self._zscore(momentum)

        # 价值因子：1/PE (PE越低越好)
        if 'pe_ttm' in data.columns:
            # 处理负PE和极端值
            pe = data['pe_ttm'].clip(lower=0)
            pe = pe.replace(0, np.nan)
            scores['value'] = self._zscore(1 / pe)

        # 质量因子：ROE (如果有的话)
        if 'roe' in data.columns:
            scores['quality'] = self._zscore(data['roe'])

        # 规模因子：市值的负数 (小市值偏好)
        if 'total_mv' in data.columns:
            scores['size'] = self._zscore(-np.log(data['total_mv']))

        return scores

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Z-Score标准化"""
        return (series - series.mean()) / series.std()

    def calculate_composite_score(self, factor_scores: pd.DataFrame) -> pd.Series:
        """计算综合得分"""
        composite = pd.Series(0, index=factor_scores.index)

        for factor, weight in self.factors.items():
            if factor in factor_scores.columns:
                composite += factor_scores[factor].fillna(0) * weight

        return composite

    def select_stocks(self, data: pd.DataFrame) -> List[str]:
        """根据综合得分选股"""
        factor_scores = self.calculate_factor_scores(data)
        composite_scores = self.calculate_composite_score(factor_scores)

        # 选取得分最高的K只股票
        top_stocks = composite_scores.nlargest(self.top_k).index.tolist()

        return top_stocks

    def on_bar(self, bar: BarData):
        """处理K线数据"""
        # 更新因子数据
        ts_code = bar.symbol
        self._factor_data[ts_code] = {
            'close': bar.close,
            # 其他因子数据需要从engine获取
        }

        self._bar_count += 1

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        向量化信号生成

        每个调仓日重新计算因子得分并选股
        """
        close = data['close']
        n_days = len(close)

        # 计算动量因子
        momentum = close.pct_change(20)
        momentum_rank = momentum.rank(axis=1, pct=True)

        # 综合因子得分（简化版：仅使用动量）
        composite_rank = momentum_rank

        # 生成调仓日标记
        rebalance_days = pd.Series(False, index=close.index)
        rebalance_days.iloc[::self.rebalance_freq] = True

        # 入场信号：调仓日且进入前K
        top_k_mask = composite_rank.rank(axis=1, ascending=False) <= self.top_k

        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)

        # 调仓日更新信号
        for i, is_rebalance in enumerate(rebalance_days):
            if is_rebalance and i > 0:
                prev_in_portfolio = top_k_mask.iloc[i-1] if i > 0 else pd.Series(False, index=close.columns)
                curr_in_portfolio = top_k_mask.iloc[i]

                # 新进入的股票
                entries.iloc[i] = curr_in_portfolio & ~prev_in_portfolio
                # 退出的股票
                exits.iloc[i] = ~curr_in_portfolio & prev_in_portfolio

        return entries, exits
```

---

## 9. 完整使用示例

```python
"""
DuckDB 回测引擎使用示例
"""

from backtest_engine import (
    BacktestEngine, DataLoader, DualMAStrategy,
    MomentumStrategy, MultiFactorStrategy
)

def main():
    # 1. 初始化数据加载器
    data_loader = DataLoader(
        db_path='/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db',
        cache_size_mb=2048
    )

    # 2. 创建回测引擎
    engine = BacktestEngine(
        data_loader=data_loader,
        initial_capital=1_000_000.0,
        commission_rate=0.0003,  # 万三
        slippage=0.001,          # 0.1%
        stamp_duty=0.001         # 印花税
    )

    # 3. 创建策略
    # 示例1：双均线策略
    strategy = DualMAStrategy(
        short_period=5,
        long_period=20,
        position_size=0.2
    )

    # 示例2：动量策略
    # strategy = MomentumStrategy(
    #     lookback_period=20,
    #     top_k=10,
    #     rebalance_freq=5
    # )

    # 示例3：多因子策略
    # strategy = MultiFactorStrategy(
    #     factors={'momentum': 0.4, 'value': 0.3, 'size': 0.3},
    #     top_k=20,
    #     rebalance_freq=20
    # )

    engine.set_strategy(strategy)

    # 4. 定义股票池
    universe = data_loader.load_universe(
        date='20200101',
        filters={
            'market': ['主板'],
            'min_list_days': 252,
            'exclude_st': True
        }
    )
    print(f"股票池大小: {len(universe)}")

    # 5. 运行回测
    result = engine.run(
        start_date='20200101',
        end_date='20231231',
        universe=universe[:100],  # 限制股票数量以加快回测
        benchmark='000001.SH',
        mode='event'
    )

    # 6. 输出结果
    print(result.summary())

    # 7. 绘制图表
    fig = result.plot()
    fig.savefig('backtest_result.png', dpi=150)

    # 8. 导出详细报告
    result.nav_history.to_csv('nav_history.csv')
    result.trade_history.to_csv('trade_history.csv')

    # 9. 月度收益
    print("\n月度收益率:")
    print(result.monthly_returns)

    # 清理
    data_loader.close()


if __name__ == '__main__':
    main()
```

---

## 10. 性能优化建议

### 10.1 数据加载优化

```python
# 1. 使用 Parquet 格式缓存常用数据
def export_to_parquet(conn, table_name, output_path):
    """将表导出为Parquet格式"""
    conn.execute(f"""
    COPY (SELECT * FROM {table_name})
    TO '{output_path}' (FORMAT PARQUET)
    """)

# 2. 使用分区表加速查询
def create_partitioned_view(conn):
    """创建按年份分区的视图"""
    conn.execute("""
    CREATE OR REPLACE VIEW daily_partitioned AS
    SELECT *,
           EXTRACT(YEAR FROM CAST(trade_date AS DATE)) as year
    FROM daily
    """)

# 3. 创建索引
def create_indexes(conn):
    """创建查询索引"""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_ts_code ON daily(ts_code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_trade_date ON daily(trade_date)")
```

### 10.2 内存优化

```python
# 1. 使用 Arrow 格式减少内存拷贝
def load_as_arrow(conn, query):
    """使用Arrow格式加载数据"""
    return conn.execute(query).arrow()

# 2. 分批处理大数据集
def process_in_batches(data_loader, ts_codes, batch_size=100):
    """分批处理股票数据"""
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i+batch_size]
        yield data_loader.load_stock_data(batch, start_date, end_date)

# 3. 使用生成器避免一次性加载全部数据
def bar_generator(data_loader, ts_codes, start_date, end_date):
    """K线数据生成器"""
    for ts_code in ts_codes:
        df = data_loader.load_stock_data(ts_code, start_date, end_date)
        for idx, row in df.iterrows():
            yield BarData(
                datetime=idx[0],
                symbol=ts_code,
                **row.to_dict()
            )
```

### 10.3 计算优化

```python
# 1. 使用 NumPy 向量化计算
def vectorized_ma(prices: np.ndarray, period: int) -> np.ndarray:
    """向量化移动平均计算"""
    return np.convolve(prices, np.ones(period)/period, mode='valid')

# 2. 使用 Numba JIT 编译
from numba import jit

@jit(nopython=True)
def fast_sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """快速计算夏普比率"""
    excess = returns - rf_rate / 252
    return np.mean(excess) / np.std(excess) * np.sqrt(252)

# 3. 使用并行计算
from concurrent.futures import ProcessPoolExecutor

def parallel_backtest(strategies, data, n_workers=4):
    """并行回测多个策略"""
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            lambda s: run_single_backtest(s, data),
            strategies
        ))
    return results
```

---

## 11. 扩展接口

### 11.1 自定义成交模型

```python
class CustomFillModel(FillModel):
    """自定义成交模型示例"""

    def __init__(self, config: Dict):
        self.config = config

    def get_fill_price(self, order: Order, bar: BarData) -> Tuple[Optional[float], int]:
        # 实现自定义逻辑
        pass
```

### 11.2 自定义风控模块

```python
class RiskManager:
    """风险管理模块"""

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_drawdown: float = 0.2,
        daily_loss_limit: float = 0.05
    ):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit

    def check_order(self, order: Order, portfolio: PositionManager) -> bool:
        """检查订单是否符合风控要求"""
        # 单只股票最大仓位检查
        if order.direction == OrderDirection.BUY:
            nav = portfolio.get_nav()
            order_value = order.volume * order.price
            if order_value / nav > self.max_position_size:
                return False

        return True

    def check_drawdown(self, portfolio: PositionManager) -> bool:
        """检查是否触发最大回撤"""
        nav_history = portfolio.get_nav_history()
        if len(nav_history) < 2:
            return True

        peak = nav_history.max()
        current = nav_history.iloc[-1]
        drawdown = (peak - current) / peak

        return drawdown <= self.max_drawdown
```

---

## 12. 总结

本设计文档描述了一个完整的基于 DuckDB 的回测引擎架构，主要特点包括：

1. **高效数据管理**: 利用 DuckDB 的高性能查询能力，支持批量数据加载和内存缓存
2. **灵活的事件引擎**: 支持事件驱动和向量化两种回测模式
3. **完整的交易模拟**: 包含订单管理、持仓跟踪、滑点和手续费模拟
4. **清晰的策略接口**: 提供 `on_bar`、`on_order_filled`、`generate_signals` 等标准接口
5. **全面的绩效分析**: 包括收益率、风险指标、Alpha/Beta 分析和绩效归因
6. **丰富的示例策略**: 提供双均线、动量、多因子等经典策略实现

该架构设计充分考虑了扩展性和性能优化，可作为实际量化交易系统的基础框架。
