# 量价与资金流特征工程设计文档 v2.0

> 基于 DuckDB 数据库设计的量价和资金流特征体系
>
> 数据库路径: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`

---

## 目录

1. [数据源概览](#1-数据源概览)
2. [量价特征设计](#2-量价特征设计)
3. [资金流特征设计](#3-资金流特征设计)
4. [Python实现代码](#4-python实现代码)
5. [特征使用建议](#5-特征使用建议)

---

## 1. 数据源概览

### 1.1 相关数据表

| 表名 | 描述 | 关键字段 |
|------|------|----------|
| `daily` | 日线行情 | ts_code, trade_date, open, high, low, close, vol, amount |
| `daily_basic` | 每日指标 | turnover_rate, turnover_rate_f, volume_ratio |
| `moneyflow` | 个股资金流向 | buy_sm/md/lg/elg_amount, sell_sm/md/lg/elg_amount, net_mf_amount |
| `moneyflow_dc` | 资金流向(东财) | net_amount, net_amount_rate, buy_elg/lg/md/sm_amount_rate |
| `stk_factor_pro` | 技术因子 | 261个技术指标因子 |

### 1.2 字段说明

**daily 表**
- `vol`: 成交量 (手)
- `amount`: 成交额 (千元)
- `pct_chg`: 涨跌幅 (%)

**daily_basic 表**
- `turnover_rate`: 换手率 (基于总股本)
- `turnover_rate_f`: 换手率 (基于自由流通股本)
- `volume_ratio`: 量比

**moneyflow 表**
- `buy_sm/md/lg/elg_amount`: 小单/中单/大单/特大单买入金额 (万元)
- `sell_sm/md/lg/elg_amount`: 小单/中单/大单/特大单卖出金额 (万元)
- `net_mf_amount`: 净流入金额 (万元)

---

## 2. 量价特征设计

### 2.1 量比特征 (Volume Ratio)

#### 2.1.1 基础量比

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `volume_ratio` | 当日量比 | 直接取 `daily_basic.volume_ratio` |
| `volume_ratio_5d` | 5日平均量比 | `SMA(volume_ratio, 5)` |
| `volume_ratio_zscore` | 量比Z-Score | `(VR - MA(VR,20)) / STD(VR,20)` |

#### 2.1.2 量比区间分类

| 特征名 | 定义 | 分类规则 |
|--------|------|----------|
| `vr_level` | 量比等级 | 0: VR<0.5(极度萎缩), 1: 0.5≤VR<0.8(萎缩), 2: 0.8≤VR<1.5(正常), 3: 1.5≤VR<2.5(温和放量), 4: 2.5≤VR<5(明显放量), 5: VR≥5(异常放量) |

---

### 2.2 换手率特征 (Turnover Rate)

#### 2.2.1 基础换手率

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `turnover_rate` | 总股本换手率 | 直接取 `daily_basic.turnover_rate` |
| `turnover_rate_f` | 自由流通换手率 | 直接取 `daily_basic.turnover_rate_f` |
| `turnover_rate_5d` | 5日累计换手率 | `SUM(turnover_rate, 5)` |
| `turnover_rate_20d` | 20日累计换手率 | `SUM(turnover_rate, 20)` |

#### 2.2.2 换手率变化

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `turnover_chg_1d` | 换手率日变化 | `turnover_rate / turnover_rate.shift(1) - 1` |
| `turnover_chg_5d` | 换手率5日变化率 | `turnover_rate / MA(turnover_rate, 5) - 1` |
| `turnover_percentile_60d` | 60日换手率分位数 | `RANK(turnover_rate, 60) / 60` |

#### 2.2.3 换手率等级

| 特征名 | 定义 | 分类规则 |
|--------|------|----------|
| `turnover_level` | 换手率等级 | 0: <1%(冷淡), 1: 1-3%(低迷), 2: 3-7%(温和), 3: 7-15%(活跃), 4: 15-25%(高度活跃), 5: >25%(极度活跃) |

---

### 2.3 成交量突变特征 (Volume Surge)

#### 2.3.1 成交量倍数

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `vol_ma5_ratio` | 成交量/5日均量 | `vol / MA(vol, 5)` |
| `vol_ma10_ratio` | 成交量/10日均量 | `vol / MA(vol, 10)` |
| `vol_ma20_ratio` | 成交量/20日均量 | `vol / MA(vol, 20)` |
| `vol_ma60_ratio` | 成交量/60日均量 | `vol / MA(vol, 60)` |

#### 2.3.2 成交量突变信号

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `vol_surge_signal` | 成交量突破信号 | `1 if vol > 2*MA(vol,5) else 0` |
| `vol_shrink_signal` | 成交量萎缩信号 | `1 if vol < 0.5*MA(vol,5) else 0` |
| `vol_breakout_20d` | 创20日新高 | `1 if vol == MAX(vol, 20) else 0` |

#### 2.3.3 成交量标准差

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `vol_zscore_20d` | 成交量20日Z-Score | `(vol - MA(vol,20)) / STD(vol,20)` |
| `vol_volatility_20d` | 成交量20日波动率 | `STD(vol,20) / MA(vol,20)` |

---

### 2.4 量价背离特征 (Price-Volume Divergence)

#### 2.4.1 量价关系

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `pv_corr_5d` | 5日量价相关性 | `CORR(close, vol, 5)` |
| `pv_corr_10d` | 10日量价相关性 | `CORR(close, vol, 10)` |
| `pv_corr_20d` | 20日量价相关性 | `CORR(close, vol, 20)` |

#### 2.4.2 量价背离信号

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `pv_divergence_up` | 量增价跌背离 | `1 if (pct_chg < 0 and vol_ma5_ratio > 1.5) else 0` |
| `pv_divergence_down` | 量缩价涨背离 | `1 if (pct_chg > 0 and vol_ma5_ratio < 0.7) else 0` |
| `pv_divergence_score` | 量价背离评分 | `SIGN(pct_chg) * (1 - vol_ma5_ratio)` |

#### 2.4.3 趋势背离

| 特征名 | 定义 | 说明 |
|--------|------|------|
| `vol_trend_5d` | 成交量5日趋势 | 5日成交量线性回归斜率 |
| `price_trend_5d` | 价格5日趋势 | 5日价格线性回归斜率 |
| `trend_divergence` | 趋势背离 | `SIGN(vol_trend) != SIGN(price_trend)` |

---

### 2.5 连续放量/缩量特征 (Consecutive Volume Patterns)

#### 2.5.1 连续计数

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `consecutive_surge_days` | 连续放量天数 | 连续 `vol > MA(vol,5)` 的天数 |
| `consecutive_shrink_days` | 连续缩量天数 | 连续 `vol < MA(vol,5)` 的天数 |
| `vol_increase_streak` | 成交量连涨天数 | 连续 `vol > vol.shift(1)` 的天数 |
| `vol_decrease_streak` | 成交量连跌天数 | 连续 `vol < vol.shift(1)` 的天数 |

#### 2.5.2 累计效应

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `cum_vol_ratio_3d` | 3日累计成交量/前3日 | `SUM(vol,3) / SUM(vol.shift(3),3)` |
| `cum_vol_ratio_5d` | 5日累计成交量/前5日 | `SUM(vol,5) / SUM(vol.shift(5),5)` |

#### 2.5.3 放量/缩量模式

| 特征名 | 定义 | 分类规则 |
|--------|------|----------|
| `vol_pattern` | 成交量模式 | -2: 持续缩量, -1: 温和缩量, 0: 正常, 1: 温和放量, 2: 持续放量 |

---

## 3. 资金流特征设计

### 3.1 主力净流入特征

#### 3.1.1 基础资金流

| 特征名 | 数据源 | 定义 | 公式 |
|--------|--------|------|------|
| `net_mf_amount` | moneyflow | 总净流入额 | 直接取原始字段 (万元) |
| `net_amount` | moneyflow_dc | 总净流入额(东财) | 直接取原始字段 (万元) |
| `main_net_inflow` | moneyflow | 主力净流入 | `(buy_lg_amount + buy_elg_amount) - (sell_lg_amount + sell_elg_amount)` |
| `retail_net_inflow` | moneyflow | 散户净流入 | `(buy_sm_amount + buy_md_amount) - (sell_sm_amount + sell_md_amount)` |

#### 3.1.2 主力净流入率

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `main_inflow_rate` | 主力净流入占比 | `main_net_inflow / total_amount * 100` |
| `retail_inflow_rate` | 散户净流入占比 | `retail_net_inflow / total_amount * 100` |
| `net_amount_rate` | 总净流入率 | 直接取 `moneyflow_dc.net_amount_rate` |

#### 3.1.3 资金流累计

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `main_inflow_5d` | 5日主力累计净流入 | `SUM(main_net_inflow, 5)` |
| `main_inflow_10d` | 10日主力累计净流入 | `SUM(main_net_inflow, 10)` |
| `main_inflow_20d` | 20日主力累计净流入 | `SUM(main_net_inflow, 20)` |

---

### 3.2 大中小单分布特征

#### 3.2.1 各类资金占比

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `elg_buy_ratio` | 特大单买入占比 | `buy_elg_amount / total_buy_amount` |
| `lg_buy_ratio` | 大单买入占比 | `buy_lg_amount / total_buy_amount` |
| `md_buy_ratio` | 中单买入占比 | `buy_md_amount / total_buy_amount` |
| `sm_buy_ratio` | 小单买入占比 | `buy_sm_amount / total_buy_amount` |

#### 3.2.2 资金结构指标

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `main_ratio` | 主力资金占比 | `(buy_elg + buy_lg + sell_elg + sell_lg) / total` |
| `retail_ratio` | 散户资金占比 | `(buy_sm + buy_md + sell_sm + sell_md) / total` |
| `main_retail_ratio` | 主散比 | `main_amount / retail_amount` |

#### 3.2.3 东财资金分布(直接使用)

| 特征名 | 数据源 | 定义 |
|--------|--------|------|
| `buy_elg_amount_rate` | moneyflow_dc | 特大单买入占比 (%) |
| `buy_lg_amount_rate` | moneyflow_dc | 大单买入占比 (%) |
| `buy_md_amount_rate` | moneyflow_dc | 中单买入占比 (%) |
| `buy_sm_amount_rate` | moneyflow_dc | 小单买入占比 (%) |

---

### 3.3 资金流动量特征

#### 3.3.1 资金流动量

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `mf_momentum_5d` | 5日资金流动量 | `SUM(net_mf_amount, 5)` |
| `mf_momentum_10d` | 10日资金流动量 | `SUM(net_mf_amount, 10)` |
| `mf_momentum_diff` | 资金流动量差 | `mf_momentum_5d - mf_momentum_5d.shift(5)` |

#### 3.3.2 资金流强度

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `mf_intensity` | 资金流强度 | `abs(net_mf_amount) / total_amount` |
| `mf_intensity_ma5` | 5日平均资金流强度 | `MA(mf_intensity, 5)` |
| `mf_direction_ratio_5d` | 5日资金流向一致性 | `SUM(SIGN(net_mf_amount),5) / 5` |

#### 3.3.3 资金流趋势

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `mf_trend_5d` | 5日资金流趋势 | 5日净流入线性回归斜率 |
| `mf_acceleration` | 资金流加速度 | `net_mf_amount - net_mf_amount.shift(1)` |
| `positive_mf_days_5d` | 5日净流入天数 | `COUNT(net_mf_amount > 0, 5)` |

---

### 3.4 资金流与价格关系

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `mf_price_corr_5d` | 5日资金流与价格相关性 | `CORR(net_mf_amount, close, 5)` |
| `mf_price_corr_10d` | 10日资金流与价格相关性 | `CORR(net_mf_amount, close, 10)` |
| `mf_price_divergence` | 资金价格背离 | `1 if (mf>0 and pct_chg<0) or (mf<0 and pct_chg>0) else 0` |

---

### 3.5 资金流综合评分

| 特征名 | 定义 | 公式 |
|--------|------|------|
| `mf_score` | 资金流综合评分 | 综合主力净流入率、资金方向一致性、趋势等多因子 |
| `mf_strength_level` | 资金流强度等级 | 0-5分级：极弱、弱、中性、较强、强、极强 |

---

## 4. Python实现代码

### 4.1 数据加载模块

```python
"""
量价与资金流特征工程
基于 DuckDB 数据库实现
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


class FeatureDatabase:
    """DuckDB 数据库连接管理"""

    def __init__(self, db_path: str = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """连接数据库"""
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        return self.conn

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_daily_data(
    conn: duckdb.DuckDBPyConnection,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    加载日线数据，合并 daily 和 daily_basic

    Parameters
    ----------
    conn : duckdb连接
    ts_code : 股票代码，如 '000001.SZ'
    start_date : 开始日期，如 '20240101'
    end_date : 结束日期

    Returns
    -------
    DataFrame with columns: ts_code, trade_date, open, high, low, close,
                           vol, amount, pct_chg, turnover_rate, volume_ratio, etc.
    """
    sql = """
    SELECT
        d.ts_code,
        d.trade_date,
        d.open,
        d.high,
        d.low,
        d.close,
        d.pre_close,
        d.pct_chg,
        d.vol,
        d.amount,
        db.turnover_rate,
        db.turnover_rate_f,
        db.volume_ratio,
        db.circ_mv
    FROM daily d
    LEFT JOIN daily_basic db
        ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
    WHERE 1=1
    """

    if ts_code:
        sql += f" AND d.ts_code = '{ts_code}'"
    if start_date:
        sql += f" AND d.trade_date >= '{start_date}'"
    if end_date:
        sql += f" AND d.trade_date <= '{end_date}'"

    sql += " ORDER BY d.ts_code, d.trade_date"

    return conn.execute(sql).fetchdf()


def load_moneyflow_data(
    conn: duckdb.DuckDBPyConnection,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    加载资金流数据，合并 moneyflow 和 moneyflow_dc

    Returns
    -------
    DataFrame with all money flow columns
    """
    sql = """
    SELECT
        mf.ts_code,
        mf.trade_date,
        mf.buy_sm_amount,
        mf.sell_sm_amount,
        mf.buy_md_amount,
        mf.sell_md_amount,
        mf.buy_lg_amount,
        mf.sell_lg_amount,
        mf.buy_elg_amount,
        mf.sell_elg_amount,
        mf.net_mf_amount,
        dc.net_amount,
        dc.net_amount_rate,
        dc.buy_elg_amount_rate,
        dc.buy_lg_amount_rate,
        dc.buy_md_amount_rate,
        dc.buy_sm_amount_rate
    FROM moneyflow mf
    LEFT JOIN moneyflow_dc dc
        ON mf.ts_code = dc.ts_code AND mf.trade_date = dc.trade_date
    WHERE 1=1
    """

    if ts_code:
        sql += f" AND mf.ts_code = '{ts_code}'"
    if start_date:
        sql += f" AND mf.trade_date >= '{start_date}'"
    if end_date:
        sql += f" AND mf.trade_date <= '{end_date}'"

    sql += " ORDER BY mf.ts_code, mf.trade_date"

    return conn.execute(sql).fetchdf()
```

### 4.2 量价特征计算模块

```python
class VolumeFeatureCalculator:
    """量价特征计算器"""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : DataFrame with columns: ts_code, trade_date, vol, amount,
             pct_chg, close, turnover_rate, volume_ratio
        """
        self.df = df.copy()
        self.df = self.df.sort_values(['ts_code', 'trade_date'])

    def calculate_all(self) -> pd.DataFrame:
        """计算所有量价特征"""
        self._calc_volume_ratio_features()
        self._calc_turnover_features()
        self._calc_volume_surge_features()
        self._calc_price_volume_divergence()
        self._calc_consecutive_volume_patterns()
        return self.df

    def _calc_volume_ratio_features(self):
        """计算量比特征"""
        g = self.df.groupby('ts_code')

        # 基础量比 (已有)
        # volume_ratio 直接来自数据库

        # 5日平均量比
        self.df['volume_ratio_5d'] = g['volume_ratio'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        # 量比Z-Score (20日)
        vr_ma20 = g['volume_ratio'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        vr_std20 = g['volume_ratio'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        self.df['volume_ratio_zscore'] = (self.df['volume_ratio'] - vr_ma20) / vr_std20.replace(0, np.nan)

        # 量比等级
        self.df['vr_level'] = pd.cut(
            self.df['volume_ratio'],
            bins=[-np.inf, 0.5, 0.8, 1.5, 2.5, 5, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)

    def _calc_turnover_features(self):
        """计算换手率特征"""
        g = self.df.groupby('ts_code')

        # 累计换手率
        self.df['turnover_rate_5d'] = g['turnover_rate'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        self.df['turnover_rate_20d'] = g['turnover_rate'].transform(
            lambda x: x.rolling(20, min_periods=1).sum()
        )

        # 换手率变化
        self.df['turnover_chg_1d'] = g['turnover_rate'].transform(
            lambda x: x.pct_change()
        )

        tr_ma5 = g['turnover_rate'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        self.df['turnover_chg_5d'] = self.df['turnover_rate'] / tr_ma5 - 1

        # 60日换手率分位数
        self.df['turnover_percentile_60d'] = g['turnover_rate'].transform(
            lambda x: x.rolling(60, min_periods=20).apply(
                lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False
            )
        )

        # 换手率等级
        self.df['turnover_level'] = pd.cut(
            self.df['turnover_rate'],
            bins=[-np.inf, 1, 3, 7, 15, 25, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)

    def _calc_volume_surge_features(self):
        """计算成交量突变特征"""
        g = self.df.groupby('ts_code')

        # 成交量均线
        for period in [5, 10, 20, 60]:
            vol_ma = g['vol'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
            self.df[f'vol_ma{period}_ratio'] = self.df['vol'] / vol_ma.replace(0, np.nan)

        # 成交量突变信号
        vol_ma5 = g['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        self.df['vol_surge_signal'] = (self.df['vol'] > 2 * vol_ma5).astype(int)
        self.df['vol_shrink_signal'] = (self.df['vol'] < 0.5 * vol_ma5).astype(int)

        # 创20日新高
        vol_max20 = g['vol'].transform(lambda x: x.rolling(20, min_periods=1).max())
        self.df['vol_breakout_20d'] = (self.df['vol'] >= vol_max20).astype(int)

        # 成交量Z-Score
        vol_ma20 = g['vol'].transform(lambda x: x.rolling(20, min_periods=5).mean())
        vol_std20 = g['vol'].transform(lambda x: x.rolling(20, min_periods=5).std())
        self.df['vol_zscore_20d'] = (self.df['vol'] - vol_ma20) / vol_std20.replace(0, np.nan)

        # 成交量波动率
        self.df['vol_volatility_20d'] = vol_std20 / vol_ma20.replace(0, np.nan)

    def _calc_price_volume_divergence(self):
        """计算量价背离特征"""
        g = self.df.groupby('ts_code')

        # 量价相关性
        for period in [5, 10, 20]:
            self.df[f'pv_corr_{period}d'] = g.apply(
                lambda x: x['close'].rolling(period, min_periods=3).corr(x['vol'])
            ).reset_index(level=0, drop=True)

        # 量价背离信号
        vol_ma5_ratio = self.df['vol_ma5_ratio']
        pct_chg = self.df['pct_chg']

        self.df['pv_divergence_up'] = (
            (pct_chg < 0) & (vol_ma5_ratio > 1.5)
        ).astype(int)

        self.df['pv_divergence_down'] = (
            (pct_chg > 0) & (vol_ma5_ratio < 0.7)
        ).astype(int)

        # 量价背离评分
        self.df['pv_divergence_score'] = np.sign(pct_chg) * (1 - vol_ma5_ratio)

        # 趋势背离 (使用线性回归斜率)
        def calc_slope(series):
            if len(series) < 3:
                return np.nan
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series.values, 1)
            return slope

        self.df['vol_trend_5d'] = g['vol'].transform(
            lambda x: x.rolling(5, min_periods=3).apply(calc_slope, raw=False)
        )
        self.df['price_trend_5d'] = g['close'].transform(
            lambda x: x.rolling(5, min_periods=3).apply(calc_slope, raw=False)
        )
        self.df['trend_divergence'] = (
            np.sign(self.df['vol_trend_5d']) != np.sign(self.df['price_trend_5d'])
        ).astype(int)

    def _calc_consecutive_volume_patterns(self):
        """计算连续放量/缩量特征"""
        g = self.df.groupby('ts_code')

        # 相对于5日均量的放量/缩量
        vol_ma5 = g['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        is_above_ma5 = (self.df['vol'] > vol_ma5).astype(int)
        is_below_ma5 = (self.df['vol'] < vol_ma5).astype(int)

        # 连续放量/缩量天数
        def count_consecutive(series):
            """计算连续相同值的天数"""
            result = []
            count = 0
            prev = None
            for val in series:
                if val == 1:
                    count += 1
                else:
                    count = 0
                result.append(count)
            return result

        self.df['consecutive_surge_days'] = g.apply(
            lambda x: pd.Series(count_consecutive(
                (x['vol'] > x['vol'].rolling(5, min_periods=1).mean()).astype(int)
            ), index=x.index)
        ).reset_index(level=0, drop=True)

        self.df['consecutive_shrink_days'] = g.apply(
            lambda x: pd.Series(count_consecutive(
                (x['vol'] < x['vol'].rolling(5, min_periods=1).mean()).astype(int)
            ), index=x.index)
        ).reset_index(level=0, drop=True)

        # 成交量连涨/连跌
        self.df['vol_increase_streak'] = g.apply(
            lambda x: pd.Series(count_consecutive(
                (x['vol'] > x['vol'].shift(1)).astype(int)
            ), index=x.index)
        ).reset_index(level=0, drop=True)

        self.df['vol_decrease_streak'] = g.apply(
            lambda x: pd.Series(count_consecutive(
                (x['vol'] < x['vol'].shift(1)).astype(int)
            ), index=x.index)
        ).reset_index(level=0, drop=True)

        # 累计成交量比
        vol_sum_3d = g['vol'].transform(lambda x: x.rolling(3, min_periods=1).sum())
        vol_sum_3d_prev = g['vol'].transform(
            lambda x: x.shift(3).rolling(3, min_periods=1).sum()
        )
        self.df['cum_vol_ratio_3d'] = vol_sum_3d / vol_sum_3d_prev.replace(0, np.nan)

        vol_sum_5d = g['vol'].transform(lambda x: x.rolling(5, min_periods=1).sum())
        vol_sum_5d_prev = g['vol'].transform(
            lambda x: x.shift(5).rolling(5, min_periods=1).sum()
        )
        self.df['cum_vol_ratio_5d'] = vol_sum_5d / vol_sum_5d_prev.replace(0, np.nan)

        # 成交量模式分类
        def classify_vol_pattern(row):
            if row['consecutive_surge_days'] >= 3:
                return 2  # 持续放量
            elif row['consecutive_surge_days'] >= 1:
                return 1  # 温和放量
            elif row['consecutive_shrink_days'] >= 3:
                return -2  # 持续缩量
            elif row['consecutive_shrink_days'] >= 1:
                return -1  # 温和缩量
            else:
                return 0  # 正常

        self.df['vol_pattern'] = self.df.apply(classify_vol_pattern, axis=1)
```

### 4.3 资金流特征计算模块

```python
class MoneyFlowFeatureCalculator:
    """资金流特征计算器"""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : DataFrame from load_moneyflow_data()
        """
        self.df = df.copy()
        self.df = self.df.sort_values(['ts_code', 'trade_date'])
        self._calc_base_amounts()

    def _calc_base_amounts(self):
        """计算基础金额字段"""
        # 总买入/卖出
        self.df['total_buy_amount'] = (
            self.df['buy_sm_amount'].fillna(0) +
            self.df['buy_md_amount'].fillna(0) +
            self.df['buy_lg_amount'].fillna(0) +
            self.df['buy_elg_amount'].fillna(0)
        )
        self.df['total_sell_amount'] = (
            self.df['sell_sm_amount'].fillna(0) +
            self.df['sell_md_amount'].fillna(0) +
            self.df['sell_lg_amount'].fillna(0) +
            self.df['sell_elg_amount'].fillna(0)
        )
        self.df['total_amount'] = self.df['total_buy_amount'] + self.df['total_sell_amount']

        # 主力/散户金额
        self.df['main_buy_amount'] = (
            self.df['buy_lg_amount'].fillna(0) +
            self.df['buy_elg_amount'].fillna(0)
        )
        self.df['main_sell_amount'] = (
            self.df['sell_lg_amount'].fillna(0) +
            self.df['sell_elg_amount'].fillna(0)
        )
        self.df['retail_buy_amount'] = (
            self.df['buy_sm_amount'].fillna(0) +
            self.df['buy_md_amount'].fillna(0)
        )
        self.df['retail_sell_amount'] = (
            self.df['sell_sm_amount'].fillna(0) +
            self.df['sell_md_amount'].fillna(0)
        )

    def calculate_all(self) -> pd.DataFrame:
        """计算所有资金流特征"""
        self._calc_net_inflow_features()
        self._calc_order_distribution()
        self._calc_money_flow_momentum()
        self._calc_money_flow_score()
        return self.df

    def _calc_net_inflow_features(self):
        """计算主力净流入特征"""
        g = self.df.groupby('ts_code')

        # 主力/散户净流入
        self.df['main_net_inflow'] = self.df['main_buy_amount'] - self.df['main_sell_amount']
        self.df['retail_net_inflow'] = self.df['retail_buy_amount'] - self.df['retail_sell_amount']

        # 净流入率
        self.df['main_inflow_rate'] = (
            self.df['main_net_inflow'] / self.df['total_amount'].replace(0, np.nan) * 100
        )
        self.df['retail_inflow_rate'] = (
            self.df['retail_net_inflow'] / self.df['total_amount'].replace(0, np.nan) * 100
        )

        # 累计净流入
        for period in [5, 10, 20]:
            self.df[f'main_inflow_{period}d'] = g['main_net_inflow'].transform(
                lambda x: x.rolling(period, min_periods=1).sum()
            )
            self.df[f'net_mf_{period}d'] = g['net_mf_amount'].transform(
                lambda x: x.rolling(period, min_periods=1).sum()
            )

    def _calc_order_distribution(self):
        """计算大中小单分布特征"""
        # 各类资金买入占比
        for order_type in ['elg', 'lg', 'md', 'sm']:
            buy_col = f'buy_{order_type}_amount'
            self.df[f'{order_type}_buy_ratio'] = (
                self.df[buy_col].fillna(0) /
                self.df['total_buy_amount'].replace(0, np.nan)
            )

        # 主力/散户资金占比
        main_total = self.df['main_buy_amount'] + self.df['main_sell_amount']
        retail_total = self.df['retail_buy_amount'] + self.df['retail_sell_amount']

        self.df['main_ratio'] = main_total / self.df['total_amount'].replace(0, np.nan)
        self.df['retail_ratio'] = retail_total / self.df['total_amount'].replace(0, np.nan)

        # 主散比
        self.df['main_retail_ratio'] = (
            main_total / retail_total.replace(0, np.nan)
        )

    def _calc_money_flow_momentum(self):
        """计算资金流动量特征"""
        g = self.df.groupby('ts_code')

        # 资金流动量
        self.df['mf_momentum_5d'] = g['net_mf_amount'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        self.df['mf_momentum_10d'] = g['net_mf_amount'].transform(
            lambda x: x.rolling(10, min_periods=1).sum()
        )
        self.df['mf_momentum_diff'] = (
            self.df['mf_momentum_5d'] -
            g['mf_momentum_5d'].transform(lambda x: x.shift(5))
        )

        # 资金流强度
        self.df['mf_intensity'] = (
            self.df['net_mf_amount'].abs() /
            self.df['total_amount'].replace(0, np.nan)
        )
        self.df['mf_intensity_ma5'] = g['mf_intensity'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        # 资金流方向一致性 (5日内净流入的天数比例)
        self.df['mf_direction'] = np.sign(self.df['net_mf_amount'])
        self.df['mf_direction_ratio_5d'] = g['mf_direction'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        # 资金流趋势
        def calc_slope(series):
            if len(series) < 3 or series.isna().all():
                return np.nan
            valid = series.dropna()
            if len(valid) < 3:
                return np.nan
            x = np.arange(len(valid))
            slope, _ = np.polyfit(x, valid.values, 1)
            return slope

        self.df['mf_trend_5d'] = g['net_mf_amount'].transform(
            lambda x: x.rolling(5, min_periods=3).apply(calc_slope, raw=False)
        )

        # 资金流加速度
        self.df['mf_acceleration'] = g['net_mf_amount'].transform(lambda x: x.diff())

        # 净流入天数
        self.df['positive_mf_days_5d'] = g.apply(
            lambda x: (x['net_mf_amount'] > 0).rolling(5, min_periods=1).sum()
        ).reset_index(level=0, drop=True)

    def _calc_money_flow_score(self):
        """计算资金流综合评分"""
        # 归一化各因子
        def normalize(series, lower=0, upper=100):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return 50
            return (series - min_val) / (max_val - min_val) * (upper - lower) + lower

        g = self.df.groupby('ts_code')

        # 综合评分 (0-100)
        # 考虑因子：主力净流入率、方向一致性、趋势
        score_inflow = g['main_inflow_rate'].transform(
            lambda x: normalize(x.clip(-20, 20))
        )
        score_direction = (self.df['mf_direction_ratio_5d'] + 1) / 2 * 100
        score_trend = g['mf_trend_5d'].transform(
            lambda x: normalize(x.clip(-1e6, 1e6))
        )

        self.df['mf_score'] = (
            score_inflow * 0.4 +
            score_direction * 0.3 +
            score_trend * 0.3
        )

        # 资金流强度等级
        self.df['mf_strength_level'] = pd.cut(
            self.df['mf_score'],
            bins=[0, 20, 35, 50, 65, 80, 100],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)
```

### 4.4 特征合并与导出

```python
class FeatureEngineer:
    """特征工程主类"""

    def __init__(self, db_path: str = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"):
        self.db_path = db_path

    def compute_features(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算完整的量价和资金流特征

        Parameters
        ----------
        ts_code : 股票代码
        start_date : 开始日期
        end_date : 结束日期

        Returns
        -------
        DataFrame with all features
        """
        with FeatureDatabase(self.db_path) as conn:
            # 加载数据
            daily_df = load_daily_data(conn, ts_code, start_date, end_date)
            mf_df = load_moneyflow_data(conn, ts_code, start_date, end_date)

        # 计算量价特征
        vol_calc = VolumeFeatureCalculator(daily_df)
        vol_features = vol_calc.calculate_all()

        # 计算资金流特征
        mf_calc = MoneyFlowFeatureCalculator(mf_df)
        mf_features = mf_calc.calculate_all()

        # 合并特征
        result = vol_features.merge(
            mf_features,
            on=['ts_code', 'trade_date'],
            how='left'
        )

        return result

    def get_feature_list(self) -> dict:
        """获取特征列表"""
        return {
            'volume_ratio_features': [
                'volume_ratio', 'volume_ratio_5d', 'volume_ratio_zscore', 'vr_level'
            ],
            'turnover_features': [
                'turnover_rate', 'turnover_rate_f', 'turnover_rate_5d',
                'turnover_rate_20d', 'turnover_chg_1d', 'turnover_chg_5d',
                'turnover_percentile_60d', 'turnover_level'
            ],
            'volume_surge_features': [
                'vol_ma5_ratio', 'vol_ma10_ratio', 'vol_ma20_ratio', 'vol_ma60_ratio',
                'vol_surge_signal', 'vol_shrink_signal', 'vol_breakout_20d',
                'vol_zscore_20d', 'vol_volatility_20d'
            ],
            'divergence_features': [
                'pv_corr_5d', 'pv_corr_10d', 'pv_corr_20d',
                'pv_divergence_up', 'pv_divergence_down', 'pv_divergence_score',
                'vol_trend_5d', 'price_trend_5d', 'trend_divergence'
            ],
            'consecutive_features': [
                'consecutive_surge_days', 'consecutive_shrink_days',
                'vol_increase_streak', 'vol_decrease_streak',
                'cum_vol_ratio_3d', 'cum_vol_ratio_5d', 'vol_pattern'
            ],
            'net_inflow_features': [
                'main_net_inflow', 'retail_net_inflow',
                'main_inflow_rate', 'retail_inflow_rate',
                'main_inflow_5d', 'main_inflow_10d', 'main_inflow_20d'
            ],
            'order_distribution_features': [
                'elg_buy_ratio', 'lg_buy_ratio', 'md_buy_ratio', 'sm_buy_ratio',
                'main_ratio', 'retail_ratio', 'main_retail_ratio'
            ],
            'mf_momentum_features': [
                'mf_momentum_5d', 'mf_momentum_10d', 'mf_momentum_diff',
                'mf_intensity', 'mf_intensity_ma5', 'mf_direction_ratio_5d',
                'mf_trend_5d', 'mf_acceleration', 'positive_mf_days_5d'
            ],
            'mf_score_features': [
                'mf_score', 'mf_strength_level'
            ]
        }


# 使用示例
if __name__ == "__main__":
    # 初始化特征工程器
    fe = FeatureEngineer()

    # 计算单只股票特征
    features = fe.compute_features(
        ts_code='000001.SZ',
        start_date='20240101',
        end_date='20260130'
    )

    print(f"特征数据形状: {features.shape}")
    print(f"\n特征列表:")
    for category, cols in fe.get_feature_list().items():
        print(f"  {category}: {len(cols)} features")

    # 查看部分特征
    print(f"\n最近5日数据预览:")
    display_cols = [
        'ts_code', 'trade_date', 'close', 'pct_chg', 'vol',
        'volume_ratio', 'vol_ma5_ratio', 'vol_pattern',
        'main_inflow_rate', 'mf_score'
    ]
    print(features[display_cols].tail())
```

---

## 5. 特征使用建议

### 5.1 特征分类与适用场景

| 特征类别 | 适用场景 | 说明 |
|----------|----------|------|
| 量比特征 | 短期交易信号 | 适合识别异常放量、交易活跃度变化 |
| 换手率特征 | 市场情绪分析 | 高换手率可能预示趋势反转 |
| 成交量突变 | 突破确认 | 放量突破更有效，缩量突破需警惕 |
| 量价背离 | 趋势反转预警 | 量增价跌可能预示底部，量缩价涨可能预示顶部 |
| 连续量能 | 趋势持续性判断 | 持续放量/缩量反映资金态度 |
| 主力净流入 | 主力动向跟踪 | 关注主力资金的持续性流向 |
| 资金分布 | 市场结构分析 | 主散比可判断筹码集中度 |
| 资金动量 | 趋势强度评估 | 结合价格趋势判断多空力量 |

### 5.2 特征组合建议

**趋势确认组合**
```python
# 上涨趋势确认
bullish_confirmation = (
    (df['pct_chg'] > 0) &
    (df['vol_ma5_ratio'] > 1.2) &
    (df['main_inflow_rate'] > 0) &
    (df['mf_direction_ratio_5d'] > 0.6)
)
```

**量价背离预警组合**
```python
# 顶部预警
top_warning = (
    (df['pv_divergence_down'] == 1) &
    (df['main_inflow_rate'] < -5) &
    (df['mf_trend_5d'] < 0)
)

# 底部预警
bottom_warning = (
    (df['pv_divergence_up'] == 1) &
    (df['main_inflow_rate'] > 5) &
    (df['consecutive_shrink_days'] == 0)
)
```

### 5.3 注意事项

1. **数据质量**: 资金流数据可能存在缺失，需做好空值处理
2. **市场差异**: 不同市值股票的量价特征表现不同，建议分组分析
3. **时效性**: 短周期特征更适合日内/短线，长周期特征适合中线
4. **因子正交**: 使用时注意特征相关性，避免多重共线性
5. **回测验证**: 所有特征需经过历史回测验证有效性

### 5.4 扩展方向

1. **行业相对强弱**: 计算个股相对行业的量价特征
2. **市场整体情绪**: 汇总个股资金流构建市场情绪指标
3. **因子动态权重**: 根据市场状态动态调整特征权重
4. **机器学习特征**: 将离散特征进行编码，用于模型训练

---

## 附录：特征完整列表

### A. 量价特征 (30+)

| 序号 | 特征名 | 类型 | 描述 |
|------|--------|------|------|
| 1 | volume_ratio | float | 当日量比 |
| 2 | volume_ratio_5d | float | 5日平均量比 |
| 3 | volume_ratio_zscore | float | 量比Z-Score |
| 4 | vr_level | int | 量比等级(0-5) |
| 5 | turnover_rate | float | 总股本换手率 |
| 6 | turnover_rate_f | float | 自由流通换手率 |
| 7 | turnover_rate_5d | float | 5日累计换手率 |
| 8 | turnover_rate_20d | float | 20日累计换手率 |
| 9 | turnover_chg_1d | float | 换手率日变化 |
| 10 | turnover_chg_5d | float | 换手率5日变化 |
| 11 | turnover_percentile_60d | float | 60日换手率分位 |
| 12 | turnover_level | int | 换手率等级(0-5) |
| 13 | vol_ma5_ratio | float | 成交量/5日均量 |
| 14 | vol_ma10_ratio | float | 成交量/10日均量 |
| 15 | vol_ma20_ratio | float | 成交量/20日均量 |
| 16 | vol_ma60_ratio | float | 成交量/60日均量 |
| 17 | vol_surge_signal | int | 放量信号(0/1) |
| 18 | vol_shrink_signal | int | 缩量信号(0/1) |
| 19 | vol_breakout_20d | int | 20日量能新高(0/1) |
| 20 | vol_zscore_20d | float | 成交量Z-Score |
| 21 | vol_volatility_20d | float | 成交量波动率 |
| 22 | pv_corr_5d | float | 5日量价相关性 |
| 23 | pv_corr_10d | float | 10日量价相关性 |
| 24 | pv_corr_20d | float | 20日量价相关性 |
| 25 | pv_divergence_up | int | 量增价跌背离 |
| 26 | pv_divergence_down | int | 量缩价涨背离 |
| 27 | pv_divergence_score | float | 量价背离评分 |
| 28 | vol_trend_5d | float | 成交量5日趋势 |
| 29 | price_trend_5d | float | 价格5日趋势 |
| 30 | trend_divergence | int | 趋势背离(0/1) |
| 31 | consecutive_surge_days | int | 连续放量天数 |
| 32 | consecutive_shrink_days | int | 连续缩量天数 |
| 33 | vol_increase_streak | int | 成交量连涨天数 |
| 34 | vol_decrease_streak | int | 成交量连跌天数 |
| 35 | cum_vol_ratio_3d | float | 3日累计成交量比 |
| 36 | cum_vol_ratio_5d | float | 5日累计成交量比 |
| 37 | vol_pattern | int | 成交量模式(-2~2) |

### B. 资金流特征 (25+)

| 序号 | 特征名 | 类型 | 描述 |
|------|--------|------|------|
| 1 | main_net_inflow | float | 主力净流入(万元) |
| 2 | retail_net_inflow | float | 散户净流入(万元) |
| 3 | main_inflow_rate | float | 主力净流入率(%) |
| 4 | retail_inflow_rate | float | 散户净流入率(%) |
| 5 | main_inflow_5d | float | 5日主力累计净流入 |
| 6 | main_inflow_10d | float | 10日主力累计净流入 |
| 7 | main_inflow_20d | float | 20日主力累计净流入 |
| 8 | net_mf_5d | float | 5日总净流入 |
| 9 | net_mf_10d | float | 10日总净流入 |
| 10 | elg_buy_ratio | float | 特大单买入占比 |
| 11 | lg_buy_ratio | float | 大单买入占比 |
| 12 | md_buy_ratio | float | 中单买入占比 |
| 13 | sm_buy_ratio | float | 小单买入占比 |
| 14 | main_ratio | float | 主力资金占比 |
| 15 | retail_ratio | float | 散户资金占比 |
| 16 | main_retail_ratio | float | 主散比 |
| 17 | mf_momentum_5d | float | 5日资金流动量 |
| 18 | mf_momentum_10d | float | 10日资金流动量 |
| 19 | mf_momentum_diff | float | 资金流动量差 |
| 20 | mf_intensity | float | 资金流强度 |
| 21 | mf_intensity_ma5 | float | 5日平均资金流强度 |
| 22 | mf_direction_ratio_5d | float | 5日资金流向一致性 |
| 23 | mf_trend_5d | float | 5日资金流趋势 |
| 24 | mf_acceleration | float | 资金流加速度 |
| 25 | positive_mf_days_5d | int | 5日净流入天数 |
| 26 | mf_score | float | 资金流综合评分 |
| 27 | mf_strength_level | int | 资金流强度等级(0-5) |

---

*文档生成时间: 2026-01-31*
*版本: v2.0*
