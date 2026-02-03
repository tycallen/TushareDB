# 股票投资风险模型设计文档

## 概述

本文档详细描述了一套基于 Barra 风格的多因子风险模型设计方案，用于股票投资组合的风险管理。模型利用 Tushare 数据库中的日线行情、估值指标、行业分类和财务数据构建风险因子，并提供完整的风险度量和压力测试框架。

---

## 1. 风险因子模型 (类 Barra)

### 1.1 因子模型理论框架

股票收益率可分解为:

$$r_i = \alpha_i + \sum_{k=1}^{K} \beta_{ik} f_k + \epsilon_i$$

其中:
- $r_i$: 股票 $i$ 的收益率
- $\alpha_i$: 股票特质收益
- $\beta_{ik}$: 股票 $i$ 对因子 $k$ 的暴露
- $f_k$: 因子 $k$ 的收益率
- $\epsilon_i$: 特质风险

### 1.2 市场因子

#### Beta 因子

**数据来源:**
- `daily`: 个股日线行情 (`pct_chg`)
- `sw_daily`: 申万行业指数作为市场基准

**计算方法:**
```python
# 使用滚动窗口回归计算 Beta
# 窗口期: 252 交易日 (1年)
# 最小观测值: 126 交易日

def calculate_beta(stock_returns, market_returns, window=252):
    """
    计算个股相对市场的 Beta 值

    Beta = Cov(r_stock, r_market) / Var(r_market)
    """
    rolling_cov = stock_returns.rolling(window).cov(market_returns)
    rolling_var = market_returns.rolling(window).var()
    beta = rolling_cov / rolling_var
    return beta
```

**数据表连接:**
```sql
-- 获取个股收益率
SELECT ts_code, trade_date, pct_chg / 100.0 as ret
FROM daily
WHERE trade_date >= '20230101'

-- 获取市场收益率 (使用沪深300或全市场等权)
SELECT trade_date, AVG(pct_chg) / 100.0 as market_ret
FROM daily
WHERE trade_date >= '20230101'
GROUP BY trade_date
```

---

### 1.3 行业因子

#### 申万一级行业暴露

**数据来源:**
- `index_member_all`: 申万行业分类 (`l1_code`, `l1_name`, `ts_code`, `in_date`, `out_date`)

**行业列表 (31个申万一级行业):**

| 行业代码 | 行业名称 | 行业代码 | 行业名称 |
|---------|---------|---------|---------|
| 801010.SI | 农林牧渔 | 801720.SI | 建筑装饰 |
| 801030.SI | 基础化工 | 801730.SI | 电力设备 |
| 801040.SI | 钢铁 | 801740.SI | 国防军工 |
| 801050.SI | 有色金属 | 801750.SI | 计算机 |
| 801080.SI | 电子 | 801760.SI | 传媒 |
| 801110.SI | 家用电器 | 801770.SI | 通信 |
| 801120.SI | 食品饮料 | 801780.SI | 银行 |
| 801130.SI | 纺织服饰 | 801790.SI | 非银金融 |
| 801140.SI | 轻工制造 | 801880.SI | 汽车 |
| 801150.SI | 医药生物 | 801890.SI | 机械设备 |
| 801160.SI | 公用事业 | 801950.SI | 煤炭 |
| 801170.SI | 交通运输 | 801960.SI | 石油石化 |
| 801180.SI | 房地产 | 801970.SI | 环保 |
| 801200.SI | 商贸零售 | 801980.SI | 美容护理 |
| 801210.SI | 社会服务 | 801230.SI | 综合 |
| 801710.SI | 建筑材料 | | |

**计算方法:**
```python
def get_industry_exposure(ts_code, trade_date):
    """
    获取个股行业暴露 (哑变量编码)
    返回长度为 31 的向量，对应行业为 1，其余为 0
    """
    # 查询当前有效的行业分类
    query = """
    SELECT l1_code, l1_name
    FROM index_member_all
    WHERE ts_code = ?
      AND in_date <= ?
      AND (out_date IS NULL OR out_date > ?)
    """
    # 转换为哑变量矩阵
    industry_dummies = pd.get_dummies(industry_series)
    return industry_dummies
```

**数据表查询:**
```sql
-- 获取个股当前行业分类
SELECT ts_code, l1_code, l1_name
FROM index_member_all
WHERE in_date <= '20240101'
  AND (out_date IS NULL OR out_date > '20240101')
```

---

### 1.4 风格因子

#### 1.4.1 Size (市值因子)

**数据来源:**
- `daily_basic`: 市值数据 (`total_mv`, `circ_mv`)

**计算方法:**
```python
def calculate_size_factor(total_mv):
    """
    Size = ln(总市值)
    标准化处理: Z-score 或 MAD 标准化
    """
    ln_size = np.log(total_mv)
    # Z-score 标准化
    size_zscore = (ln_size - ln_size.mean()) / ln_size.std()
    return size_zscore
```

**数据表查询:**
```sql
SELECT ts_code, trade_date,
       LOG(total_mv) as ln_total_mv,
       LOG(circ_mv) as ln_circ_mv
FROM daily_basic
WHERE trade_date = '20240101'
```

---

#### 1.4.2 Value (估值因子)

**数据来源:**
- `daily_basic`: 估值指标 (`pe_ttm`, `pb`, `ps_ttm`, `dv_ttm`)

**子因子构成:**

| 子因子 | 计算公式 | 数据字段 |
|-------|---------|---------|
| E/P (盈利收益率) | 1 / PE_TTM | `pe_ttm` |
| B/P (账面市值比) | 1 / PB | `pb` |
| S/P (营收市值比) | 1 / PS_TTM | `ps_ttm` |
| D/P (股息率) | DV_TTM | `dv_ttm` |

**计算方法:**
```python
def calculate_value_factor(daily_basic_df):
    """
    Value 因子 = 加权平均(E/P, B/P, S/P, D/P)
    权重: [0.35, 0.35, 0.15, 0.15]
    """
    # 计算各子因子
    ep = 1 / daily_basic_df['pe_ttm']  # 注意处理负值和极值
    bp = 1 / daily_basic_df['pb']
    sp = 1 / daily_basic_df['ps_ttm']
    dp = daily_basic_df['dv_ttm'] / 100

    # 异常值处理 (Winsorize at 1% and 99%)
    ep = winsorize(ep, limits=[0.01, 0.01])
    bp = winsorize(bp, limits=[0.01, 0.01])
    sp = winsorize(sp, limits=[0.01, 0.01])
    dp = winsorize(dp, limits=[0.01, 0.01])

    # 标准化
    ep_z = zscore(ep)
    bp_z = zscore(bp)
    sp_z = zscore(sp)
    dp_z = zscore(dp)

    # 加权合成
    value = 0.35 * ep_z + 0.35 * bp_z + 0.15 * sp_z + 0.15 * dp_z
    return value
```

**数据表查询:**
```sql
SELECT ts_code, trade_date,
       1.0 / NULLIF(pe_ttm, 0) as ep,
       1.0 / NULLIF(pb, 0) as bp,
       1.0 / NULLIF(ps_ttm, 0) as sp,
       dv_ttm / 100.0 as dp
FROM daily_basic
WHERE trade_date = '20240101'
  AND pe_ttm > 0 AND pb > 0 AND ps_ttm > 0
```

---

#### 1.4.3 Momentum (动量因子)

**数据来源:**
- `daily`: 日线收益率 (`pct_chg`)

**子因子构成:**

| 子因子 | 计算公式 | 回看期 |
|-------|---------|-------|
| 短期动量 | 过去 21 天累计收益 | 21 天 |
| 中期动量 | 过去 126 天累计收益 (剔除最近 21 天) | 126 天 |
| 长期动量 | 过去 252 天累计收益 (剔除最近 21 天) | 252 天 |

**计算方法:**
```python
def calculate_momentum_factor(returns_df, trade_date):
    """
    Momentum = 加权平均(短期动量, 中期动量, 长期动量)
    权重: [0.2, 0.5, 0.3]
    """
    # 短期动量: 过去 21 天
    short_mom = returns_df.rolling(21).apply(lambda x: (1 + x).prod() - 1)

    # 中期动量: 过去 126 天，剔除最近 21 天
    mid_returns = returns_df.shift(21).rolling(105).apply(lambda x: (1 + x).prod() - 1)

    # 长期动量: 过去 252 天，剔除最近 21 天
    long_returns = returns_df.shift(21).rolling(231).apply(lambda x: (1 + x).prod() - 1)

    # 标准化和加权合成
    momentum = 0.2 * zscore(short_mom) + 0.5 * zscore(mid_returns) + 0.3 * zscore(long_returns)
    return momentum
```

**数据表查询:**
```sql
-- 计算累计收益率
WITH daily_returns AS (
    SELECT ts_code, trade_date, pct_chg / 100.0 as ret
    FROM daily
    WHERE trade_date BETWEEN '20230101' AND '20240101'
)
SELECT ts_code,
       EXP(SUM(LN(1 + ret)) OVER (
           PARTITION BY ts_code
           ORDER BY trade_date
           ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
       )) - 1 as mom_252d
FROM daily_returns
```

---

#### 1.4.4 Volatility (波动率因子)

**数据来源:**
- `daily`: 日线收益率 (`pct_chg`)

**子因子构成:**

| 子因子 | 计算公式 | 说明 |
|-------|---------|------|
| 历史波动率 | 过去 252 天收益率标准差 | 年化处理 |
| CAPM 残差波动率 | 回归残差的标准差 | 特质波动率 |
| 波动率变化 | 短期波动率 - 长期波动率 | 波动率动量 |

**计算方法:**
```python
def calculate_volatility_factor(returns_df, market_returns):
    """
    Volatility = 加权平均(历史波动率, 特质波动率, 波动率变化)
    """
    # 历史波动率 (年化)
    hist_vol = returns_df.rolling(252).std() * np.sqrt(252)

    # 特质波动率 (CAPM 残差)
    def calc_idio_vol(stock_ret, market_ret, window=252):
        model = LinearRegression()
        model.fit(market_ret.values.reshape(-1, 1), stock_ret.values)
        residuals = stock_ret - model.predict(market_ret.values.reshape(-1, 1))
        return residuals.std() * np.sqrt(252)

    idio_vol = calc_idio_vol(returns_df, market_returns)

    # 波动率变化
    short_vol = returns_df.rolling(21).std() * np.sqrt(252)
    long_vol = returns_df.rolling(126).std() * np.sqrt(252)
    vol_change = short_vol - long_vol

    # 标准化和加权
    volatility = 0.5 * zscore(hist_vol) + 0.3 * zscore(idio_vol) + 0.2 * zscore(vol_change)
    return volatility
```

**数据表查询:**
```sql
-- 计算滚动波动率
SELECT ts_code, trade_date,
       STDDEV(pct_chg / 100.0) OVER (
           PARTITION BY ts_code
           ORDER BY trade_date
           ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
       ) * SQRT(252) as volatility_252d
FROM daily
WHERE trade_date BETWEEN '20230101' AND '20240101'
```

---

#### 1.4.5 Liquidity (流动性因子)

**数据来源:**
- `daily`: 成交量、成交额 (`vol`, `amount`)
- `daily_basic`: 换手率 (`turnover_rate`, `turnover_rate_f`)

**子因子构成:**

| 子因子 | 计算公式 | 说明 |
|-------|---------|------|
| 换手率 | 平均换手率 (过去 21 天) | 流动性代理 |
| 成交额/市值 | amount / total_mv | Amihud 改进 |
| 换手率波动 | 换手率的标准差 | 流动性稳定性 |

**计算方法:**
```python
def calculate_liquidity_factor(daily_df, daily_basic_df):
    """
    Liquidity = 加权平均(换手率, 成交额/市值, 换手率波动)
    """
    # 平均换手率 (过去 21 天)
    avg_turnover = daily_basic_df['turnover_rate_f'].rolling(21).mean()

    # 成交额 / 市值
    amount_mv_ratio = daily_df['amount'] / daily_basic_df['total_mv']
    avg_amount_mv = amount_mv_ratio.rolling(21).mean()

    # 换手率波动
    turnover_vol = daily_basic_df['turnover_rate_f'].rolling(21).std()

    # 标准化和加权
    liquidity = 0.4 * zscore(avg_turnover) + 0.4 * zscore(avg_amount_mv) + 0.2 * zscore(-turnover_vol)
    return liquidity
```

**数据表查询:**
```sql
SELECT d.ts_code, d.trade_date,
       AVG(db.turnover_rate_f) OVER (
           PARTITION BY d.ts_code
           ORDER BY d.trade_date
           ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
       ) as avg_turnover_21d,
       AVG(d.amount / db.total_mv) OVER (
           PARTITION BY d.ts_code
           ORDER BY d.trade_date
           ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
       ) as avg_amount_mv_21d
FROM daily d
JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
WHERE d.trade_date BETWEEN '20231201' AND '20240101'
```

---

#### 1.4.6 Growth (成长性因子)

**数据来源:**
- `fina_indicator_vip`: 财务指标 (`netprofit_yoy`, `or_yoy`, `assets_yoy`, `roe`)

**子因子构成:**

| 子因子 | 字段 | 说明 |
|-------|------|------|
| 净利润同比增长 | `netprofit_yoy` | 盈利增长 |
| 营收同比增长 | `or_yoy` | 收入增长 |
| 总资产同比增长 | `assets_yoy` | 规模扩张 |
| ROE 同比变化 | `roe_yoy` | 盈利能力改善 |

**计算方法:**
```python
def calculate_growth_factor(fina_df):
    """
    Growth = 加权平均(净利润增长, 营收增长, 资产增长, ROE变化)
    权重: [0.35, 0.35, 0.15, 0.15]
    """
    # 获取最新财报数据
    latest_fina = fina_df.sort_values('end_date').groupby('ts_code').last()

    # 异常值处理
    netprofit_yoy = winsorize(latest_fina['netprofit_yoy'], limits=[0.01, 0.01])
    or_yoy = winsorize(latest_fina['or_yoy'], limits=[0.01, 0.01])
    assets_yoy = winsorize(latest_fina['assets_yoy'], limits=[0.01, 0.01])
    roe_yoy = winsorize(latest_fina['roe_yoy'], limits=[0.01, 0.01])

    # 标准化和加权
    growth = (0.35 * zscore(netprofit_yoy) +
              0.35 * zscore(or_yoy) +
              0.15 * zscore(assets_yoy) +
              0.15 * zscore(roe_yoy))
    return growth
```

**数据表查询:**
```sql
-- 获取最新财务指标
SELECT ts_code, end_date,
       netprofit_yoy,
       or_yoy,
       assets_yoy,
       roe_yoy
FROM fina_indicator_vip
WHERE end_date = (
    SELECT MAX(end_date)
    FROM fina_indicator_vip f2
    WHERE f2.ts_code = fina_indicator_vip.ts_code
)
```

---

#### 1.4.7 Leverage (杠杆因子)

**数据来源:**
- `fina_indicator_vip`: 杠杆指标 (`debt_to_assets`, `debt_to_eqt`, `current_ratio`)
- `balancesheet`: 资产负债表 (`total_liab`, `total_assets`)

**子因子构成:**

| 子因子 | 字段/计算 | 说明 |
|-------|---------|------|
| 资产负债率 | `debt_to_assets` | 总体杠杆 |
| 负债权益比 | `debt_to_eqt` | 权益杠杆 |
| 流动比率 (负) | `-current_ratio` | 短期偿债能力 (取负) |

**计算方法:**
```python
def calculate_leverage_factor(fina_df):
    """
    Leverage = 加权平均(资产负债率, 负债权益比, -流动比率)
    权重: [0.4, 0.4, 0.2]
    """
    # 获取最新财报数据
    latest_fina = fina_df.sort_values('end_date').groupby('ts_code').last()

    # 异常值处理
    debt_to_assets = winsorize(latest_fina['debt_to_assets'], limits=[0.01, 0.01])
    debt_to_eqt = winsorize(latest_fina['debt_to_eqt'], limits=[0.01, 0.01])
    current_ratio = winsorize(latest_fina['current_ratio'], limits=[0.01, 0.01])

    # 标准化 (流动比率取负，高杠杆对应高值)
    leverage = (0.4 * zscore(debt_to_assets) +
                0.4 * zscore(debt_to_eqt) +
                0.2 * zscore(-current_ratio))
    return leverage
```

**数据表查询:**
```sql
SELECT ts_code, end_date,
       debt_to_assets,
       debt_to_eqt,
       current_ratio
FROM fina_indicator_vip
WHERE end_date >= '20230101'
ORDER BY ts_code, end_date DESC
```

---

### 1.5 因子暴露矩阵构建

**完整因子列表:**

| 类别 | 因子名称 | 数量 |
|-----|---------|-----|
| 市场因子 | Beta | 1 |
| 行业因子 | 申万一级行业 | 31 |
| 风格因子 | Size, Value, Momentum, Volatility, Liquidity, Growth, Leverage | 7 |
| **总计** | | **39** |

**因子暴露矩阵 X:**

$$X_{n \times k} = \begin{bmatrix} \beta_1 & I_{11} & ... & I_{1,31} & Size_1 & Value_1 & ... & Leverage_1 \\ \beta_2 & I_{21} & ... & I_{2,31} & Size_2 & Value_2 & ... & Leverage_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\ \beta_n & I_{n1} & ... & I_{n,31} & Size_n & Value_n & ... & Leverage_n \end{bmatrix}$$

其中 $n$ 为股票数量，$k=39$ 为因子数量。

---

## 2. 协方差矩阵估计

### 2.1 样本协方差矩阵

**计算方法:**
```python
def sample_covariance(returns_df, window=252):
    """
    计算样本协方差矩阵

    参数:
    - returns_df: 收益率矩阵 (T x N)
    - window: 估计窗口

    返回:
    - 协方差矩阵 (N x N)
    """
    return returns_df.tail(window).cov()
```

**特点:**
- 优点: 无偏估计，计算简单
- 缺点: 当 N > T 时奇异，估计误差大

---

### 2.2 收缩估计 (Ledoit-Wolf)

**理论框架:**

$$\Sigma_{shrunk} = \alpha \cdot F + (1-\alpha) \cdot S$$

其中:
- $S$: 样本协方差矩阵
- $F$: 结构化目标矩阵 (通常为对角矩阵)
- $\alpha$: 最优收缩系数

**计算方法:**
```python
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(returns_df, window=252):
    """
    Ledoit-Wolf 收缩估计

    自动计算最优收缩强度
    """
    returns_array = returns_df.tail(window).values
    lw = LedoitWolf().fit(returns_array)
    return pd.DataFrame(
        lw.covariance_,
        index=returns_df.columns,
        columns=returns_df.columns
    )
```

**最优收缩系数计算:**
```python
def optimal_shrinkage_intensity(S, F, returns):
    """
    计算 Ledoit-Wolf 最优收缩强度

    alpha* = argmin E[||alpha*F + (1-alpha)*S - Sigma||^2]
    """
    T, N = returns.shape

    # 计算 pi (估计误差的平方和)
    pi = sum([sum([(returns[:,i] * returns[:,j] - S[i,j])**2
                   for j in range(N)]) for i in range(N)]) / T

    # 计算 rho (收缩偏差)
    rho = sum([(S[i,i] - F[i,i])**2 for i in range(N)])

    # 最优收缩强度
    alpha = max(0, min(1, (pi - rho) / T))
    return alpha
```

---

### 2.3 因子协方差 + 特质风险

**结构化协方差矩阵:**

$$\Sigma = X F X^T + D$$

其中:
- $X$: 因子暴露矩阵 (N x K)
- $F$: 因子协方差矩阵 (K x K)
- $D$: 特质风险对角矩阵 (N x N)

**计算步骤:**

```python
def factor_covariance_model(returns_df, factor_returns_df, exposures_df):
    """
    因子协方差模型

    步骤:
    1. 估计因子收益率
    2. 计算因子协方差矩阵
    3. 计算特质风险
    4. 合成总协方差矩阵
    """
    # 1. 因子收益率 (截面回归)
    factor_returns = estimate_factor_returns(returns_df, exposures_df)

    # 2. 因子协方差矩阵 (使用 Newey-West 调整)
    F = factor_covariance_newey_west(factor_returns, lags=5)

    # 3. 特质风险 (残差波动率)
    residuals = returns_df - (exposures_df @ factor_returns.T)
    D = np.diag(residuals.var())

    # 4. 合成协方差矩阵
    X = exposures_df.values
    Sigma = X @ F @ X.T + D

    return Sigma, F, D
```

**因子收益率估计 (截面回归):**
```python
def estimate_factor_returns(returns_df, exposures_df):
    """
    使用 WLS 回归估计因子收益率

    r_t = X_t * f_t + epsilon_t

    f_t = (X'WX)^{-1} X'W r_t

    权重 W 使用市值加权
    """
    factor_returns = []
    for date in returns_df.index:
        X = exposures_df.loc[date]
        r = returns_df.loc[date]
        w = market_cap_weights[date]  # 市值权重

        # WLS 回归
        W = np.diag(w)
        f = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ r
        factor_returns.append(f)

    return pd.DataFrame(factor_returns, index=returns_df.index)
```

**Newey-West 调整:**
```python
def factor_covariance_newey_west(factor_returns, lags=5):
    """
    Newey-West 协方差估计器 (处理自相关和异方差)

    F = Gamma_0 + sum_{j=1}^{L} (1 - j/(L+1)) * (Gamma_j + Gamma_j')
    """
    T = len(factor_returns)
    K = factor_returns.shape[1]

    # 去均值
    demeaned = factor_returns - factor_returns.mean()

    # Gamma_0
    Gamma_0 = (demeaned.T @ demeaned) / T

    # 加权求和滞后协方差
    F = Gamma_0.copy()
    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)
        Gamma_j = (demeaned.iloc[j:].T @ demeaned.iloc[:-j]) / T
        F += weight * (Gamma_j + Gamma_j.T)

    return F
```

---

## 3. 风险分解

### 3.1 系统性风险 vs 特质风险

**定义:**
- **系统性风险**: 由因子驱动的风险 $X F X^T$
- **特质风险**: 股票特有的风险 $D$

**投资组合风险分解:**

$$\sigma_p^2 = w^T \Sigma w = w^T X F X^T w + w^T D w$$

其中:
- $w^T X F X^T w$: 系统性风险贡献
- $w^T D w$: 特质风险贡献

```python
def decompose_portfolio_risk(weights, exposures, factor_cov, idio_var):
    """
    分解投资组合风险

    返回:
    - total_risk: 总风险
    - systematic_risk: 系统性风险
    - idiosyncratic_risk: 特质风险
    - risk_ratio: 系统性/特质风险比例
    """
    X = exposures.values
    F = factor_cov
    D = np.diag(idio_var)
    w = weights.values

    # 投资组合因子暴露
    portfolio_exposure = X.T @ w

    # 系统性风险 (方差)
    systematic_var = portfolio_exposure.T @ F @ portfolio_exposure

    # 特质风险 (方差)
    idio_var = w.T @ D @ w

    # 总风险
    total_var = systematic_var + idio_var

    return {
        'total_risk': np.sqrt(total_var),
        'systematic_risk': np.sqrt(systematic_var),
        'idiosyncratic_risk': np.sqrt(idio_var),
        'systematic_pct': systematic_var / total_var,
        'idiosyncratic_pct': idio_var / total_var
    }
```

---

### 3.2 因子贡献分析

**因子风险贡献:**

对于因子 $k$，其对组合风险的贡献:

$$RC_k = \frac{\partial \sigma_p}{\partial f_k} \cdot \beta_{p,k} = \frac{(\beta_{p,k})^2 \cdot \sigma_k^2 + \sum_{j \neq k} \beta_{p,k} \beta_{p,j} \sigma_{k,j}}{\sigma_p}$$

```python
def factor_risk_contribution(weights, exposures, factor_cov):
    """
    计算各因子的风险贡献

    返回:
    - factor_rc: 各因子风险贡献
    - factor_rc_pct: 各因子风险贡献比例
    """
    X = exposures.values
    F = factor_cov
    w = weights.values

    # 投资组合因子暴露
    beta_p = X.T @ w

    # 因子方差贡献
    factor_var_contrib = beta_p * (F @ beta_p)

    # 组合因子风险
    total_factor_var = beta_p.T @ F @ beta_p

    # 风险贡献
    factor_rc = factor_var_contrib / np.sqrt(total_factor_var)
    factor_rc_pct = factor_var_contrib / total_factor_var

    return pd.DataFrame({
        'exposure': beta_p,
        'risk_contribution': factor_rc,
        'risk_contribution_pct': factor_rc_pct
    }, index=factor_names)
```

---

### 3.3 边际风险贡献 (MRC)

**定义:**

边际风险贡献衡量增加单位持仓对组合风险的边际影响:

$$MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

**成分风险贡献 (CRC):**

$$CRC_i = w_i \cdot MRC_i = \frac{w_i (\Sigma w)_i}{\sigma_p}$$

注意: $\sum_i CRC_i = \sigma_p$

```python
def marginal_risk_contribution(weights, covariance):
    """
    计算边际风险贡献和成分风险贡献

    返回:
    - mrc: 边际风险贡献
    - crc: 成分风险贡献
    - crc_pct: 成分风险贡献比例
    """
    w = weights.values
    Sigma = covariance.values

    # 组合方差
    port_var = w.T @ Sigma @ w
    port_vol = np.sqrt(port_var)

    # 边际风险贡献
    mrc = (Sigma @ w) / port_vol

    # 成分风险贡献
    crc = w * mrc
    crc_pct = crc / port_vol

    return pd.DataFrame({
        'weight': w,
        'marginal_risk_contribution': mrc,
        'component_risk_contribution': crc,
        'component_risk_pct': crc_pct
    }, index=weights.index)
```

---

## 4. 风险指标计算

### 4.1 VaR (Value at Risk)

#### 4.1.1 历史模拟法

**方法:**
直接使用历史收益率分布的分位数

```python
def var_historical(returns, confidence_level=0.95, window=252):
    """
    历史模拟法 VaR

    参数:
    - returns: 历史收益率序列
    - confidence_level: 置信水平 (默认 95%)
    - window: 回看窗口

    返回:
    - VaR 值 (正数表示损失)
    """
    sorted_returns = returns.tail(window).sort_values()
    var_percentile = 1 - confidence_level
    var_index = int(len(sorted_returns) * var_percentile)
    var = -sorted_returns.iloc[var_index]
    return var
```

#### 4.1.2 参数法 (正态分布假设)

**方法:**
假设收益率服从正态分布

$$VaR_\alpha = -(\mu - z_\alpha \cdot \sigma)$$

```python
from scipy.stats import norm

def var_parametric(returns, confidence_level=0.95, window=252):
    """
    参数法 VaR (正态分布假设)

    参数:
    - returns: 历史收益率序列
    - confidence_level: 置信水平
    - window: 估计窗口
    """
    mu = returns.tail(window).mean()
    sigma = returns.tail(window).std()
    z_alpha = norm.ppf(1 - confidence_level)
    var = -(mu + z_alpha * sigma)
    return var
```

#### 4.1.3 投资组合 VaR

```python
def portfolio_var(weights, returns_df, confidence_level=0.95, method='parametric'):
    """
    投资组合 VaR

    参数:
    - weights: 投资组合权重
    - returns_df: 股票收益率矩阵
    - confidence_level: 置信水平
    - method: 'parametric' 或 'historical'
    """
    # 计算组合收益率
    portfolio_returns = (returns_df * weights).sum(axis=1)

    if method == 'parametric':
        return var_parametric(portfolio_returns, confidence_level)
    else:
        return var_historical(portfolio_returns, confidence_level)
```

---

### 4.2 CVaR / Expected Shortfall

**定义:**
条件 VaR，即超过 VaR 的预期损失

$$CVaR_\alpha = E[-R | R < -VaR_\alpha]$$

```python
def cvar(returns, confidence_level=0.95, window=252):
    """
    条件 VaR (Expected Shortfall)

    参数:
    - returns: 历史收益率序列
    - confidence_level: 置信水平
    - window: 回看窗口

    返回:
    - CVaR 值 (正数表示损失)
    """
    var = var_historical(returns, confidence_level, window)
    tail_losses = returns.tail(window)[returns.tail(window) < -var]
    cvar = -tail_losses.mean()
    return cvar

def cvar_parametric(returns, confidence_level=0.95, window=252):
    """
    参数法 CVaR (正态分布假设)
    """
    mu = returns.tail(window).mean()
    sigma = returns.tail(window).std()
    z_alpha = norm.ppf(1 - confidence_level)
    cvar = -mu + sigma * norm.pdf(z_alpha) / (1 - confidence_level)
    return cvar
```

---

### 4.3 最大回撤 (Maximum Drawdown)

**定义:**
从历史最高点到最低点的最大跌幅

$$MDD = \max_{t \in [0,T]} \left( \frac{HWM_t - V_t}{HWM_t} \right)$$

其中 $HWM_t$ 是到时刻 $t$ 的历史最高净值。

```python
def maximum_drawdown(returns):
    """
    计算最大回撤

    参数:
    - returns: 收益率序列

    返回:
    - max_drawdown: 最大回撤
    - start_date: 回撤起始日期
    - end_date: 回撤结束日期
    - recovery_date: 恢复日期 (如有)
    """
    # 计算累计净值
    cum_returns = (1 + returns).cumprod()

    # 历史最高点
    rolling_max = cum_returns.expanding().max()

    # 回撤序列
    drawdown = (cum_returns - rolling_max) / rolling_max

    # 最大回撤
    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = cum_returns[:end_date].idxmax()

    # 恢复日期
    recovery = cum_returns[end_date:][cum_returns[end_date:] >= cum_returns[start_date]]
    recovery_date = recovery.index[0] if len(recovery) > 0 else None

    return {
        'max_drawdown': -max_drawdown,
        'start_date': start_date,
        'end_date': end_date,
        'recovery_date': recovery_date,
        'duration': (end_date - start_date).days if hasattr(end_date - start_date, 'days') else None
    }
```

---

### 4.4 跟踪误差 (Tracking Error)

**定义:**
组合收益与基准收益之差的波动率

$$TE = \sqrt{Var(R_p - R_b)} = \sqrt{\frac{\sum_{t=1}^{T}(R_{p,t} - R_{b,t} - \bar{e})^2}{T-1}}$$

```python
def tracking_error(portfolio_returns, benchmark_returns, annualize=True):
    """
    计算跟踪误差

    参数:
    - portfolio_returns: 组合收益率
    - benchmark_returns: 基准收益率
    - annualize: 是否年化

    返回:
    - tracking_error: 跟踪误差
    """
    excess_returns = portfolio_returns - benchmark_returns
    te = excess_returns.std()

    if annualize:
        te = te * np.sqrt(252)

    return te

def information_ratio(portfolio_returns, benchmark_returns, annualize=True):
    """
    信息比率 = 超额收益 / 跟踪误差
    """
    excess_returns = portfolio_returns - benchmark_returns

    mean_excess = excess_returns.mean()
    te = excess_returns.std()

    if annualize:
        mean_excess = mean_excess * 252
        te = te * np.sqrt(252)

    ir = mean_excess / te if te > 0 else 0
    return ir
```

---

## 5. 压力测试

### 5.1 历史情景测试

**重要历史事件:**

| 事件 | 起始日期 | 结束日期 | 上证指数跌幅 |
|-----|---------|---------|------------|
| 2008 金融危机 | 2008-01-15 | 2008-11-04 | -72% |
| 2015 股灾 | 2015-06-12 | 2015-08-26 | -45% |
| 2018 贸易战 | 2018-01-29 | 2018-12-27 | -31% |
| 2020 新冠疫情 | 2020-01-14 | 2020-03-23 | -16% |
| 2022 市场调整 | 2021-12-13 | 2022-04-27 | -26% |

```python
def historical_scenario_test(weights, returns_df, scenario_dates):
    """
    历史情景测试

    参数:
    - weights: 当前投资组合权重
    - returns_df: 历史收益率数据
    - scenario_dates: 情景日期范围 {'start': 'YYYYMMDD', 'end': 'YYYYMMDD'}

    返回:
    - scenario_return: 情景期间组合收益
    - max_drawdown: 情景期间最大回撤
    """
    scenario_returns = returns_df.loc[scenario_dates['start']:scenario_dates['end']]
    portfolio_returns = (scenario_returns * weights).sum(axis=1)

    total_return = (1 + portfolio_returns).prod() - 1
    mdd = maximum_drawdown(portfolio_returns)

    return {
        'total_return': total_return,
        'max_drawdown': mdd['max_drawdown'],
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'worst_day': portfolio_returns.min(),
        'best_day': portfolio_returns.max()
    }

# 定义历史情景
HISTORICAL_SCENARIOS = {
    '2008_financial_crisis': {'start': '20080115', 'end': '20081104'},
    '2015_stock_crash': {'start': '20150612', 'end': '20150826'},
    '2018_trade_war': {'start': '20180129', 'end': '20181227'},
    '2020_covid': {'start': '20200114', 'end': '20200323'},
    '2022_adjustment': {'start': '20211213', 'end': '20220427'}
}
```

**数据表查询:**
```sql
-- 获取历史情景期间的收益率
SELECT ts_code, trade_date, pct_chg / 100.0 as ret
FROM daily
WHERE trade_date BETWEEN '20150612' AND '20150826'
ORDER BY ts_code, trade_date
```

---

### 5.2 假设情景测试

**设计假设情景:**

| 情景 | 市场跌幅 | 行业影响 | 风格因子冲击 |
|-----|---------|---------|-------------|
| 流动性紧缩 | -15% | 金融 -25% | 小市值 -20%, 高杠杆 -25% |
| 经济衰退 | -25% | 周期行业 -35% | 高Beta -30%, 高杠杆 -20% |
| 利率急升 | -10% | 房地产 -30%, 银行 -15% | 高杠杆 -25%, 成长 -20% |
| 地缘政治 | -20% | 科技 -30%, 国防 +10% | 动量反转, 大市值 +5% |
| 通胀失控 | -20% | 消费 -25% | 高估值 -30% |

```python
def hypothetical_scenario_test(weights, exposures, factor_shocks, market_shock):
    """
    假设情景测试

    参数:
    - weights: 组合权重
    - exposures: 因子暴露矩阵
    - factor_shocks: 各因子冲击幅度 (dict)
    - market_shock: 市场整体冲击

    返回:
    - scenario_loss: 情景损失
    - factor_contributions: 各因子损失贡献
    """
    # 组合因子暴露
    portfolio_exposure = exposures.T @ weights

    # 因子冲击向量
    shock_vector = np.array([factor_shocks.get(f, 0) for f in exposures.columns])

    # 因子损失贡献
    factor_loss = portfolio_exposure * shock_vector

    # 总损失 (因子损失 + 市场冲击)
    total_loss = factor_loss.sum() + market_shock

    return {
        'total_loss': total_loss,
        'market_contribution': market_shock,
        'factor_contributions': pd.Series(factor_loss, index=exposures.columns)
    }

# 定义假设情景
HYPOTHETICAL_SCENARIOS = {
    'liquidity_crisis': {
        'market_shock': -0.15,
        'factor_shocks': {
            'Beta': -0.10,
            'Size': -0.20,
            'Liquidity': -0.15,
            'Leverage': -0.25,
            '801780.SI': -0.25,  # 银行
            '801790.SI': -0.25   # 非银金融
        }
    },
    'economic_recession': {
        'market_shock': -0.25,
        'factor_shocks': {
            'Beta': -0.30,
            'Momentum': -0.15,
            'Leverage': -0.20,
            '801040.SI': -0.35,  # 钢铁
            '801050.SI': -0.35,  # 有色金属
            '801880.SI': -0.30   # 汽车
        }
    },
    'rate_spike': {
        'market_shock': -0.10,
        'factor_shocks': {
            'Leverage': -0.25,
            'Growth': -0.20,
            '801180.SI': -0.30,  # 房地产
            '801780.SI': -0.15   # 银行
        }
    }
}
```

---

### 5.3 极端事件模拟

**蒙特卡洛模拟:**

```python
def monte_carlo_stress_test(weights, covariance, n_simulations=10000, horizon=21):
    """
    蒙特卡洛压力测试

    参数:
    - weights: 组合权重
    - covariance: 协方差矩阵
    - n_simulations: 模拟次数
    - horizon: 时间范围 (交易日)

    返回:
    - var_95: 95% VaR
    - var_99: 99% VaR
    - expected_shortfall: 预期损失
    - worst_case: 最坏情况
    """
    # Cholesky 分解
    L = np.linalg.cholesky(covariance)

    # 生成随机收益
    random_returns = np.random.normal(0, 1, (n_simulations, len(weights), horizon))
    correlated_returns = np.einsum('ij,njk->nik', L, random_returns)

    # 计算组合收益
    portfolio_returns = np.einsum('i,nik->nk', weights, correlated_returns)
    cumulative_returns = (1 + portfolio_returns).prod(axis=1) - 1

    # 风险指标
    var_95 = np.percentile(cumulative_returns, 5)
    var_99 = np.percentile(cumulative_returns, 1)
    es_95 = cumulative_returns[cumulative_returns <= var_95].mean()

    return {
        'var_95': -var_95,
        'var_99': -var_99,
        'expected_shortfall_95': -es_95,
        'worst_case': -cumulative_returns.min(),
        'probability_loss_10pct': (cumulative_returns < -0.10).mean()
    }
```

**尾部风险分析 (极值理论):**

```python
from scipy.stats import genpareto

def extreme_value_analysis(returns, threshold_percentile=5):
    """
    极值理论分析 (GPD 拟合)

    参数:
    - returns: 历史收益率
    - threshold_percentile: 阈值百分位

    返回:
    - var_99: 99% VaR (EVT)
    - var_99_9: 99.9% VaR (EVT)
    - expected_shortfall: 预期损失
    """
    # 设定阈值
    threshold = np.percentile(returns, threshold_percentile)

    # 提取超过阈值的损失
    exceedances = threshold - returns[returns < threshold]

    # 拟合 GPD 分布
    shape, loc, scale = genpareto.fit(exceedances, floc=0)

    # 计算极端 VaR
    n = len(returns)
    n_exceed = len(exceedances)

    def gpd_var(p):
        """计算给定概率的 VaR"""
        if shape != 0:
            return threshold - scale / shape * (1 - ((n / n_exceed) * (1 - p)) ** (-shape))
        else:
            return threshold - scale * np.log((n / n_exceed) * (1 - p))

    return {
        'var_99': -gpd_var(0.99),
        'var_99_9': -gpd_var(0.999),
        'gpd_shape': shape,
        'gpd_scale': scale
    }
```

---

## 6. 实现架构

### 6.1 模块结构

```
risk_model/
├── __init__.py
├── factors/
│   ├── __init__.py
│   ├── market.py          # Beta 因子
│   ├── industry.py        # 行业因子
│   ├── style.py           # 风格因子
│   └── utils.py           # 因子工具函数
├── covariance/
│   ├── __init__.py
│   ├── sample.py          # 样本协方差
│   ├── shrinkage.py       # 收缩估计
│   └── factor_model.py    # 因子协方差模型
├── risk_metrics/
│   ├── __init__.py
│   ├── var.py             # VaR 计算
│   ├── cvar.py            # CVaR 计算
│   ├── drawdown.py        # 最大回撤
│   └── tracking.py        # 跟踪误差
├── decomposition/
│   ├── __init__.py
│   ├── systematic.py      # 系统性/特质风险分解
│   ├── factor_contrib.py  # 因子贡献
│   └── marginal.py        # 边际风险贡献
├── stress_test/
│   ├── __init__.py
│   ├── historical.py      # 历史情景
│   ├── hypothetical.py    # 假设情景
│   └── monte_carlo.py     # 蒙特卡洛模拟
└── data/
    ├── __init__.py
    └── loader.py          # 数据加载器
```

### 6.2 数据加载器

```python
import duckdb

class RiskDataLoader:
    """风险模型数据加载器"""

    def __init__(self, db_path='/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_daily_returns(self, start_date, end_date, ts_codes=None):
        """获取日收益率数据"""
        query = f"""
        SELECT ts_code, trade_date, pct_chg / 100.0 as ret
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """
        if ts_codes:
            query += f" AND ts_code IN ({','.join([f'\'{c}\'' for c in ts_codes])})"
        return self.conn.execute(query).df()

    def get_daily_basic(self, start_date, end_date, ts_codes=None):
        """获取估值指标数据"""
        query = f"""
        SELECT ts_code, trade_date,
               total_mv, circ_mv, pe_ttm, pb, ps_ttm, dv_ttm,
               turnover_rate, turnover_rate_f
        FROM daily_basic
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """
        if ts_codes:
            query += f" AND ts_code IN ({','.join([f'\'{c}\'' for c in ts_codes])})"
        return self.conn.execute(query).df()

    def get_industry_mapping(self, trade_date):
        """获取行业分类映射"""
        query = f"""
        SELECT ts_code, l1_code, l1_name
        FROM index_member_all
        WHERE in_date <= '{trade_date}'
          AND (out_date IS NULL OR out_date > '{trade_date}')
        """
        return self.conn.execute(query).df()

    def get_financial_indicators(self, end_date):
        """获取财务指标"""
        query = f"""
        SELECT ts_code, end_date,
               netprofit_yoy, or_yoy, assets_yoy, roe_yoy,
               debt_to_assets, debt_to_eqt, current_ratio
        FROM fina_indicator_vip
        WHERE end_date <= '{end_date}'
        """
        return self.conn.execute(query).df()

    def get_sw_index_returns(self, start_date, end_date):
        """获取申万行业指数收益率"""
        query = f"""
        SELECT ts_code, trade_date, pct_change / 100.0 as ret
        FROM sw_daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """
        return self.conn.execute(query).df()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
```

### 6.3 主入口类

```python
class RiskModel:
    """风险模型主类"""

    def __init__(self, db_path, lookback_window=252):
        self.data_loader = RiskDataLoader(db_path)
        self.lookback_window = lookback_window

        # 因子计算器
        self.market_factor = MarketFactorCalculator()
        self.industry_factor = IndustryFactorCalculator()
        self.style_factors = StyleFactorCalculator()

        # 协方差估计器
        self.cov_estimator = CovarianceEstimator()

        # 风险指标计算器
        self.risk_metrics = RiskMetricsCalculator()

        # 压力测试器
        self.stress_tester = StressTester()

    def calculate_factor_exposures(self, trade_date, ts_codes):
        """计算因子暴露矩阵"""
        exposures = {}

        # 市场因子
        exposures['Beta'] = self.market_factor.calculate_beta(
            self.data_loader, trade_date, ts_codes
        )

        # 行业因子
        industry_exp = self.industry_factor.get_exposures(
            self.data_loader, trade_date, ts_codes
        )
        exposures.update(industry_exp)

        # 风格因子
        style_exp = self.style_factors.calculate_all(
            self.data_loader, trade_date, ts_codes
        )
        exposures.update(style_exp)

        return pd.DataFrame(exposures)

    def estimate_covariance(self, trade_date, method='factor_model'):
        """估计协方差矩阵"""
        if method == 'sample':
            return self.cov_estimator.sample_covariance(...)
        elif method == 'ledoit_wolf':
            return self.cov_estimator.ledoit_wolf(...)
        elif method == 'factor_model':
            return self.cov_estimator.factor_model(...)

    def calculate_portfolio_risk(self, weights, trade_date):
        """计算组合风险指标"""
        # 因子暴露
        exposures = self.calculate_factor_exposures(trade_date, weights.index)

        # 协方差矩阵
        cov_matrix, factor_cov, idio_var = self.estimate_covariance(trade_date)

        # 风险分解
        risk_decomp = self.risk_metrics.decompose_risk(
            weights, exposures, factor_cov, idio_var
        )

        # VaR/CVaR
        returns_df = self.data_loader.get_daily_returns(...)
        var_95 = self.risk_metrics.calculate_var(weights, returns_df, 0.95)
        cvar_95 = self.risk_metrics.calculate_cvar(weights, returns_df, 0.95)

        return {
            'risk_decomposition': risk_decomp,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'exposures': exposures
        }

    def run_stress_tests(self, weights, trade_date):
        """运行压力测试"""
        results = {}

        # 历史情景
        results['historical'] = self.stress_tester.historical_scenarios(
            weights, self.data_loader
        )

        # 假设情景
        exposures = self.calculate_factor_exposures(trade_date, weights.index)
        results['hypothetical'] = self.stress_tester.hypothetical_scenarios(
            weights, exposures
        )

        # 蒙特卡洛模拟
        cov_matrix = self.estimate_covariance(trade_date)
        results['monte_carlo'] = self.stress_tester.monte_carlo(
            weights, cov_matrix
        )

        return results
```

---

## 7. 附录

### 7.1 数据表字段映射

| 模块 | 数据表 | 关键字段 |
|-----|-------|---------|
| 市场因子 | daily | ts_code, trade_date, pct_chg |
| 行业因子 | index_member_all | ts_code, l1_code, l1_name, in_date, out_date |
| 行业收益 | sw_daily | ts_code, trade_date, pct_change |
| Size | daily_basic | ts_code, trade_date, total_mv, circ_mv |
| Value | daily_basic | pe_ttm, pb, ps_ttm, dv_ttm |
| Momentum | daily | pct_chg |
| Volatility | daily | pct_chg |
| Liquidity | daily, daily_basic | vol, amount, turnover_rate_f |
| Growth | fina_indicator_vip | netprofit_yoy, or_yoy, assets_yoy, roe_yoy |
| Leverage | fina_indicator_vip | debt_to_assets, debt_to_eqt, current_ratio |

### 7.2 参数配置

```python
# 因子计算参数
FACTOR_PARAMS = {
    'beta': {
        'lookback_window': 252,
        'min_observations': 126
    },
    'momentum': {
        'short_window': 21,
        'mid_window': 126,
        'long_window': 252,
        'skip_recent': 21
    },
    'volatility': {
        'lookback_window': 252,
        'short_window': 21,
        'long_window': 126
    },
    'liquidity': {
        'lookback_window': 21
    }
}

# 协方差估计参数
COV_PARAMS = {
    'sample': {
        'window': 252
    },
    'ledoit_wolf': {
        'window': 252
    },
    'factor_model': {
        'newey_west_lags': 5,
        'eigenfactor_adjustment': True
    }
}

# VaR 计算参数
VAR_PARAMS = {
    'confidence_levels': [0.95, 0.99],
    'lookback_window': 252,
    'horizon': 1  # 1 day
}

# 压力测试参数
STRESS_TEST_PARAMS = {
    'monte_carlo': {
        'n_simulations': 10000,
        'horizon': 21
    },
    'evt': {
        'threshold_percentile': 5
    }
}
```

### 7.3 依赖库

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
duckdb>=0.8.0
statsmodels>=0.12.0
```

---

## 8. 更新日志

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0.0 | 2026-01-31 | 初始版本，包含完整风险模型框架 |

---

*本文档由风险管理专家设计，基于 Barra 多因子风险模型框架，结合 A 股市场特点和可用数据进行定制。*
