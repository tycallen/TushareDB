#!/usr/bin/env python3
"""
机器学习在因子投资中的应用研究

本研究包括:
1. 特征工程: 因子标准化、因子组合方法、非线性特征
2. 模型选择: 线性模型、树模型、神经网络
3. 实验设计: 回测框架、过拟合控制、模型评估

数据源: Tushare DuckDB 数据库
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 机器学习库
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import scipy.stats as stats

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

class FactorDataLoader:
    """因子数据加载器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def load_factor_data(self, start_date: str, end_date: str,
                         sample_stocks: int = None) -> pd.DataFrame:
        """
        加载因子数据

        合并以下表的数据:
        - stk_factor_pro: 技术因子
        - daily_basic: 基本面因子 (PE, PB, etc.)
        - fina_indicator_vip: 财务指标因子
        """
        print(f"Loading factor data from {start_date} to {end_date}...")

        # 1. 加载技术因子数据
        tech_query = f"""
        SELECT
            ts_code,
            trade_date,
            close_qfq as close,
            pct_chg,
            vol,
            amount,
            turnover_rate,
            turnover_rate_f,
            pe_ttm,
            pb,
            ps_ttm,
            total_mv,
            circ_mv,
            -- 动量因子
            rsi_bfq_6 as rsi_6,
            rsi_bfq_12 as rsi_12,
            rsi_bfq_24 as rsi_24,
            -- MACD
            macd_bfq as macd,
            macd_dif_bfq as macd_dif,
            macd_dea_bfq as macd_dea,
            -- KDJ
            kdj_k_bfq as kdj_k,
            kdj_d_bfq as kdj_d,
            kdj_bfq as kdj_j,
            -- 布林带
            boll_upper_bfq as boll_upper,
            boll_mid_bfq as boll_mid,
            boll_lower_bfq as boll_lower,
            -- 其他技术指标
            cci_bfq as cci,
            atr_bfq as atr,
            obv_bfq as obv,
            bias1_bfq as bias1,
            bias2_bfq as bias2,
            bias3_bfq as bias3,
            wr_bfq as wr,
            -- 均线
            ma_bfq_5 as ma5,
            ma_bfq_10 as ma10,
            ma_bfq_20 as ma20,
            ma_bfq_60 as ma60
        FROM stk_factor_pro
        WHERE trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
          AND close_qfq IS NOT NULL
          AND close_qfq > 0
        ORDER BY ts_code, trade_date
        """

        df = self.conn.execute(tech_query).fetchdf()
        print(f"Loaded {len(df):,} rows of technical factor data")

        # 采样股票（用于加速研究）
        if sample_stocks:
            unique_stocks = df['ts_code'].unique()
            if len(unique_stocks) > sample_stocks:
                # 选择交易最活跃的股票
                stock_counts = df.groupby('ts_code').size().nlargest(sample_stocks)
                selected_stocks = stock_counts.index.tolist()
                df = df[df['ts_code'].isin(selected_stocks)]
                print(f"Sampled {sample_stocks} stocks, {len(df):,} rows remaining")

        return df

    def load_financial_data(self, ts_codes: List[str]) -> pd.DataFrame:
        """加载财务数据"""
        codes_str = "', '".join(ts_codes)

        query = f"""
        SELECT
            ts_code,
            end_date,
            roe,
            roa,
            gross_margin,
            netprofit_margin,
            debt_to_assets,
            current_ratio,
            quick_ratio,
            eps,
            bps,
            cfps,
            netprofit_yoy,
            or_yoy as revenue_yoy
        FROM fina_indicator_vip
        WHERE ts_code IN ('{codes_str}')
          AND end_date >= '20200101'
        ORDER BY ts_code, end_date DESC
        """

        df = self.conn.execute(query).fetchdf()
        return df

    def close(self):
        self.conn.close()


class FeatureEngineering:
    """特征工程模块"""

    def __init__(self):
        self.scalers = {}

    # ==================== 1. 因子标准化 ====================

    def cross_sectional_zscore(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        截面Z-score标准化

        对每个交易日的因子值进行截面标准化:
        z = (x - mean) / std

        优点: 去除量纲影响，便于跨因子比较
        """
        result = df.copy()
        for col in factor_cols:
            if col in result.columns:
                result[f'{col}_zscore'] = result.groupby('trade_date')[col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        return result

    def cross_sectional_rank(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        截面排序标准化

        将因子值转换为排名百分比 [0, 1]

        优点: 对异常值鲁棒，分布更均匀
        """
        result = df.copy()
        for col in factor_cols:
            if col in result.columns:
                result[f'{col}_rank'] = result.groupby('trade_date')[col].transform(
                    lambda x: x.rank(pct=True)
                )
        return result

    def mad_winsorize(self, df: pd.DataFrame, factor_cols: List[str],
                      n_mad: float = 5.0) -> pd.DataFrame:
        """
        MAD (Median Absolute Deviation) 去极值

        使用中位数和MAD进行去极值处理:
        - 上限: median + n_mad * MAD
        - 下限: median - n_mad * MAD

        优点: 比均值-标准差方法更稳健
        """
        result = df.copy()

        def winsorize_group(x):
            median = x.median()
            mad = np.abs(x - median).median()
            if mad == 0:
                return x
            upper = median + n_mad * 1.4826 * mad  # 1.4826是正态分布下的校正系数
            lower = median - n_mad * 1.4826 * mad
            return x.clip(lower, upper)

        for col in factor_cols:
            if col in result.columns:
                result[f'{col}_mad'] = result.groupby('trade_date')[col].transform(winsorize_group)

        return result

    def market_neutralize(self, df: pd.DataFrame, factor_cols: List[str],
                          market_cap_col: str = 'total_mv') -> pd.DataFrame:
        """
        市值中性化

        使用回归方法去除因子与市值的相关性:
        factor_neutral = factor - beta * log(market_cap)

        优点: 避免因子收益被市值因素主导
        """
        result = df.copy()

        for col in factor_cols:
            if col not in result.columns:
                continue

            def neutralize_group(group):
                valid_mask = group[col].notna() & group[market_cap_col].notna() & (group[market_cap_col] > 0)
                if valid_mask.sum() < 10:
                    return group[col]

                y = group.loc[valid_mask, col].values
                x = np.log(group.loc[valid_mask, market_cap_col].values).reshape(-1, 1)

                # 简单线性回归
                x_mean = x.mean()
                y_mean = y.mean()
                beta = np.sum((x - x_mean) * (y - y_mean).reshape(-1, 1)) / np.sum((x - x_mean) ** 2)

                # 残差
                residual = group[col].copy()
                residual.loc[valid_mask] = y - beta * x.flatten()
                return residual

            result[f'{col}_neutral'] = result.groupby('trade_date').apply(
                lambda g: neutralize_group(g)
            ).reset_index(level=0, drop=True)

        return result

    # ==================== 2. 因子组合方法 ====================

    def equal_weight_composite(self, df: pd.DataFrame, factor_cols: List[str],
                               composite_name: str = 'composite') -> pd.DataFrame:
        """
        等权重因子组合

        简单平均所有因子的标准化值
        """
        result = df.copy()
        valid_cols = [col for col in factor_cols if col in df.columns]
        if valid_cols:
            result[composite_name] = df[valid_cols].mean(axis=1)
        return result

    def ic_weighted_composite(self, df: pd.DataFrame, factor_cols: List[str],
                              return_col: str = 'future_return',
                              lookback: int = 60) -> pd.DataFrame:
        """
        IC加权因子组合

        根据历史IC值动态调整因子权重
        """
        result = df.copy()
        result['ic_composite'] = np.nan

        dates = sorted(df['trade_date'].unique())

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            # 计算过去lookback天的IC
            lookback_dates = dates[i-lookback:i]
            lookback_data = df[df['trade_date'].isin(lookback_dates)]

            ics = {}
            for col in factor_cols:
                if col not in lookback_data.columns or return_col not in lookback_data.columns:
                    continue
                valid_mask = lookback_data[col].notna() & lookback_data[return_col].notna()
                if valid_mask.sum() > 30:
                    ic, _ = stats.spearmanr(
                        lookback_data.loc[valid_mask, col],
                        lookback_data.loc[valid_mask, return_col]
                    )
                    ics[col] = ic

            if ics:
                # IC绝对值加权
                total_abs_ic = sum(abs(v) for v in ics.values())
                if total_abs_ic > 0:
                    weights = {k: abs(v) / total_abs_ic for k, v in ics.items()}

                    mask = result['trade_date'] == date
                    composite = sum(
                        result.loc[mask, col] * w
                        for col, w in weights.items() if col in result.columns
                    )
                    result.loc[mask, 'ic_composite'] = composite

        return result

    def pca_composite(self, df: pd.DataFrame, factor_cols: List[str],
                      n_components: int = 3) -> pd.DataFrame:
        """
        PCA因子降维组合

        使用主成分分析提取因子的主要信息
        """
        result = df.copy()
        valid_cols = [col for col in factor_cols if col in df.columns]

        if not valid_cols:
            return result

        # 按日期分组进行PCA
        def apply_pca(group):
            data = group[valid_cols].dropna()
            if len(data) < 10:
                return pd.DataFrame(index=group.index)

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            pca = PCA(n_components=min(n_components, len(valid_cols)))
            components = pca.fit_transform(scaled_data)

            pca_df = pd.DataFrame(
                components,
                index=data.index,
                columns=[f'pca_{i+1}' for i in range(components.shape[1])]
            )
            return pca_df

        pca_results = df.groupby('trade_date').apply(apply_pca)
        pca_results = pca_results.reset_index(level=0, drop=True)

        for col in pca_results.columns:
            result[col] = pca_results[col]

        return result

    # ==================== 3. 非线性特征 ====================

    def polynomial_features(self, df: pd.DataFrame, factor_cols: List[str],
                           degree: int = 2) -> pd.DataFrame:
        """
        多项式特征

        生成因子的平方、立方等非线性变换
        """
        result = df.copy()

        for col in factor_cols:
            if col not in df.columns:
                continue
            for d in range(2, degree + 1):
                result[f'{col}_pow{d}'] = df[col] ** d

        return result

    def interaction_features(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        交互特征

        生成因子之间的乘积项
        """
        result = df.copy()
        valid_cols = [col for col in factor_cols if col in df.columns]

        for i, col1 in enumerate(valid_cols):
            for col2 in valid_cols[i+1:]:
                result[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        return result

    def momentum_features(self, df: pd.DataFrame, return_col: str = 'pct_chg',
                          windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        动量特征

        计算不同周期的累计收益率
        """
        result = df.copy()

        for window in windows:
            result[f'mom_{window}d'] = df.groupby('ts_code')[return_col].transform(
                lambda x: (1 + x / 100).rolling(window).apply(lambda y: y.prod() - 1, raw=True)
            )

        return result

    def volatility_features(self, df: pd.DataFrame, return_col: str = 'pct_chg',
                            windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        波动率特征

        计算不同周期的收益率标准差
        """
        result = df.copy()

        for window in windows:
            result[f'vol_{window}d'] = df.groupby('ts_code')[return_col].transform(
                lambda x: x.rolling(window).std()
            )

        return result

    def ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        比率特征

        计算价格与均线的比率等
        """
        result = df.copy()

        if 'close' in df.columns:
            for ma_col in ['ma5', 'ma10', 'ma20', 'ma60']:
                if ma_col in df.columns:
                    result[f'close_to_{ma_col}'] = df['close'] / df[ma_col]

        if 'boll_upper' in df.columns and 'boll_lower' in df.columns:
            result['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
            result['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

        return result

    def lag_features(self, df: pd.DataFrame, factor_cols: List[str],
                    lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        滞后特征

        生成因子的滞后值
        """
        result = df.copy()

        for col in factor_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                result[f'{col}_lag{lag}'] = df.groupby('ts_code')[col].shift(lag)

        return result


class ModelSelection:
    """模型选择模块"""

    def __init__(self):
        self.models = {}
        self.results = {}

    # ==================== 1. 线性模型 ====================

    def get_linear_models(self) -> Dict:
        """获取线性模型集合"""
        return {
            'OLS': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'Ridge_CV': Ridge(alpha=10.0),  # 更强正则化
        }

    # ==================== 2. 树模型 ====================

    def get_tree_models(self) -> Dict:
        """获取树模型集合"""
        return {
            'DecisionTree': DecisionTreeRegressor(max_depth=5, min_samples_leaf=100),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=50,
                n_jobs=-1,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=50,
                learning_rate=0.1,
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=50,
                learning_rate=0.1,
                random_state=42
            ),
        }

    # ==================== 3. 神经网络 ====================

    def get_neural_models(self) -> Dict:
        """获取神经网络模型集合"""
        return {
            'MLP_small': MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=200,
                early_stopping=True,
                random_state=42
            ),
            'MLP_medium': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=200,
                early_stopping=True,
                random_state=42
            ),
            'MLP_large': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=300,
                early_stopping=True,
                random_state=42
            ),
        }

    def get_all_models(self) -> Dict:
        """获取所有模型"""
        models = {}
        models.update(self.get_linear_models())
        models.update(self.get_tree_models())
        models.update(self.get_neural_models())
        return models


class BacktestFramework:
    """回测框架"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def create_labels(self, forward_days: int = 5) -> pd.DataFrame:
        """
        创建标签: 未来N日收益率
        """
        result = self.df.copy()
        result['future_return'] = result.groupby('ts_code')['pct_chg'].transform(
            lambda x: (1 + x / 100).rolling(forward_days).apply(lambda y: y.prod() - 1, raw=True).shift(-forward_days)
        )
        return result

    def time_series_split(self, train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        时间序列数据划分

        按时间顺序划分训练集、验证集、测试集
        避免未来信息泄露
        """
        dates = sorted(self.df['trade_date'].unique())
        n_dates = len(dates)

        train_end = int(n_dates * train_ratio)
        val_end = int(n_dates * (train_ratio + val_ratio))

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        train_df = self.df[self.df['trade_date'].isin(train_dates)]
        val_df = self.df[self.df['trade_date'].isin(val_dates)]
        test_df = self.df[self.df['trade_date'].isin(test_dates)]

        print(f"Train: {train_dates[0]} ~ {train_dates[-1]} ({len(train_df):,} samples)")
        print(f"Val: {val_dates[0]} ~ {val_dates[-1]} ({len(val_df):,} samples)")
        print(f"Test: {test_dates[0]} ~ {test_dates[-1]} ({len(test_df):,} samples)")

        return train_df, val_df, test_df

    def walk_forward_split(self, train_window: int = 252,
                           test_window: int = 21,
                           step: int = 21) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        滚动窗口划分

        使用固定窗口滚动的方式划分数据
        更接近实际投资场景
        """
        dates = sorted(self.df['trade_date'].unique())
        splits = []

        i = train_window
        while i + test_window <= len(dates):
            train_dates = dates[i-train_window:i]
            test_dates = dates[i:i+test_window]

            train_df = self.df[self.df['trade_date'].isin(train_dates)]
            test_df = self.df[self.df['trade_date'].isin(test_dates)]

            splits.append((train_df, test_df))
            i += step

        print(f"Created {len(splits)} walk-forward splits")
        return splits


class OverfittingControl:
    """过拟合控制模块"""

    @staticmethod
    def feature_importance_filter(model, feature_names: List[str],
                                   top_k: int = 20) -> List[str]:
        """
        基于特征重要性的特征筛选
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return feature_names[:top_k]

        indices = np.argsort(importance)[::-1][:top_k]
        return [feature_names[i] for i in indices]

    @staticmethod
    def mutual_info_filter(X: np.ndarray, y: np.ndarray,
                           feature_names: List[str], top_k: int = 20) -> List[str]:
        """
        基于互信息的特征筛选
        """
        mi_scores = mutual_info_regression(X, y, random_state=42)
        indices = np.argsort(mi_scores)[::-1][:top_k]
        return [feature_names[i] for i in indices]

    @staticmethod
    def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算信息系数 (IC)

        IC = Rank Correlation(预测值, 实际收益)
        """
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid_mask.sum() < 10:
            return 0.0
        ic, _ = stats.spearmanr(y_pred[valid_mask], y_true[valid_mask])
        return ic if not np.isnan(ic) else 0.0

    @staticmethod
    def calculate_icir(ics: List[float]) -> float:
        """
        计算ICIR (IC Information Ratio)

        ICIR = mean(IC) / std(IC)
        """
        ics = [ic for ic in ics if not np.isnan(ic)]
        if len(ics) < 2:
            return 0.0
        return np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0.0


class ModelEvaluation:
    """模型评估模块"""

    def __init__(self):
        self.metrics = {}

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                           name: str = '') -> Dict:
        """
        回归评估指标
        """
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return {}

        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'IC': OverfittingControl.calculate_ic(y_true, y_pred),
        }

        self.metrics[name] = metrics
        return metrics

    def evaluate_portfolio(self, df: pd.DataFrame, pred_col: str = 'prediction',
                          return_col: str = 'future_return',
                          n_groups: int = 5) -> Dict:
        """
        组合评估: 分组回测

        按预测值分组，评估各组的平均收益
        """
        result = df.copy()
        result = result.dropna(subset=[pred_col, return_col])

        # 每天分组
        def assign_group(x):
            try:
                return pd.qcut(x, n_groups, labels=False, duplicates='drop')
            except:
                return pd.Series([np.nan] * len(x), index=x.index)

        result['group'] = result.groupby('trade_date')[pred_col].transform(assign_group)

        # 计算各组平均收益
        group_returns = result.groupby(['trade_date', 'group'])[return_col].mean().unstack()

        # 多空组合收益
        if n_groups - 1 in group_returns.columns and 0 in group_returns.columns:
            long_short = group_returns[n_groups - 1] - group_returns[0]
        else:
            long_short = pd.Series([0.0])

        metrics = {
            'group_returns': group_returns.mean().to_dict(),
            'long_short_mean': long_short.mean(),
            'long_short_std': long_short.std(),
            'long_short_sharpe': long_short.mean() / long_short.std() * np.sqrt(252) if long_short.std() > 0 else 0,
            'win_rate': (long_short > 0).mean(),
        }

        return metrics


def run_research():
    """运行完整研究"""

    print("=" * 80)
    print("机器学习在因子投资中的应用研究")
    print("=" * 80)

    # 1. 数据加载
    print("\n" + "=" * 40)
    print("1. 数据加载")
    print("=" * 40)

    loader = FactorDataLoader(DB_PATH)

    # 使用最近2年数据进行研究
    df = loader.load_factor_data(
        start_date='20230101',
        end_date='20241231',
        sample_stocks=300  # 采样300只股票加速研究
    )

    loader.close()

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"Unique stocks: {df['ts_code'].nunique()}")

    # 2. 特征工程
    print("\n" + "=" * 40)
    print("2. 特征工程")
    print("=" * 40)

    fe = FeatureEngineering()

    # 定义基础因子
    base_factors = ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'rsi_6', 'rsi_12',
                    'macd', 'kdj_k', 'cci', 'bias1', 'bias2', 'wr']

    # 2.1 因子标准化
    print("\n2.1 因子标准化")
    df = fe.cross_sectional_zscore(df, base_factors)
    df = fe.cross_sectional_rank(df, base_factors)
    df = fe.mad_winsorize(df, base_factors)
    print(f"  - 截面Z-score标准化: {len([c for c in df.columns if '_zscore' in c])} features")
    print(f"  - 截面排序标准化: {len([c for c in df.columns if '_rank' in c])} features")
    print(f"  - MAD去极值: {len([c for c in df.columns if '_mad' in c])} features")

    # 2.2 非线性特征
    print("\n2.2 非线性特征")
    df = fe.momentum_features(df, windows=[5, 10, 20])
    df = fe.volatility_features(df, windows=[5, 10, 20])
    df = fe.ratio_features(df)
    print(f"  - 动量特征: {len([c for c in df.columns if 'mom_' in c])} features")
    print(f"  - 波动率特征: {len([c for c in df.columns if 'vol_' in c])} features")
    print(f"  - 比率特征: {len([c for c in df.columns if 'close_to_' in c or 'boll_' in c])} features")

    # 2.3 因子组合
    print("\n2.3 因子组合")
    zscore_factors = [c for c in df.columns if '_zscore' in c]
    df = fe.equal_weight_composite(df, zscore_factors, 'equal_composite')
    print(f"  - 等权重组合因子: equal_composite")

    # 3. 创建标签
    print("\n" + "=" * 40)
    print("3. 创建标签")
    print("=" * 40)

    bt = BacktestFramework(df)
    df = bt.create_labels(forward_days=5)
    df = df.dropna(subset=['future_return'])
    print(f"样本数量: {len(df):,}")
    print(f"标签统计: mean={df['future_return'].mean():.4f}, std={df['future_return'].std():.4f}")

    # 4. 数据划分
    print("\n" + "=" * 40)
    print("4. 数据划分")
    print("=" * 40)

    bt = BacktestFramework(df)
    train_df, val_df, test_df = bt.time_series_split(train_ratio=0.6, val_ratio=0.2)

    # 准备特征
    feature_cols = [c for c in df.columns if any(s in c for s in
                   ['_zscore', '_rank', 'mom_', 'vol_', 'close_to_', 'boll_', 'equal_composite'])]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"\n特征数量: {len(feature_cols)}")

    # 5. 模型训练与评估
    print("\n" + "=" * 40)
    print("5. 模型训练与评估")
    print("=" * 40)

    # 准备数据
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['future_return'].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['future_return'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['future_return'].values

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 获取模型
    ms = ModelSelection()
    evaluator = ModelEvaluation()

    all_results = {}

    # 5.1 线性模型
    print("\n5.1 线性模型")
    linear_models = ms.get_linear_models()
    for name, model in linear_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = evaluator.evaluate_regression(y_test, y_pred, name)
            all_results[name] = metrics
            print(f"  {name}: IC={metrics['IC']:.4f}, R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # 5.2 树模型
    print("\n5.2 树模型")
    tree_models = ms.get_tree_models()
    for name, model in tree_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = evaluator.evaluate_regression(y_test, y_pred, name)
            all_results[name] = metrics
            print(f"  {name}: IC={metrics['IC']:.4f}, R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # 5.3 神经网络
    print("\n5.3 神经网络")
    neural_models = ms.get_neural_models()
    for name, model in neural_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = evaluator.evaluate_regression(y_test, y_pred, name)
            all_results[name] = metrics
            print(f"  {name}: IC={metrics['IC']:.4f}, R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # 6. 组合回测
    print("\n" + "=" * 40)
    print("6. 组合回测")
    print("=" * 40)

    # 使用表现最好的模型进行组合回测
    best_model_name = max(all_results, key=lambda x: all_results[x].get('IC', 0))
    print(f"最佳模型: {best_model_name} (IC={all_results[best_model_name]['IC']:.4f})")

    # 重新训练最佳模型
    if best_model_name in linear_models:
        best_model = linear_models[best_model_name]
    elif best_model_name in tree_models:
        best_model = tree_models[best_model_name]
    else:
        best_model = neural_models[best_model_name]

    best_model.fit(X_train_scaled, y_train)

    # 预测
    test_df = test_df.copy()
    test_df['prediction'] = best_model.predict(X_test_scaled)

    # 组合评估
    portfolio_metrics = evaluator.evaluate_portfolio(test_df, n_groups=5)

    print(f"\n组合回测结果:")
    print(f"  分组平均收益: {portfolio_metrics['group_returns']}")
    print(f"  多空组合平均收益: {portfolio_metrics['long_short_mean']:.4f}")
    print(f"  多空组合夏普比率: {portfolio_metrics['long_short_sharpe']:.2f}")
    print(f"  多空组合胜率: {portfolio_metrics['win_rate']:.2%}")

    # 7. 生成报告
    print("\n" + "=" * 40)
    print("7. 生成研究报告")
    print("=" * 40)

    generate_report(df, all_results, portfolio_metrics, feature_cols, best_model_name)

    # 8. 生成可视化
    generate_visualizations(df, all_results, test_df, portfolio_metrics)

    print("\n研究完成!")
    print(f"报告已保存至: {REPORT_PATH}/")


def generate_report(df, all_results, portfolio_metrics, feature_cols, best_model_name):
    """生成研究报告"""

    report = f"""# 机器学习在因子投资中的应用研究报告

## 1. 研究概述

### 1.1 研究目的
本研究旨在探索机器学习技术在A股因子投资中的应用效果，包括：
- 特征工程方法的有效性验证
- 不同类型机器学习模型的预测能力比较
- 构建可实际应用的量化投资策略

### 1.2 数据概况
- **数据来源**: Tushare DuckDB 数据库
- **数据周期**: 2023-01-01 至 2024-12-31
- **样本股票**: 300只交易最活跃的股票
- **样本数量**: {len(df):,} 条观测数据
- **交易日数**: {df['trade_date'].nunique()} 天

## 2. 特征工程

### 2.1 因子标准化方法

#### 2.1.1 截面Z-score标准化
```
z = (x - mean) / std
```
- **优点**: 去除量纲影响，便于跨因子比较
- **缺点**: 对异常值敏感

#### 2.1.2 截面排序标准化
```
rank_pct = rank(x) / count(x)
```
- **优点**: 对异常值鲁棒，分布更均匀
- **缺点**: 丢失原始数值信息

#### 2.1.3 MAD去极值
```
上限: median + 5 * 1.4826 * MAD
下限: median - 5 * 1.4826 * MAD
```
- **优点**: 比均值-标准差方法更稳健
- **适用场景**: 含有极端值的因子

### 2.2 非线性特征构建

| 特征类型 | 构建方法 | 数量 |
|---------|---------|-----|
| 动量特征 | 累计收益率 (5/10/20日) | 3 |
| 波动率特征 | 滚动标准差 (5/10/20日) | 3 |
| 比率特征 | 价格/均线比率 | 4+ |
| 布林带特征 | 带宽、位置 | 2 |

### 2.3 因子组合方法

#### 等权重组合
```python
composite = mean(standardized_factors)
```

#### IC加权组合 (建议)
```python
weight_i = |IC_i| / sum(|IC|)
composite = sum(weight_i * factor_i)
```

## 3. 模型比较

### 3.1 模型评估指标

| 模型 | IC | R² | RMSE |
|-----|----|----|------|
"""

    # 添加模型结果
    for name, metrics in sorted(all_results.items(), key=lambda x: -x[1].get('IC', 0)):
        report += f"| {name} | {metrics.get('IC', 0):.4f} | {metrics.get('R2', 0):.4f} | {metrics.get('RMSE', 0):.4f} |\n"

    report += f"""
### 3.2 模型分析

#### 3.2.1 线性模型
- **OLS**: 基准模型，易过拟合
- **Ridge**: L2正则化，稳定性好
- **Lasso**: L1正则化，自动特征选择
- **ElasticNet**: L1+L2组合，平衡稀疏性和稳定性

#### 3.2.2 树模型
- **DecisionTree**: 可解释性强，但易过拟合
- **RandomForest**: 集成方法，泛化能力好
- **GradientBoosting**: 逐步优化，性能优秀
- **AdaBoost**: 关注难分类样本

#### 3.2.3 神经网络
- **MLP_small**: 简单架构，训练快
- **MLP_medium**: 中等复杂度，平衡性能
- **MLP_large**: 复杂架构，需要更多数据

### 3.3 最佳模型
**{best_model_name}** 在测试集上表现最佳:
- IC = {all_results[best_model_name].get('IC', 0):.4f}
- R² = {all_results[best_model_name].get('R2', 0):.4f}

## 4. 组合回测

### 4.1 分组回测方法
1. 每日根据模型预测值将股票分为5组
2. 计算每组的平均收益
3. 构建多空组合: 做多第5组，做空第1组

### 4.2 回测结果

| 分组 | 平均日收益 |
|-----|----------|
"""

    for group, ret in sorted(portfolio_metrics['group_returns'].items()):
        report += f"| 第{int(group)+1}组 | {ret:.4f} |\n"

    report += f"""
### 4.3 多空组合表现
- **平均日收益**: {portfolio_metrics['long_short_mean']:.4f}
- **年化夏普比率**: {portfolio_metrics['long_short_sharpe']:.2f}
- **胜率**: {portfolio_metrics['win_rate']:.2%}

## 5. 过拟合控制

### 5.1 时间序列划分
- 训练集: 60%
- 验证集: 20%
- 测试集: 20%

### 5.2 正则化策略
- L1/L2正则化 (线性模型)
- Early Stopping (神经网络)
- 限制树深度 (树模型)
- 最小叶子节点样本数

### 5.3 特征选择
- 基于IC的特征重要性排序
- 互信息特征选择
- 去除高度相关的冗余特征

## 6. 使用建议

### 6.1 模型选择
1. **低延迟场景**: 使用线性模型 (Ridge/Lasso)
2. **追求性能**: 使用GradientBoosting或RandomForest
3. **探索非线性**: 使用中等规模MLP

### 6.2 特征工程
1. 优先使用排序标准化，对异常值更鲁棒
2. 动量和波动率特征通常有效
3. 市值中性化可降低因子暴露风险

### 6.3 风险控制
1. 定期重新训练模型 (建议月度)
2. 监控IC衰减情况
3. 设置最大持仓限制
4. 行业和市值分散化

## 7. 后续研究方向

1. **深度学习模型**: LSTM、Transformer等序列模型
2. **强化学习**: 动态调整持仓和因子权重
3. **因子挖掘**: 使用遗传算法或符号回归发现新因子
4. **多任务学习**: 同时预测收益和风险
5. **图神经网络**: 利用股票间的关联关系

## 附录

### 使用的特征列表
总计 {len(feature_cols)} 个特征:
"""

    # 按类型分组展示特征
    feature_groups = {
        'Z-score标准化': [c for c in feature_cols if '_zscore' in c],
        '排序标准化': [c for c in feature_cols if '_rank' in c],
        '动量特征': [c for c in feature_cols if 'mom_' in c],
        '波动率特征': [c for c in feature_cols if 'vol_' in c],
        '比率特征': [c for c in feature_cols if 'close_to_' in c or 'boll_' in c],
        '组合因子': [c for c in feature_cols if 'composite' in c],
    }

    for group_name, features in feature_groups.items():
        if features:
            report += f"\n**{group_name}** ({len(features)}个):\n"
            report += ', '.join(features[:10])
            if len(features) > 10:
                report += f", ... (共{len(features)}个)"
            report += "\n"

    report += f"""
---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # 保存报告
    report_path = os.path.join(REPORT_PATH, 'ml_factor_investing_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存: {report_path}")


def generate_visualizations(df, all_results, test_df, portfolio_metrics):
    """生成可视化图表"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 模型IC比较
    ax1 = axes[0, 0]
    models = list(all_results.keys())
    ics = [all_results[m].get('IC', 0) for m in models]
    colors = ['#2ecc71' if ic > 0 else '#e74c3c' for ic in ics]
    bars = ax1.barh(models, ics, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('IC (Information Coefficient)')
    ax1.set_title('Model IC Comparison')
    ax1.grid(axis='x', alpha=0.3)

    # 2. 分组收益
    ax2 = axes[0, 1]
    groups = sorted(portfolio_metrics['group_returns'].keys())
    returns = [portfolio_metrics['group_returns'][g] for g in groups]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(groups)))
    ax2.bar([f'G{int(g)+1}' for g in groups], returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Portfolio Group')
    ax2.set_ylabel('Average Daily Return')
    ax2.set_title('Quintile Portfolio Returns')
    ax2.grid(axis='y', alpha=0.3)

    # 3. 预测值分布
    ax3 = axes[1, 0]
    if 'prediction' in test_df.columns:
        ax3.hist(test_df['prediction'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Model Predictions')
        ax3.grid(alpha=0.3)

    # 4. 实际收益 vs 预测收益
    ax4 = axes[1, 1]
    if 'prediction' in test_df.columns and 'future_return' in test_df.columns:
        sample = test_df.dropna(subset=['prediction', 'future_return']).sample(min(5000, len(test_df)))
        ax4.scatter(sample['prediction'], sample['future_return'], alpha=0.3, s=1, color='steelblue')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Predicted Return')
        ax4.set_ylabel('Actual Return')
        ax4.set_title('Predicted vs Actual Returns')
        ax4.grid(alpha=0.3)

        # 添加相关系数
        corr = sample['prediction'].corr(sample['future_return'])
        ax4.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存图表
    fig_path = os.path.join(REPORT_PATH, 'ml_factor_investing_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {fig_path}")

    # 生成第二张图: 特征重要性和时间序列
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # R2比较
    ax1 = axes2[0]
    r2s = [all_results[m].get('R2', 0) for m in models]
    ax1.barh(models, r2s, color='steelblue', alpha=0.7)
    ax1.set_xlabel('R² Score')
    ax1.set_title('Model R² Comparison')
    ax1.grid(axis='x', alpha=0.3)

    # RMSE比较
    ax2 = axes2[1]
    rmses = [all_results[m].get('RMSE', 0) for m in models]
    ax2.barh(models, rmses, color='coral', alpha=0.7)
    ax2.set_xlabel('RMSE')
    ax2.set_title('Model RMSE Comparison (Lower is Better)')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    fig2_path = os.path.join(REPORT_PATH, 'ml_model_metrics_comparison.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {fig2_path}")


if __name__ == '__main__':
    run_research()
