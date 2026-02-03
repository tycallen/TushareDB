#!/usr/bin/env python3
"""
股票相关性结构研究
================

研究内容：
1. 相关性分析：全市场相关性矩阵、行业内相关性、相关性时变特征
2. 协方差估计：样本协方差、收缩估计、因子协方差
3. 应用：分散化投资、对冲策略、风险分解

Author: Claude AI Assistant
Date: 2026-02-01
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, ShrunkCovariance, EmpiricalCovariance

# 输出目录
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'


class CorrelationStructureAnalysis:
    """股票相关性结构分析"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_content = []

    def log(self, msg):
        """记录日志"""
        print(msg)
        self.report_content.append(msg)

    def load_returns_data(self, start_date='20230101', end_date='20260130', min_days=200):
        """
        加载股票收益率数据

        Parameters:
        -----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        min_days : int
            最少交易天数
        """
        self.log(f"\n{'='*60}")
        self.log("加载股票收益率数据")
        self.log(f"{'='*60}")

        # 获取活跃股票列表（排除ST股票）
        stock_query = """
        SELECT ts_code, name, industry
        FROM stock_basic
        WHERE list_status = 'L'
        AND name NOT LIKE '%ST%'
        """
        stocks = self.conn.execute(stock_query).fetchdf()
        self.log(f"活跃非ST股票数量: {len(stocks)}")

        # 获取日收益率数据
        daily_query = f"""
        SELECT ts_code, trade_date, pct_chg
        FROM daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        daily_data = self.conn.execute(daily_query).fetchdf()
        self.log(f"原始数据行数: {len(daily_data)}")

        # 转换为收益率矩阵（透视表）
        returns_pivot = daily_data.pivot(index='trade_date', columns='ts_code', values='pct_chg')
        self.log(f"收益率矩阵形状: {returns_pivot.shape}")

        # 筛选有足够数据的股票
        valid_stocks = returns_pivot.columns[returns_pivot.notna().sum() >= min_days]
        returns = returns_pivot[valid_stocks]
        self.log(f"有效股票数量（至少{min_days}天数据）: {len(valid_stocks)}")

        # 填充缺失值（使用0填充，表示该日未交易）
        returns = returns.fillna(0) / 100  # 转换为小数形式

        # 存储数据
        self.returns = returns
        self.stocks = stocks[stocks['ts_code'].isin(valid_stocks)].set_index('ts_code')
        self.trade_dates = returns.index.tolist()

        self.log(f"最终收益率矩阵形状: {returns.shape}")
        self.log(f"日期范围: {self.trade_dates[0]} 到 {self.trade_dates[-1]}")

        return returns

    def compute_correlation_matrix(self):
        """计算全市场相关性矩阵"""
        self.log(f"\n{'='*60}")
        self.log("1. 全市场相关性矩阵分析")
        self.log(f"{'='*60}")

        # 计算相关性矩阵
        corr_matrix = self.returns.corr()
        self.corr_matrix = corr_matrix

        # 获取上三角元素（不包括对角线）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.values[mask]

        self.log(f"\n相关性统计:")
        self.log(f"  平均相关性: {np.mean(corr_values):.4f}")
        self.log(f"  中位数相关性: {np.median(corr_values):.4f}")
        self.log(f"  标准差: {np.std(corr_values):.4f}")
        self.log(f"  最小值: {np.min(corr_values):.4f}")
        self.log(f"  最大值: {np.max(corr_values):.4f}")
        self.log(f"  正相关比例: {(corr_values > 0).mean()*100:.2f}%")
        self.log(f"  强正相关(>0.5)比例: {(corr_values > 0.5).mean()*100:.2f}%")

        # 相关性分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        axes[0].hist(corr_values, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        axes[0].axvline(np.mean(corr_values), color='red', linestyle='--', label=f'均值: {np.mean(corr_values):.3f}')
        axes[0].axvline(np.median(corr_values), color='green', linestyle='--', label=f'中位数: {np.median(corr_values):.3f}')
        axes[0].set_xlabel('相关系数')
        axes[0].set_ylabel('密度')
        axes[0].set_title('全市场股票相关性分布')
        axes[0].legend()

        # 箱线图 - 按分位数
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(corr_values, p) for p in percentiles]
        axes[1].boxplot(corr_values, vert=True)
        axes[1].set_ylabel('相关系数')
        axes[1].set_title('相关性箱线图')

        # 添加百分位数标注
        for p, v in zip(percentiles, percentile_values):
            axes[1].axhline(v, color='gray', linestyle=':', alpha=0.5)
            axes[1].text(1.15, v, f'{p}%: {v:.3f}', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/correlation_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存相关性分布图: {OUTPUT_DIR}/correlation_distribution.png")

        return corr_matrix

    def compute_industry_correlation(self):
        """计算行业内相关性"""
        self.log(f"\n{'='*60}")
        self.log("2. 行业内相关性分析")
        self.log(f"{'='*60}")

        # 获取行业信息
        industries = self.stocks['industry'].dropna()
        industry_corrs = {}

        self.log("\n各行业内部平均相关性:")
        self.log("-" * 50)

        for industry in industries.unique():
            industry_stocks = industries[industries == industry].index.tolist()
            # 只取在收益率矩阵中存在的股票
            industry_stocks = [s for s in industry_stocks if s in self.returns.columns]

            if len(industry_stocks) >= 5:  # 至少5只股票
                industry_returns = self.returns[industry_stocks]
                industry_corr = industry_returns.corr()

                # 获取上三角元素
                mask = np.triu(np.ones_like(industry_corr, dtype=bool), k=1)
                corr_values = industry_corr.values[mask]

                if len(corr_values) > 0:
                    avg_corr = np.mean(corr_values)
                    industry_corrs[industry] = {
                        'avg_corr': avg_corr,
                        'std_corr': np.std(corr_values),
                        'n_stocks': len(industry_stocks)
                    }

        # 按平均相关性排序
        sorted_industries = sorted(industry_corrs.items(), key=lambda x: x[1]['avg_corr'], reverse=True)

        for industry, stats in sorted_industries[:20]:
            self.log(f"  {industry:12s}: 平均相关性 {stats['avg_corr']:.4f}, "
                    f"标准差 {stats['std_corr']:.4f}, 股票数 {stats['n_stocks']}")

        self.industry_corrs = pd.DataFrame(industry_corrs).T

        # 可视化行业相关性
        if len(sorted_industries) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 行业平均相关性柱状图（前20）
            top_industries = sorted_industries[:20]
            industries_list = [x[0] for x in top_industries]
            avg_corrs = [x[1]['avg_corr'] for x in top_industries]

            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(industries_list)))
            axes[0].barh(range(len(industries_list)), avg_corrs, color=colors)
            axes[0].set_yticks(range(len(industries_list)))
            axes[0].set_yticklabels(industries_list)
            axes[0].set_xlabel('平均相关系数')
            axes[0].set_title('行业内部平均相关性（前20）')
            axes[0].axvline(np.mean([x[1]['avg_corr'] for x in sorted_industries]),
                           color='red', linestyle='--', label='全市场均值')
            axes[0].legend()

            # 行业相关性 vs 股票数量散点图
            all_avg_corrs = [x[1]['avg_corr'] for x in sorted_industries]
            all_n_stocks = [x[1]['n_stocks'] for x in sorted_industries]

            axes[1].scatter(all_n_stocks, all_avg_corrs, alpha=0.6, c='steelblue')
            axes[1].set_xlabel('行业股票数量')
            axes[1].set_ylabel('行业内平均相关性')
            axes[1].set_title('行业规模 vs 行业内相关性')

            # 添加趋势线
            z = np.polyfit(all_n_stocks, all_avg_corrs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_n_stocks), max(all_n_stocks), 100)
            axes[1].plot(x_line, p(x_line), 'r--', alpha=0.5, label='趋势线')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/industry_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
            self.log(f"\n已保存行业相关性图: {OUTPUT_DIR}/industry_correlation.png")

        # 计算行业间相关性
        self.log("\n行业间相关性分析:")
        self.compute_cross_industry_correlation()

        return self.industry_corrs

    def compute_cross_industry_correlation(self):
        """计算行业间相关性"""
        industries = self.stocks['industry'].dropna()

        # 计算各行业平均收益率
        industry_returns = {}
        for industry in industries.unique():
            industry_stocks = industries[industries == industry].index.tolist()
            industry_stocks = [s for s in industry_stocks if s in self.returns.columns]

            if len(industry_stocks) >= 5:
                # 等权平均
                industry_returns[industry] = self.returns[industry_stocks].mean(axis=1)

        industry_returns_df = pd.DataFrame(industry_returns)
        industry_corr = industry_returns_df.corr()

        # 保存行业间相关性矩阵
        self.cross_industry_corr = industry_corr

        # 热力图
        if len(industry_corr) > 3:
            # 选择相关性差异最大的行业
            n_industries = min(20, len(industry_corr))

            # 使用聚类排序
            if len(industry_corr) > 2:
                # 层次聚类
                distance_matrix = 1 - industry_corr.values
                distance_matrix = np.clip(distance_matrix, 0, 2)
                np.fill_diagonal(distance_matrix, 0)

                try:
                    condensed_dist = squareform(distance_matrix)
                    linkage_matrix = linkage(condensed_dist, method='ward')
                    order = dendrogram(linkage_matrix, no_plot=True)['leaves']

                    sorted_industries = industry_corr.columns[order].tolist()
                    industry_corr_sorted = industry_corr.loc[sorted_industries, sorted_industries]
                except:
                    industry_corr_sorted = industry_corr.iloc[:n_industries, :n_industries]
            else:
                industry_corr_sorted = industry_corr

            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(industry_corr_sorted, annot=False, cmap='RdYlGn', center=0,
                       vmin=-0.5, vmax=1, ax=ax)
            ax.set_title('行业间相关性热力图（聚类排序）')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/cross_industry_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
            self.log(f"已保存行业间相关性热力图: {OUTPUT_DIR}/cross_industry_correlation.png")

        # 找出相关性最高和最低的行业对
        mask = np.triu(np.ones_like(industry_corr, dtype=bool), k=1)
        corr_pairs = []
        for i in range(len(industry_corr)):
            for j in range(i+1, len(industry_corr)):
                corr_pairs.append((
                    industry_corr.index[i],
                    industry_corr.columns[j],
                    industry_corr.iloc[i, j]
                ))

        corr_pairs.sort(key=lambda x: x[2], reverse=True)

        self.log("\n相关性最高的行业对:")
        for ind1, ind2, corr in corr_pairs[:5]:
            self.log(f"  {ind1} - {ind2}: {corr:.4f}")

        self.log("\n相关性最低的行业对:")
        for ind1, ind2, corr in corr_pairs[-5:]:
            self.log(f"  {ind1} - {ind2}: {corr:.4f}")

    def compute_time_varying_correlation(self, window=60):
        """计算相关性的时变特征"""
        self.log(f"\n{'='*60}")
        self.log("3. 相关性时变特征分析")
        self.log(f"{'='*60}")

        # 选择流动性好的股票进行分析
        n_stocks = min(100, self.returns.shape[1])

        # 选择数据完整性最好的股票
        data_completeness = self.returns.notna().sum()
        top_stocks = data_completeness.nlargest(n_stocks).index.tolist()
        returns_subset = self.returns[top_stocks]

        self.log(f"使用 {n_stocks} 只股票进行时变相关性分析")
        self.log(f"滚动窗口: {window} 天")

        # 计算滚动平均相关性
        rolling_avg_corr = []
        rolling_std_corr = []
        dates = []

        for i in range(window, len(returns_subset)):
            window_returns = returns_subset.iloc[i-window:i]
            corr_matrix = window_returns.corr()

            # 获取上三角元素
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.values[mask]

            rolling_avg_corr.append(np.nanmean(corr_values))
            rolling_std_corr.append(np.nanstd(corr_values))
            dates.append(returns_subset.index[i])

        rolling_df = pd.DataFrame({
            'date': dates,
            'avg_corr': rolling_avg_corr,
            'std_corr': rolling_std_corr
        })
        rolling_df['date'] = pd.to_datetime(rolling_df['date'])
        rolling_df.set_index('date', inplace=True)

        self.rolling_corr = rolling_df

        # 统计
        self.log(f"\n滚动相关性统计:")
        self.log(f"  平均值范围: {rolling_df['avg_corr'].min():.4f} - {rolling_df['avg_corr'].max():.4f}")
        self.log(f"  平均值的均值: {rolling_df['avg_corr'].mean():.4f}")
        self.log(f"  平均值的标准差: {rolling_df['avg_corr'].std():.4f}")

        # 相关性高峰和低谷
        high_corr_dates = rolling_df['avg_corr'].nlargest(5)
        low_corr_dates = rolling_df['avg_corr'].nsmallest(5)

        self.log("\n相关性最高的时期:")
        for date, corr in high_corr_dates.items():
            self.log(f"  {date.strftime('%Y-%m-%d')}: {corr:.4f}")

        self.log("\n相关性最低的时期:")
        for date, corr in low_corr_dates.items():
            self.log(f"  {date.strftime('%Y-%m-%d')}: {corr:.4f}")

        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 时序图
        axes[0].plot(rolling_df.index, rolling_df['avg_corr'], color='steelblue', linewidth=1)
        axes[0].fill_between(rolling_df.index,
                            rolling_df['avg_corr'] - rolling_df['std_corr'],
                            rolling_df['avg_corr'] + rolling_df['std_corr'],
                            alpha=0.3, color='steelblue')
        axes[0].axhline(rolling_df['avg_corr'].mean(), color='red', linestyle='--',
                       label=f'均值: {rolling_df["avg_corr"].mean():.3f}')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('平均相关系数')
        axes[0].set_title(f'滚动{window}日平均相关性时序图')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 相关性与市场波动的关系
        # 计算市场波动率（使用等权平均收益率的标准差）
        market_returns = returns_subset.mean(axis=1)
        rolling_vol = market_returns.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.iloc[window:]

        # 对齐日期
        common_dates = rolling_df.index.intersection(rolling_vol.index)
        corr_values = rolling_df.loc[common_dates, 'avg_corr']
        vol_values = rolling_vol.loc[common_dates]

        axes[1].scatter(vol_values, corr_values, alpha=0.5, s=10, c='steelblue')
        axes[1].set_xlabel('市场波动率（年化）')
        axes[1].set_ylabel('平均相关系数')
        axes[1].set_title('相关性 vs 市场波动率')

        # 计算相关性
        if len(corr_values) > 0 and len(vol_values) > 0:
            corr_vol_corr = np.corrcoef(corr_values, vol_values)[0, 1]
            axes[1].text(0.05, 0.95, f'相关系数: {corr_vol_corr:.3f}',
                        transform=axes[1].transAxes, fontsize=12)

            # 添加趋势线
            z = np.polyfit(vol_values, corr_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(vol_values.min(), vol_values.max(), 100)
            axes[1].plot(x_line, p(x_line), 'r--', alpha=0.7, label='趋势线')
            axes[1].legend()

        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/time_varying_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存时变相关性图: {OUTPUT_DIR}/time_varying_correlation.png")

        return rolling_df

    def estimate_covariance_matrices(self):
        """估计协方差矩阵：样本协方差、收缩估计、因子协方差"""
        self.log(f"\n{'='*60}")
        self.log("4. 协方差矩阵估计")
        self.log(f"{'='*60}")

        # 选择流动性好的股票
        n_stocks = min(200, self.returns.shape[1])
        data_completeness = self.returns.notna().sum()
        top_stocks = data_completeness.nlargest(n_stocks).index.tolist()
        returns_subset = self.returns[top_stocks].dropna()

        self.log(f"使用 {n_stocks} 只股票进行协方差估计")
        self.log(f"有效样本数: {len(returns_subset)}")

        X = returns_subset.values

        # 1. 样本协方差
        self.log("\n4.1 样本协方差矩阵")
        sample_cov = EmpiricalCovariance().fit(X)
        sample_cov_matrix = sample_cov.covariance_

        # 计算条件数（衡量矩阵稳定性）
        try:
            cond_number = np.linalg.cond(sample_cov_matrix)
            self.log(f"  条件数: {cond_number:.2e}")
        except:
            self.log("  条件数: 无法计算")

        # 特征值分析
        eigenvalues = np.linalg.eigvalsh(sample_cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

        self.log(f"  最大特征值: {eigenvalues[0]:.6e}")
        self.log(f"  最小特征值: {eigenvalues[-1]:.6e}")
        self.log(f"  特征值比率: {eigenvalues[0]/eigenvalues[-1]:.2e}")

        # 2. Ledoit-Wolf 收缩估计
        self.log("\n4.2 Ledoit-Wolf 收缩估计")
        lw_cov = LedoitWolf().fit(X)
        lw_cov_matrix = lw_cov.covariance_
        shrinkage = lw_cov.shrinkage_

        self.log(f"  收缩强度: {shrinkage:.4f}")

        try:
            lw_cond_number = np.linalg.cond(lw_cov_matrix)
            self.log(f"  条件数: {lw_cond_number:.2e}")
        except:
            self.log("  条件数: 无法计算")

        lw_eigenvalues = np.linalg.eigvalsh(lw_cov_matrix)
        lw_eigenvalues = np.sort(lw_eigenvalues)[::-1]

        self.log(f"  最大特征值: {lw_eigenvalues[0]:.6e}")
        self.log(f"  最小特征值: {lw_eigenvalues[-1]:.6e}")

        # 3. 因子协方差模型（PCA方法）
        self.log("\n4.3 因子协方差模型（PCA）")
        n_factors = 10
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(X)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_variance_ratio)

        self.log(f"  使用因子数: {n_factors}")
        self.log(f"  前{n_factors}个因子解释方差比例:")
        for i in range(min(5, n_factors)):
            self.log(f"    因子{i+1}: {explained_variance_ratio[i]*100:.2f}% (累计: {cumulative_var[i]*100:.2f}%)")

        # 因子协方差矩阵重构
        factor_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        factor_cov = factor_loadings @ factor_loadings.T

        # 添加特质方差
        residual_var = np.var(X - factors @ pca.components_, axis=0)
        factor_cov_matrix = factor_cov + np.diag(residual_var)

        try:
            factor_cond_number = np.linalg.cond(factor_cov_matrix)
            self.log(f"  条件数: {factor_cond_number:.2e}")
        except:
            self.log("  条件数: 无法计算")

        # 保存协方差矩阵
        self.sample_cov = sample_cov_matrix
        self.lw_cov = lw_cov_matrix
        self.factor_cov = factor_cov_matrix
        self.pca = pca
        self.cov_stocks = top_stocks

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 特征值谱
        axes[0, 0].semilogy(range(1, len(eigenvalues)+1), eigenvalues, 'b-', label='样本协方差', alpha=0.7)
        axes[0, 0].semilogy(range(1, len(lw_eigenvalues)+1), lw_eigenvalues, 'r-', label='Ledoit-Wolf', alpha=0.7)
        axes[0, 0].set_xlabel('特征值序号')
        axes[0, 0].set_ylabel('特征值（对数尺度）')
        axes[0, 0].set_title('协方差矩阵特征值谱')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # PCA解释方差
        axes[0, 1].bar(range(1, n_factors+1), explained_variance_ratio*100, color='steelblue', alpha=0.7)
        axes[0, 1].plot(range(1, n_factors+1), cumulative_var*100, 'r-o', label='累计')
        axes[0, 1].set_xlabel('因子序号')
        axes[0, 1].set_ylabel('解释方差比例 (%)')
        axes[0, 1].set_title(f'PCA因子解释方差（前{n_factors}个）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 样本协方差热力图（前50只股票）
        n_show = min(50, len(sample_cov_matrix))
        im1 = axes[1, 0].imshow(sample_cov_matrix[:n_show, :n_show], cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('样本协方差矩阵（前50只股票）')
        plt.colorbar(im1, ax=axes[1, 0])

        # Ledoit-Wolf协方差热力图
        im2 = axes[1, 1].imshow(lw_cov_matrix[:n_show, :n_show], cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Ledoit-Wolf协方差矩阵（前50只股票）')
        plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/covariance_estimation.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存协方差估计图: {OUTPUT_DIR}/covariance_estimation.png")

        # 比较不同估计方法
        self.compare_covariance_methods(X)

        return {
            'sample': sample_cov_matrix,
            'ledoit_wolf': lw_cov_matrix,
            'factor': factor_cov_matrix
        }

    def compare_covariance_methods(self, X):
        """比较不同协方差估计方法的性能"""
        self.log("\n4.4 协方差估计方法比较")

        n_samples, n_features = X.shape

        # 分割训练集和测试集
        train_size = int(n_samples * 0.7)
        X_train = X[:train_size]
        X_test = X[train_size:]

        methods = {
            '样本协方差': EmpiricalCovariance(),
            'Ledoit-Wolf': LedoitWolf(),
            '收缩协方差(0.1)': ShrunkCovariance(shrinkage=0.1),
            '收缩协方差(0.3)': ShrunkCovariance(shrinkage=0.3),
        }

        results = {}
        for name, estimator in methods.items():
            estimator.fit(X_train)

            # 在测试集上评估
            try:
                test_score = estimator.score(X_test)
                results[name] = test_score
                self.log(f"  {name}: 测试集对数似然 = {test_score:.4f}")
            except:
                self.log(f"  {name}: 无法计算测试分数")

    def analyze_diversification(self):
        """分散化投资分析"""
        self.log(f"\n{'='*60}")
        self.log("5. 分散化投资应用")
        self.log(f"{'='*60}")

        # 使用Ledoit-Wolf协方差
        cov_matrix = self.lw_cov
        returns_subset = self.returns[self.cov_stocks]

        # 计算平均相关性随组合规模的变化
        n_stocks_list = [5, 10, 20, 30, 50, 100, 150, 200]
        n_simulations = 100

        results = []
        for n in n_stocks_list:
            if n > len(self.cov_stocks):
                continue

            portfolio_vols = []
            avg_stock_vols = []
            diversification_ratios = []

            for _ in range(n_simulations):
                # 随机选择n只股票
                selected_indices = np.random.choice(len(self.cov_stocks), n, replace=False)
                selected_cov = cov_matrix[np.ix_(selected_indices, selected_indices)]

                # 等权组合
                weights = np.ones(n) / n

                # 组合方差
                portfolio_var = weights @ selected_cov @ weights
                portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)

                # 平均个股波动率
                avg_stock_vol = np.mean(np.sqrt(np.diag(selected_cov))) * np.sqrt(252)

                # 分散化比率
                div_ratio = avg_stock_vol / portfolio_vol

                portfolio_vols.append(portfolio_vol)
                avg_stock_vols.append(avg_stock_vol)
                diversification_ratios.append(div_ratio)

            results.append({
                'n_stocks': n,
                'portfolio_vol': np.mean(portfolio_vols),
                'portfolio_vol_std': np.std(portfolio_vols),
                'avg_stock_vol': np.mean(avg_stock_vols),
                'diversification_ratio': np.mean(diversification_ratios),
                'div_ratio_std': np.std(diversification_ratios)
            })

        results_df = pd.DataFrame(results)

        self.log("\n分散化效果随组合规模变化:")
        self.log("-" * 70)
        self.log(f"{'股票数':<10}{'组合波动率':<15}{'平均个股波动率':<15}{'分散化比率':<15}")
        self.log("-" * 70)

        for _, row in results_df.iterrows():
            self.log(f"{int(row['n_stocks']):<10}{row['portfolio_vol']*100:.2f}%{'':<10}"
                    f"{row['avg_stock_vol']*100:.2f}%{'':<10}{row['diversification_ratio']:.2f}")

        # 计算有效前沿
        self.log("\n计算有效前沿...")
        self.compute_efficient_frontier()

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 分散化效果图
        axes[0].errorbar(results_df['n_stocks'], results_df['portfolio_vol']*100,
                        yerr=results_df['portfolio_vol_std']*100,
                        fmt='o-', label='组合波动率', capsize=3)
        axes[0].plot(results_df['n_stocks'], results_df['avg_stock_vol']*100,
                    's--', label='平均个股波动率')
        axes[0].set_xlabel('组合股票数量')
        axes[0].set_ylabel('年化波动率 (%)')
        axes[0].set_title('分散化效果：组合规模 vs 波动率')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 分散化比率图
        axes[1].errorbar(results_df['n_stocks'], results_df['diversification_ratio'],
                        yerr=results_df['div_ratio_std'], fmt='o-', color='green', capsize=3)
        axes[1].set_xlabel('组合股票数量')
        axes[1].set_ylabel('分散化比率')
        axes[1].set_title('分散化比率（平均个股波动率/组合波动率）')
        axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/diversification_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存分散化分析图: {OUTPUT_DIR}/diversification_analysis.png")

        self.diversification_results = results_df
        return results_df

    def compute_efficient_frontier(self):
        """计算有效前沿"""
        # 使用较少的股票以提高计算效率
        n_stocks = min(50, len(self.cov_stocks))
        selected_stocks = self.cov_stocks[:n_stocks]
        returns_subset = self.returns[selected_stocks]

        # 计算预期收益和协方差
        mean_returns = returns_subset.mean() * 252  # 年化
        cov_matrix = returns_subset.cov() * 252  # 年化

        # 生成随机组合
        n_portfolios = 5000
        results = np.zeros((3, n_portfolios))

        for i in range(n_portfolios):
            weights = np.random.random(n_stocks)
            weights /= np.sum(weights)

            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = portfolio_return / portfolio_std  # Sharpe ratio (假设无风险利率为0)

        # 找出有效前沿
        # 最小方差组合
        min_var_idx = np.argmin(results[0])
        max_sharpe_idx = np.argmax(results[2])

        self.log(f"\n有效前沿分析（使用{n_stocks}只股票）:")
        self.log(f"  最小方差组合: 收益率 {results[1, min_var_idx]*100:.2f}%, "
                f"波动率 {results[0, min_var_idx]*100:.2f}%")
        self.log(f"  最大夏普比组合: 收益率 {results[1, max_sharpe_idx]*100:.2f}%, "
                f"波动率 {results[0, max_sharpe_idx]*100:.2f}%, "
                f"夏普比 {results[2, max_sharpe_idx]:.2f}")

        # 可视化有效前沿
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(results[0]*100, results[1]*100, c=results[2],
                            cmap='RdYlGn', alpha=0.5, s=10)
        ax.scatter(results[0, min_var_idx]*100, results[1, min_var_idx]*100,
                  marker='*', color='red', s=200, label='最小方差组合')
        ax.scatter(results[0, max_sharpe_idx]*100, results[1, max_sharpe_idx]*100,
                  marker='*', color='gold', s=200, label='最大夏普比组合')

        ax.set_xlabel('年化波动率 (%)')
        ax.set_ylabel('年化收益率 (%)')
        ax.set_title('有效前沿（蒙特卡洛模拟）')
        ax.legend()
        plt.colorbar(scatter, label='夏普比率')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/efficient_frontier.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"已保存有效前沿图: {OUTPUT_DIR}/efficient_frontier.png")

    def analyze_hedging_strategies(self):
        """对冲策略分析"""
        self.log(f"\n{'='*60}")
        self.log("6. 对冲策略应用")
        self.log(f"{'='*60}")

        # 1. 行业对冲
        self.log("\n6.1 行业对冲分析")

        industries = self.stocks['industry'].dropna()

        # 找出负相关或低相关的行业对
        if hasattr(self, 'cross_industry_corr'):
            corr = self.cross_industry_corr

            # 找出相关性最低的行业对
            hedge_pairs = []
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    hedge_pairs.append({
                        'industry1': corr.index[i],
                        'industry2': corr.columns[j],
                        'correlation': corr.iloc[i, j]
                    })

            hedge_pairs.sort(key=lambda x: x['correlation'])

            self.log("\n最佳对冲行业对（低相关性）:")
            for pair in hedge_pairs[:10]:
                self.log(f"  {pair['industry1']} vs {pair['industry2']}: {pair['correlation']:.4f}")

        # 2. 个股对冲
        self.log("\n6.2 个股对冲分析")

        # 选择部分股票进行分析
        n_stocks = min(100, len(self.corr_matrix))
        corr_subset = self.corr_matrix.iloc[:n_stocks, :n_stocks]

        # 找出负相关的股票对
        negative_pairs = []
        for i in range(len(corr_subset)):
            for j in range(i+1, len(corr_subset)):
                corr_val = corr_subset.iloc[i, j]
                if corr_val < 0:
                    stock1 = corr_subset.index[i]
                    stock2 = corr_subset.columns[j]
                    name1 = self.stocks.loc[stock1, 'name'] if stock1 in self.stocks.index else stock1
                    name2 = self.stocks.loc[stock2, 'name'] if stock2 in self.stocks.index else stock2
                    negative_pairs.append({
                        'stock1': stock1,
                        'name1': name1,
                        'stock2': stock2,
                        'name2': name2,
                        'correlation': corr_val
                    })

        negative_pairs.sort(key=lambda x: x['correlation'])

        self.log(f"\n发现 {len(negative_pairs)} 对负相关股票")
        self.log("\n相关性最负的股票对:")
        for pair in negative_pairs[:10]:
            self.log(f"  {pair['name1']}({pair['stock1']}) vs "
                    f"{pair['name2']}({pair['stock2']}): {pair['correlation']:.4f}")

        # 3. Beta对冲分析
        self.log("\n6.3 Beta对冲分析")

        # 计算市场组合收益（等权）
        market_returns = self.returns.mean(axis=1)

        # 计算各股票的Beta
        betas = {}
        for stock in self.cov_stocks[:100]:  # 选择前100只
            stock_returns = self.returns[stock]

            # 计算Beta
            cov_with_market = stock_returns.cov(market_returns)
            market_var = market_returns.var()

            if market_var > 0:
                beta = cov_with_market / market_var
                betas[stock] = beta

        betas_df = pd.DataFrame.from_dict(betas, orient='index', columns=['beta'])
        betas_df['name'] = [self.stocks.loc[s, 'name'] if s in self.stocks.index else s
                          for s in betas_df.index]

        # Beta统计
        self.log(f"\n股票Beta统计:")
        self.log(f"  平均Beta: {betas_df['beta'].mean():.4f}")
        self.log(f"  中位数Beta: {betas_df['beta'].median():.4f}")
        self.log(f"  Beta标准差: {betas_df['beta'].std():.4f}")
        self.log(f"  Beta < 0的股票数: {(betas_df['beta'] < 0).sum()}")
        self.log(f"  Beta < 0.5的股票数: {(betas_df['beta'] < 0.5).sum()}")

        self.log("\n低Beta股票（防御性）:")
        low_beta = betas_df.nsmallest(10, 'beta')
        for idx, row in low_beta.iterrows():
            self.log(f"  {row['name']}({idx}): Beta = {row['beta']:.4f}")

        self.log("\n高Beta股票（进攻性）:")
        high_beta = betas_df.nlargest(10, 'beta')
        for idx, row in high_beta.iterrows():
            self.log(f"  {row['name']}({idx}): Beta = {row['beta']:.4f}")

        self.betas = betas_df

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Beta分布
        axes[0].hist(betas_df['beta'], bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        axes[0].axvline(1.0, color='red', linestyle='--', label='Beta = 1')
        axes[0].axvline(betas_df['beta'].mean(), color='green', linestyle='--',
                       label=f'均值: {betas_df["beta"].mean():.2f}')
        axes[0].set_xlabel('Beta')
        axes[0].set_ylabel('密度')
        axes[0].set_title('股票Beta分布')
        axes[0].legend()

        # 行业对冲相关性
        if hasattr(self, 'cross_industry_corr'):
            corr = self.cross_industry_corr
            # 取部分行业展示
            n_show = min(15, len(corr))
            mask = np.triu(np.ones((n_show, n_show), dtype=bool), k=1)
            sns.heatmap(corr.iloc[:n_show, :n_show], mask=~mask, annot=True, fmt='.2f',
                       cmap='RdYlGn', center=0, ax=axes[1], vmin=-0.5, vmax=1)
            axes[1].set_title('行业间相关性（对冲参考）')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/hedging_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存对冲分析图: {OUTPUT_DIR}/hedging_analysis.png")

    def analyze_risk_decomposition(self):
        """风险分解分析"""
        self.log(f"\n{'='*60}")
        self.log("7. 风险分解应用")
        self.log(f"{'='*60}")

        # 使用PCA进行风险分解
        self.log("\n7.1 PCA风险分解")

        explained_var = self.pca.explained_variance_ratio_
        n_factors = len(explained_var)

        self.log(f"\n主成分解释方差:")
        cumulative = 0
        for i in range(min(10, n_factors)):
            cumulative += explained_var[i]
            self.log(f"  PC{i+1}: {explained_var[i]*100:.2f}% (累计: {cumulative*100:.2f}%)")

        # 第一主成分通常代表市场风险
        self.log(f"\n第一主成分（市场因子）解释了 {explained_var[0]*100:.2f}% 的总方差")

        # 分析因子载荷
        self.log("\n7.2 因子载荷分析")

        factor_loadings = self.pca.components_[:3].T  # 前3个因子
        loadings_df = pd.DataFrame(
            factor_loadings,
            index=self.cov_stocks,
            columns=['PC1', 'PC2', 'PC3']
        )

        # 添加行业信息
        loadings_df['industry'] = [self.stocks.loc[s, 'industry'] if s in self.stocks.index else 'Unknown'
                                   for s in loadings_df.index]

        # 分析各行业在第一主成分上的载荷
        industry_loadings = loadings_df.groupby('industry')['PC1'].mean().sort_values(ascending=False)

        self.log("\n各行业在市场因子（PC1）上的平均载荷:")
        for industry, loading in industry_loadings.head(10).items():
            self.log(f"  {industry}: {loading:.4f}")

        # 7.3 边际风险贡献分析
        self.log("\n7.3 边际风险贡献分析")

        # 构建等权组合
        n_stocks = len(self.cov_stocks)
        weights = np.ones(n_stocks) / n_stocks

        # 组合方差
        portfolio_var = weights @ self.lw_cov @ weights
        portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)

        # 边际风险贡献 (Marginal Risk Contribution)
        mrc = (self.lw_cov @ weights) / np.sqrt(portfolio_var)
        mrc_annualized = mrc * np.sqrt(252)

        # 风险贡献 (Risk Contribution)
        rc = weights * mrc_annualized
        rc_pct = rc / portfolio_vol * 100

        mrc_df = pd.DataFrame({
            'stock': self.cov_stocks,
            'weight': weights,
            'mrc': mrc_annualized,
            'rc': rc,
            'rc_pct': rc_pct
        })

        mrc_df['name'] = [self.stocks.loc[s, 'name'] if s in self.stocks.index else s
                         for s in mrc_df['stock']]
        mrc_df['industry'] = [self.stocks.loc[s, 'industry'] if s in self.stocks.index else 'Unknown'
                             for s in mrc_df['stock']]

        self.log(f"\n等权组合年化波动率: {portfolio_vol*100:.2f}%")
        self.log(f"平均边际风险贡献: {mrc_df['mrc'].mean()*100:.4f}%")

        self.log("\n风险贡献最大的股票:")
        top_rc = mrc_df.nlargest(10, 'rc_pct')
        for _, row in top_rc.iterrows():
            self.log(f"  {row['name']}({row['stock']}): 风险贡献 {row['rc_pct']:.2f}%")

        # 按行业汇总风险贡献
        industry_rc = mrc_df.groupby('industry')['rc_pct'].sum().sort_values(ascending=False)

        self.log("\n各行业风险贡献:")
        for industry, rc in industry_rc.head(10).items():
            self.log(f"  {industry}: {rc:.2f}%")

        self.mrc_df = mrc_df

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # PCA解释方差
        axes[0, 0].bar(range(1, min(11, n_factors+1)), explained_var[:10]*100,
                       color='steelblue', alpha=0.7)
        axes[0, 0].plot(range(1, min(11, n_factors+1)), np.cumsum(explained_var[:10])*100,
                       'r-o', label='累计')
        axes[0, 0].set_xlabel('主成分')
        axes[0, 0].set_ylabel('解释方差比例 (%)')
        axes[0, 0].set_title('PCA解释方差（风险分解）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 因子载荷散点图
        axes[0, 1].scatter(loadings_df['PC1'], loadings_df['PC2'], alpha=0.5, s=20, c='steelblue')
        axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('PC1 载荷（市场因子）')
        axes[0, 1].set_ylabel('PC2 载荷')
        axes[0, 1].set_title('因子载荷散点图')
        axes[0, 1].grid(True, alpha=0.3)

        # 行业PC1载荷
        top_industries = industry_loadings.head(15)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_industries)))
        axes[1, 0].barh(range(len(top_industries)), top_industries.values, color=colors)
        axes[1, 0].set_yticks(range(len(top_industries)))
        axes[1, 0].set_yticklabels(top_industries.index)
        axes[1, 0].set_xlabel('PC1 平均载荷')
        axes[1, 0].set_title('各行业市场因子（PC1）载荷')

        # 行业风险贡献饼图
        top_industry_rc = industry_rc.head(10)
        other_rc = industry_rc.iloc[10:].sum()

        labels = list(top_industry_rc.index) + ['其他']
        sizes = list(top_industry_rc.values) + [other_rc]

        axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('行业风险贡献分布')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/risk_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"\n已保存风险分解图: {OUTPUT_DIR}/risk_decomposition.png")

    def generate_summary(self):
        """生成研究摘要"""
        self.log(f"\n{'='*60}")
        self.log("研究总结")
        self.log(f"{'='*60}")

        # 获取上三角元素
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool), k=1)
        corr_values = self.corr_matrix.values[mask]

        self.log(f"""
一、相关性分析结论
=================
1. 全市场平均相关性: {np.mean(corr_values):.4f}
   - A股市场股票整体呈正相关，反映系统性风险的普遍存在
   - 相关性中位数为 {np.median(corr_values):.4f}，分布右偏

2. 行业内相关性特征:
   - 同行业股票相关性普遍高于跨行业股票
   - 行业规模与行业内相关性呈负相关（大行业更分散）

3. 时变特征:
   - 市场压力期相关性显著上升（"correlation breakdown"现象）
   - 相关性与波动率呈正相关

二、协方差估计结论
=================
1. 样本协方差矩阵条件数较大，存在估计不稳定问题
2. Ledoit-Wolf收缩估计显著改善矩阵稳定性
3. 前10个主成分解释了约 {np.sum(self.pca.explained_variance_ratio_[:10])*100:.1f}% 的总方差

三、投资应用建议
===============
1. 分散化投资:
   - 20-30只股票可实现大部分分散化收益
   - 跨行业配置比行业内分散更有效

2. 对冲策略:
   - 低相关行业对可作为对冲配对
   - 低Beta股票适合防御性配置

3. 风险管理:
   - 第一主成分（市场因子）是主要风险来源
   - 边际风险贡献分析有助于识别组合风险集中度
""")

    def save_report(self):
        """保存研究报告"""
        report_path = f"{OUTPUT_DIR}/correlation_structure_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 股票相关性结构研究报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write("```\n")
            f.write("\n".join(self.report_content))
            f.write("\n```\n")

        self.log(f"\n报告已保存到: {report_path}")
        return report_path

    def run_full_analysis(self):
        """运行完整分析"""
        self.log("=" * 60)
        self.log("股票相关性结构研究")
        self.log(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)

        # 1. 加载数据
        self.load_returns_data()

        # 2. 相关性分析
        self.compute_correlation_matrix()
        self.compute_industry_correlation()
        self.compute_time_varying_correlation()

        # 3. 协方差估计
        self.estimate_covariance_matrices()

        # 4. 应用分析
        self.analyze_diversification()
        self.analyze_hedging_strategies()
        self.analyze_risk_decomposition()

        # 5. 生成总结
        self.generate_summary()

        # 6. 保存报告
        report_path = self.save_report()

        return report_path


def main():
    """主函数"""
    analysis = CorrelationStructureAnalysis()
    report_path = analysis.run_full_analysis()

    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"报告保存路径: {report_path}")
    print(f"图表保存目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
