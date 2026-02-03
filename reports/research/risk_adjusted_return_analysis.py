#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股风险调整收益研究分析

研究内容：
1. 风险指标计算：夏普比率、索提诺比率、卡玛比率、最大回撤
2. 风险特征分析：行业风险差异、市值与风险关系、风险的时变性
3. 策略应用：高夏普股票筛选、风险调整选股、组合优化
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# 无风险利率（年化，假设3%）
RISK_FREE_RATE = 0.03
# 交易日数
TRADING_DAYS = 252


class RiskAdjustedReturnAnalyzer:
    """风险调整收益分析器"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.daily_rf = RISK_FREE_RATE / TRADING_DAYS  # 日无风险利率

    def close(self):
        self.conn.close()

    def get_stock_returns(self, start_date='20230101', end_date='20251231', min_days=200):
        """获取股票收益率数据"""
        print(f"获取股票收益率数据: {start_date} - {end_date}")

        query = f"""
        WITH stock_data AS (
            SELECT
                d.ts_code,
                d.trade_date,
                d.pct_chg / 100 as daily_return,
                d.close,
                d.pre_close
            FROM daily d
            JOIN stock_basic sb ON d.ts_code = sb.ts_code
            WHERE d.trade_date >= '{start_date}'
                AND d.trade_date <= '{end_date}'
                AND sb.list_status = 'L'
                AND d.pct_chg IS NOT NULL
                AND d.close > 0
        ),
        stock_counts AS (
            SELECT ts_code, COUNT(*) as cnt
            FROM stock_data
            GROUP BY ts_code
            HAVING COUNT(*) >= {min_days}
        )
        SELECT sd.*
        FROM stock_data sd
        JOIN stock_counts sc ON sd.ts_code = sc.ts_code
        ORDER BY sd.ts_code, sd.trade_date
        """

        df = self.conn.execute(query).fetchdf()
        print(f"获取到 {df['ts_code'].nunique()} 只股票, {len(df)} 条记录")
        return df

    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_risk_metrics(self, df):
        """计算风险指标"""
        print("计算风险指标...")

        results = []
        grouped = df.groupby('ts_code')
        total = len(grouped)

        for i, (ts_code, group) in enumerate(grouped):
            if (i + 1) % 500 == 0:
                print(f"  处理进度: {i+1}/{total}")

            returns = group['daily_return'].values

            # 基础统计
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            # 累计收益率
            total_return = (1 + pd.Series(returns)).prod() - 1

            # 年化收益率
            n_days = len(returns)
            annual_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1

            # 年化波动率
            annual_volatility = std_return * np.sqrt(TRADING_DAYS)

            # 夏普比率 (年化)
            if annual_volatility > 0:
                sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility
            else:
                sharpe_ratio = np.nan

            # 下行标准差（索提诺比率用）
            downside_returns = returns[returns < self.daily_rf]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns, ddof=1) * np.sqrt(TRADING_DAYS)
                if downside_std > 0:
                    sortino_ratio = (annual_return - RISK_FREE_RATE) / downside_std
                else:
                    sortino_ratio = np.nan
            else:
                sortino_ratio = np.nan
                downside_std = 0

            # 最大回撤
            max_drawdown = self.calculate_max_drawdown(pd.Series(returns))

            # 卡玛比率 (Calmar Ratio)
            if max_drawdown < 0:
                calmar_ratio = annual_return / abs(max_drawdown)
            else:
                calmar_ratio = np.nan

            # 偏度和峰度
            skewness = pd.Series(returns).skew()
            kurtosis = pd.Series(returns).kurtosis()

            # 胜率
            win_rate = (returns > 0).sum() / len(returns)

            results.append({
                'ts_code': ts_code,
                'n_days': n_days,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'downside_volatility': downside_std,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'win_rate': win_rate
            })

        risk_df = pd.DataFrame(results)
        print(f"计算完成，共 {len(risk_df)} 只股票")
        return risk_df

    def get_stock_info(self):
        """获取股票基本信息"""
        query = """
        SELECT
            ts_code,
            name,
            industry,
            market
        FROM stock_basic
        WHERE list_status = 'L'
        """
        return self.conn.execute(query).fetchdf()

    def get_industry_info(self):
        """获取申万行业分类"""
        query = """
        SELECT
            ts_code,
            l1_name as sw_l1,
            l2_name as sw_l2
        FROM index_member_all
        WHERE is_new = 'Y'
        """
        return self.conn.execute(query).fetchdf()

    def get_market_cap(self, date='20251230'):
        """获取市值数据"""
        # 找到最近有数据的日期
        query = f"""
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date <= '{date}'
        """
        max_date = self.conn.execute(query).fetchone()[0]

        query = f"""
        SELECT
            ts_code,
            total_mv,
            circ_mv
        FROM daily_basic
        WHERE trade_date = '{max_date}'
            AND total_mv IS NOT NULL
        """
        return self.conn.execute(query).fetchdf()

    def analyze_industry_risk(self, risk_df, industry_df):
        """分析行业风险差异"""
        print("\n=== 行业风险差异分析 ===")

        # 合并数据
        merged = risk_df.merge(industry_df, on='ts_code', how='left')
        merged = merged.dropna(subset=['sw_l1'])

        # 按行业统计
        industry_stats = merged.groupby('sw_l1').agg({
            'sharpe_ratio': ['mean', 'median', 'std', 'count'],
            'sortino_ratio': ['mean', 'median'],
            'calmar_ratio': ['mean', 'median'],
            'max_drawdown': ['mean', 'median'],
            'annual_volatility': ['mean', 'median'],
            'annual_return': ['mean', 'median']
        }).round(4)

        # 扁平化列名
        industry_stats.columns = ['_'.join(col).strip() for col in industry_stats.columns]
        industry_stats = industry_stats.reset_index()
        industry_stats = industry_stats.sort_values('sharpe_ratio_mean', ascending=False)

        print("\n行业夏普比率排名（前10）:")
        print(industry_stats[['sw_l1', 'sharpe_ratio_mean', 'sharpe_ratio_median',
                              'sharpe_ratio_std', 'sharpe_ratio_count']].head(10).to_string(index=False))

        print("\n行业夏普比率排名（后10）:")
        print(industry_stats[['sw_l1', 'sharpe_ratio_mean', 'sharpe_ratio_median',
                              'sharpe_ratio_std', 'sharpe_ratio_count']].tail(10).to_string(index=False))

        return industry_stats, merged

    def analyze_marketcap_risk(self, risk_df, cap_df):
        """分析市值与风险关系"""
        print("\n=== 市值与风险关系分析 ===")

        # 合并数据
        merged = risk_df.merge(cap_df, on='ts_code', how='left')
        merged = merged.dropna(subset=['total_mv'])

        # 市值分组（以亿为单位）
        merged['total_mv_billion'] = merged['total_mv'] / 10000  # 转为亿

        # 定义市值分组
        bins = [0, 30, 100, 300, 1000, float('inf')]
        labels = ['微盘(<30亿)', '小盘(30-100亿)', '中盘(100-300亿)',
                  '大盘(300-1000亿)', '超大盘(>1000亿)']
        merged['cap_group'] = pd.cut(merged['total_mv_billion'], bins=bins, labels=labels)

        # 按市值分组统计
        cap_stats = merged.groupby('cap_group', observed=True).agg({
            'sharpe_ratio': ['mean', 'median', 'std', 'count'],
            'sortino_ratio': ['mean', 'median'],
            'calmar_ratio': ['mean', 'median'],
            'max_drawdown': ['mean', 'median'],
            'annual_volatility': ['mean', 'median'],
            'annual_return': ['mean', 'median']
        }).round(4)

        cap_stats.columns = ['_'.join(col).strip() for col in cap_stats.columns]
        cap_stats = cap_stats.reset_index()

        print("\n市值分组风险指标:")
        print(cap_stats.to_string(index=False))

        # 计算相关性
        corr_sharpe_mv = merged['sharpe_ratio'].corr(merged['total_mv_billion'])
        corr_vol_mv = merged['annual_volatility'].corr(merged['total_mv_billion'])
        corr_mdd_mv = merged['max_drawdown'].corr(merged['total_mv_billion'])

        print(f"\n市值与夏普比率相关性: {corr_sharpe_mv:.4f}")
        print(f"市值与年化波动率相关性: {corr_vol_mv:.4f}")
        print(f"市值与最大回撤相关性: {corr_mdd_mv:.4f}")

        return cap_stats, merged

    def analyze_risk_time_varying(self, start_date='20210101', end_date='20251231'):
        """分析风险的时变性"""
        print("\n=== 风险时变性分析 ===")

        # 按年度计算风险指标
        years = ['2021', '2022', '2023', '2024', '2025']
        yearly_results = []

        for year in years:
            year_start = f"{year}0101"
            year_end = f"{year}1231"

            print(f"  处理 {year} 年...")
            returns_df = self.get_stock_returns(year_start, year_end, min_days=100)

            if len(returns_df) > 0:
                risk_df = self.calculate_risk_metrics(returns_df)

                yearly_results.append({
                    'year': year,
                    'n_stocks': len(risk_df),
                    'sharpe_mean': risk_df['sharpe_ratio'].mean(),
                    'sharpe_median': risk_df['sharpe_ratio'].median(),
                    'sharpe_std': risk_df['sharpe_ratio'].std(),
                    'sortino_mean': risk_df['sortino_ratio'].mean(),
                    'calmar_mean': risk_df['calmar_ratio'].mean(),
                    'volatility_mean': risk_df['annual_volatility'].mean(),
                    'mdd_mean': risk_df['max_drawdown'].mean(),
                    'return_mean': risk_df['annual_return'].mean()
                })

        yearly_df = pd.DataFrame(yearly_results)
        print("\n年度风险指标变化:")
        print(yearly_df.round(4).to_string(index=False))

        return yearly_df

    def select_high_sharpe_stocks(self, risk_df, stock_info, top_n=50):
        """高夏普股票筛选"""
        print("\n=== 高夏普股票筛选 ===")

        # 合并股票信息
        merged = risk_df.merge(stock_info, on='ts_code', how='left')

        # 筛选条件：夏普>0.5, 最大回撤>-50%, 交易天数>=200
        filtered = merged[
            (merged['sharpe_ratio'] > 0.5) &
            (merged['max_drawdown'] > -0.5) &
            (merged['n_days'] >= 200) &
            (merged['annual_return'] > 0)
        ].copy()

        # 按夏普比率排序
        filtered = filtered.sort_values('sharpe_ratio', ascending=False)

        print(f"\n符合条件的股票数量: {len(filtered)}")
        print("\n高夏普比率股票TOP50:")

        top_stocks = filtered.head(top_n)[['ts_code', 'name', 'industry',
                                           'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                                           'annual_return', 'annual_volatility', 'max_drawdown']]
        print(top_stocks.round(4).to_string(index=False))

        return filtered, top_stocks

    def risk_adjusted_stock_selection(self, risk_df, stock_info, industry_df, cap_df):
        """风险调整选股策略"""
        print("\n=== 风险调整选股策略 ===")

        # 合并所有数据
        merged = risk_df.merge(stock_info, on='ts_code', how='left')
        merged = merged.merge(industry_df, on='ts_code', how='left')
        merged = merged.merge(cap_df, on='ts_code', how='left')
        merged['total_mv_billion'] = merged['total_mv'] / 10000

        # 计算综合评分
        # 各指标标准化
        for col in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            merged[f'{col}_zscore'] = (merged[col] - merged[col].mean()) / merged[col].std()

        # 最大回撤越小越好，取负值
        merged['mdd_zscore'] = -(merged['max_drawdown'] - merged['max_drawdown'].mean()) / merged['max_drawdown'].std()

        # 综合评分（等权重）
        merged['composite_score'] = (
            merged['sharpe_ratio_zscore'] * 0.3 +
            merged['sortino_ratio_zscore'] * 0.25 +
            merged['calmar_ratio_zscore'] * 0.25 +
            merged['mdd_zscore'] * 0.2
        )

        # 筛选条件
        valid = merged[
            (merged['n_days'] >= 200) &
            (merged['sharpe_ratio'] > 0) &
            (merged['annual_return'] > 0) &
            (merged['max_drawdown'] > -0.6)
        ].copy()

        valid = valid.sort_values('composite_score', ascending=False)

        print(f"\n符合条件的股票数量: {len(valid)}")

        print("\n综合评分TOP30股票:")
        top_30 = valid.head(30)[['ts_code', 'name', 'sw_l1', 'total_mv_billion',
                                 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                                 'max_drawdown', 'composite_score']]
        print(top_30.round(4).to_string(index=False))

        # 按行业分布
        print("\nTOP100股票行业分布:")
        top_100 = valid.head(100)
        industry_dist = top_100.groupby('sw_l1').size().sort_values(ascending=False)
        print(industry_dist.to_string())

        return valid, top_30

    def portfolio_optimization(self, risk_df, stock_info, n_stocks=20):
        """组合优化"""
        print("\n=== 组合优化 ===")

        # 选择高质量股票池
        merged = risk_df.merge(stock_info, on='ts_code', how='left')

        # 筛选条件
        candidates = merged[
            (merged['sharpe_ratio'] > 0.3) &
            (merged['max_drawdown'] > -0.5) &
            (merged['n_days'] >= 200) &
            (merged['annual_return'] > 0.05)
        ].copy()

        print(f"候选股票池: {len(candidates)} 只")

        # 策略1：等权重高夏普组合
        sharpe_top = candidates.nlargest(n_stocks, 'sharpe_ratio')

        print(f"\n策略1: 等权重高夏普组合 (TOP {n_stocks})")
        print(f"  平均夏普比率: {sharpe_top['sharpe_ratio'].mean():.4f}")
        print(f"  平均年化收益: {sharpe_top['annual_return'].mean():.4f}")
        print(f"  平均年化波动: {sharpe_top['annual_volatility'].mean():.4f}")
        print(f"  平均最大回撤: {sharpe_top['max_drawdown'].mean():.4f}")

        # 策略2：等权重低波动组合
        vol_bottom = candidates.nsmallest(n_stocks, 'annual_volatility')

        print(f"\n策略2: 等权重低波动组合 (波动率最低 {n_stocks})")
        print(f"  平均夏普比率: {vol_bottom['sharpe_ratio'].mean():.4f}")
        print(f"  平均年化收益: {vol_bottom['annual_return'].mean():.4f}")
        print(f"  平均年化波动: {vol_bottom['annual_volatility'].mean():.4f}")
        print(f"  平均最大回撤: {vol_bottom['max_drawdown'].mean():.4f}")

        # 策略3：高卡玛组合（收益/回撤比最优）
        calmar_top = candidates.nlargest(n_stocks, 'calmar_ratio')

        print(f"\n策略3: 等权重高卡玛组合 (TOP {n_stocks})")
        print(f"  平均夏普比率: {calmar_top['sharpe_ratio'].mean():.4f}")
        print(f"  平均年化收益: {calmar_top['annual_return'].mean():.4f}")
        print(f"  平均年化波动: {calmar_top['annual_volatility'].mean():.4f}")
        print(f"  平均最大回撤: {calmar_top['max_drawdown'].mean():.4f}")

        # 策略4：综合优化（多目标平衡）
        candidates['opt_score'] = (
            candidates['sharpe_ratio'] * 0.4 -
            candidates['annual_volatility'] * 0.3 +
            candidates['calmar_ratio'].clip(-2, 2) * 0.3
        )
        opt_top = candidates.nlargest(n_stocks, 'opt_score')

        print(f"\n策略4: 多目标优化组合 (TOP {n_stocks})")
        print(f"  平均夏普比率: {opt_top['sharpe_ratio'].mean():.4f}")
        print(f"  平均年化收益: {opt_top['annual_return'].mean():.4f}")
        print(f"  平均年化波动: {opt_top['annual_volatility'].mean():.4f}")
        print(f"  平均最大回撤: {opt_top['max_drawdown'].mean():.4f}")

        print("\n高夏普组合成分股:")
        print(sharpe_top[['ts_code', 'name', 'industry', 'sharpe_ratio',
                          'annual_return', 'max_drawdown']].round(4).to_string(index=False))

        return {
            'high_sharpe': sharpe_top,
            'low_volatility': vol_bottom,
            'high_calmar': calmar_top,
            'optimized': opt_top
        }

    def plot_risk_analysis(self, risk_df, industry_merged, cap_merged, yearly_df):
        """绘制风险分析图表"""
        print("\n绘制分析图表...")

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # 1. 夏普比率分布
        ax1 = axes[0, 0]
        valid_sharpe = risk_df['sharpe_ratio'].dropna()
        valid_sharpe = valid_sharpe[(valid_sharpe > -5) & (valid_sharpe < 5)]
        ax1.hist(valid_sharpe, bins=100, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', label='零线')
        ax1.axvline(x=valid_sharpe.median(), color='green', linestyle='--',
                    label=f'中位数: {valid_sharpe.median():.2f}')
        ax1.set_xlabel('夏普比率')
        ax1.set_ylabel('股票数量')
        ax1.set_title('夏普比率分布')
        ax1.legend()

        # 2. 索提诺比率分布
        ax2 = axes[0, 1]
        valid_sortino = risk_df['sortino_ratio'].dropna()
        valid_sortino = valid_sortino[(valid_sortino > -10) & (valid_sortino < 10)]
        ax2.hist(valid_sortino, bins=100, edgecolor='black', alpha=0.7, color='orange')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.axvline(x=valid_sortino.median(), color='green', linestyle='--',
                    label=f'中位数: {valid_sortino.median():.2f}')
        ax2.set_xlabel('索提诺比率')
        ax2.set_ylabel('股票数量')
        ax2.set_title('索提诺比率分布')
        ax2.legend()

        # 3. 最大回撤分布
        ax3 = axes[0, 2]
        valid_mdd = risk_df['max_drawdown'].dropna()
        ax3.hist(valid_mdd, bins=100, edgecolor='black', alpha=0.7, color='red')
        ax3.axvline(x=valid_mdd.median(), color='blue', linestyle='--',
                    label=f'中位数: {valid_mdd.median():.2%}')
        ax3.set_xlabel('最大回撤')
        ax3.set_ylabel('股票数量')
        ax3.set_title('最大回撤分布')
        ax3.legend()

        # 4. 行业夏普比率对比
        ax4 = axes[1, 0]
        industry_sharpe = industry_merged.groupby('sw_l1')['sharpe_ratio'].mean().sort_values(ascending=True)
        industry_sharpe = industry_sharpe.tail(15)  # 取后15个（最高的）
        colors = ['green' if x > 0 else 'red' for x in industry_sharpe.values]
        ax4.barh(industry_sharpe.index, industry_sharpe.values, color=colors, alpha=0.7)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('平均夏普比率')
        ax4.set_title('行业平均夏普比率 (TOP15)')

        # 5. 市值与夏普比率散点图
        ax5 = axes[1, 1]
        cap_data = cap_merged[(cap_merged['total_mv_billion'] < 2000) &
                              (cap_merged['sharpe_ratio'].between(-3, 3))].copy()
        scatter = ax5.scatter(cap_data['total_mv_billion'], cap_data['sharpe_ratio'],
                             alpha=0.3, s=10, c=cap_data['annual_return'], cmap='RdYlGn')
        ax5.set_xlabel('总市值 (亿元)')
        ax5.set_ylabel('夏普比率')
        ax5.set_title('市值与夏普比率关系')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax5, label='年化收益率')

        # 6. 市值分组夏普比率箱线图
        ax6 = axes[1, 2]
        cap_merged_valid = cap_merged[cap_merged['sharpe_ratio'].between(-5, 5)].copy()
        cap_groups = cap_merged_valid.groupby('cap_group', observed=True)['sharpe_ratio']
        group_data = [group.values for name, group in cap_groups]
        group_names = [name for name, group in cap_groups]
        bp = ax6.boxplot(group_data, labels=group_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax6.set_ylabel('夏普比率')
        ax6.set_title('市值分组夏普比率分布')
        ax6.tick_params(axis='x', rotation=45)

        # 7. 年度风险指标变化
        ax7 = axes[2, 0]
        ax7.plot(yearly_df['year'], yearly_df['sharpe_mean'], 'o-', label='夏普均值', linewidth=2)
        ax7.plot(yearly_df['year'], yearly_df['sharpe_median'], 's--', label='夏普中位数', linewidth=2)
        ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax7.set_xlabel('年份')
        ax7.set_ylabel('夏普比率')
        ax7.set_title('年度夏普比率变化')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 年度波动率变化
        ax8 = axes[2, 1]
        ax8.bar(yearly_df['year'], yearly_df['volatility_mean'], alpha=0.7, color='orange')
        ax8.set_xlabel('年份')
        ax8.set_ylabel('平均年化波动率')
        ax8.set_title('年度平均波动率变化')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. 收益与风险的关系
        ax9 = axes[2, 2]
        valid_data = risk_df[(risk_df['annual_volatility'] < 2) &
                             (risk_df['annual_return'].between(-1, 2))].copy()
        scatter2 = ax9.scatter(valid_data['annual_volatility'], valid_data['annual_return'],
                               alpha=0.3, s=10, c=valid_data['sharpe_ratio'],
                               cmap='RdYlGn', vmin=-1, vmax=1)
        ax9.set_xlabel('年化波动率')
        ax9.set_ylabel('年化收益率')
        ax9.set_title('收益-风险散点图')
        ax9.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax9.axhline(y=RISK_FREE_RATE, color='blue', linestyle='--', alpha=0.5, label='无风险利率')
        plt.colorbar(scatter2, ax=ax9, label='夏普比率')

        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}/risk_adjusted_return_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {REPORT_DIR}/risk_adjusted_return_analysis.png")


def generate_report(results):
    """生成分析报告"""
    report = """
# 个股风险调整收益研究报告

## 研究概述

本报告对A股市场个股进行全面的风险调整收益分析，包括：
1. 风险指标计算：夏普比率、索提诺比率、卡玛比率、最大回撤
2. 风险特征分析：行业差异、市值关系、时变性
3. 策略应用：股票筛选与组合优化

---

## 一、风险指标说明

### 1. 夏普比率 (Sharpe Ratio)
- **定义**: (年化收益率 - 无风险利率) / 年化波动率
- **解释**: 衡量单位风险所获得的超额收益
- **标准**: >1 良好, >2 优秀, <0 表现不佳

### 2. 索提诺比率 (Sortino Ratio)
- **定义**: (年化收益率 - 无风险利率) / 下行波动率
- **解释**: 只考虑下行风险的风险调整收益
- **特点**: 不惩罚上行波动，更合理评估投资表现

### 3. 卡玛比率 (Calmar Ratio)
- **定义**: 年化收益率 / |最大回撤|
- **解释**: 收益与最大回撤的比值
- **特点**: 关注极端风险，适合评估长期投资

### 4. 最大回撤 (Maximum Drawdown)
- **定义**: 从历史最高点到最低点的最大跌幅
- **解释**: 衡量最坏情况下的损失

---

## 二、风险指标统计

"""

    # 添加整体统计
    risk_df = results['risk_metrics']
    report += "### 整体统计\n\n"
    report += f"| 指标 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |\n"
    report += f"|------|------|--------|--------|--------|--------|\n"

    for col, name in [('sharpe_ratio', '夏普比率'),
                      ('sortino_ratio', '索提诺比率'),
                      ('calmar_ratio', '卡玛比率'),
                      ('max_drawdown', '最大回撤'),
                      ('annual_volatility', '年化波动率'),
                      ('annual_return', '年化收益率')]:
        data = risk_df[col].dropna()
        # 对于极端值进行裁剪
        if col in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            data = data[(data > -10) & (data < 10)]
        report += f"| {name} | {data.mean():.4f} | {data.median():.4f} | {data.std():.4f} | {data.min():.4f} | {data.max():.4f} |\n"

    report += "\n"

    # 添加行业分析
    report += "---\n\n## 三、行业风险差异分析\n\n"

    industry_stats = results['industry_stats']
    report += "### 行业夏普比率排名\n\n"
    report += "| 行业 | 夏普均值 | 夏普中位数 | 标准差 | 股票数 |\n"
    report += "|------|----------|------------|--------|--------|\n"

    for _, row in industry_stats.head(15).iterrows():
        report += f"| {row['sw_l1']} | {row['sharpe_ratio_mean']:.4f} | {row['sharpe_ratio_median']:.4f} | {row['sharpe_ratio_std']:.4f} | {int(row['sharpe_ratio_count'])} |\n"

    report += "\n### 主要发现\n\n"
    top_industry = industry_stats.iloc[0]['sw_l1']
    bottom_industry = industry_stats.iloc[-1]['sw_l1']
    report += f"- **最高风险调整收益行业**: {top_industry} (夏普均值: {industry_stats.iloc[0]['sharpe_ratio_mean']:.4f})\n"
    report += f"- **最低风险调整收益行业**: {bottom_industry} (夏普均值: {industry_stats.iloc[-1]['sharpe_ratio_mean']:.4f})\n"
    report += f"- 行业间夏普比率差异显著，需要考虑行业配置\n\n"

    # 添加市值分析
    report += "---\n\n## 四、市值与风险关系分析\n\n"

    cap_stats = results['cap_stats']
    report += "### 市值分组风险指标\n\n"
    report += "| 市值分组 | 夏普均值 | 夏普中位数 | 波动率均值 | 最大回撤均值 | 股票数 |\n"
    report += "|----------|----------|------------|------------|--------------|--------|\n"

    for _, row in cap_stats.iterrows():
        report += f"| {row['cap_group']} | {row['sharpe_ratio_mean']:.4f} | {row['sharpe_ratio_median']:.4f} | {row['annual_volatility_mean']:.4f} | {row['max_drawdown_mean']:.4f} | {int(row['sharpe_ratio_count'])} |\n"

    report += "\n### 主要发现\n\n"
    report += "- 小市值股票通常波动率更高，但夏普比率不一定更低\n"
    report += "- 大市值股票波动较低，但收益也相对稳定\n"
    report += f"- 市值与夏普比率相关性: {results.get('corr_sharpe_mv', 0):.4f}\n\n"

    # 添加时变性分析
    report += "---\n\n## 五、风险时变性分析\n\n"

    yearly_df = results['yearly_stats']
    report += "### 年度风险指标变化\n\n"
    report += "| 年份 | 股票数 | 夏普均值 | 夏普中位数 | 波动率均值 | 最大回撤均值 | 收益率均值 |\n"
    report += "|------|--------|----------|------------|------------|--------------|------------|\n"

    for _, row in yearly_df.iterrows():
        report += f"| {row['year']} | {int(row['n_stocks'])} | {row['sharpe_mean']:.4f} | {row['sharpe_median']:.4f} | {row['volatility_mean']:.4f} | {row['mdd_mean']:.4f} | {row['return_mean']:.4f} |\n"

    report += "\n### 主要发现\n\n"
    best_year = yearly_df.loc[yearly_df['sharpe_mean'].idxmax()]
    worst_year = yearly_df.loc[yearly_df['sharpe_mean'].idxmin()]
    report += f"- **最佳年份**: {best_year['year']} (夏普均值: {best_year['sharpe_mean']:.4f})\n"
    report += f"- **最差年份**: {worst_year['year']} (夏普均值: {worst_year['sharpe_mean']:.4f})\n"
    report += "- 风险指标具有明显的时变特征，需动态调整策略\n\n"

    # 添加策略应用
    report += "---\n\n## 六、策略应用\n\n"

    report += "### 1. 高夏普股票筛选\n\n"
    report += "筛选条件:\n"
    report += "- 夏普比率 > 0.5\n"
    report += "- 最大回撤 > -50%\n"
    report += "- 交易天数 >= 200\n"
    report += "- 年化收益率 > 0\n\n"

    top_stocks = results['top_sharpe_stocks']
    report += f"符合条件股票数: {len(results['filtered_stocks'])}\n\n"
    report += "**TOP20 高夏普股票**:\n\n"
    report += "| 代码 | 名称 | 行业 | 夏普 | 索提诺 | 卡玛 | 年化收益 | 最大回撤 |\n"
    report += "|------|------|------|------|--------|------|----------|----------|\n"

    for _, row in top_stocks.head(20).iterrows():
        report += f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['sharpe_ratio']:.3f} | {row['sortino_ratio']:.3f} | {row['calmar_ratio']:.3f} | {row['annual_return']:.2%} | {row['max_drawdown']:.2%} |\n"

    report += "\n### 2. 风险调整综合选股\n\n"
    report += "评分公式:\n"
    report += "- 夏普比率Z分数 * 0.30\n"
    report += "- 索提诺比率Z分数 * 0.25\n"
    report += "- 卡玛比率Z分数 * 0.25\n"
    report += "- 最大回撤Z分数(反向) * 0.20\n\n"

    top_30 = results['composite_top30']
    report += "**综合评分TOP20股票**:\n\n"
    report += "| 代码 | 名称 | 行业 | 市值(亿) | 夏普 | 索提诺 | 卡玛 | 最大回撤 | 综合评分 |\n"
    report += "|------|------|------|----------|------|--------|------|----------|----------|\n"

    for _, row in top_30.head(20).iterrows():
        mv = row.get('total_mv_billion', 0)
        mv_str = f"{mv:.1f}" if pd.notna(mv) else "N/A"
        report += f"| {row['ts_code']} | {row['name']} | {row.get('sw_l1', 'N/A')} | {mv_str} | {row['sharpe_ratio']:.3f} | {row['sortino_ratio']:.3f} | {row['calmar_ratio']:.3f} | {row['max_drawdown']:.2%} | {row['composite_score']:.3f} |\n"

    report += "\n### 3. 组合优化策略对比\n\n"

    portfolios = results['portfolios']
    report += "| 策略 | 平均夏普 | 平均年化收益 | 平均波动率 | 平均最大回撤 |\n"
    report += "|------|----------|--------------|------------|---------------|\n"

    for name, label in [('high_sharpe', '高夏普组合'),
                        ('low_volatility', '低波动组合'),
                        ('high_calmar', '高卡玛组合'),
                        ('optimized', '多目标优化组合')]:
        pf = portfolios[name]
        report += f"| {label} | {pf['sharpe_ratio'].mean():.3f} | {pf['annual_return'].mean():.2%} | {pf['annual_volatility'].mean():.2%} | {pf['max_drawdown'].mean():.2%} |\n"

    report += "\n### 4. 高夏普组合成分股\n\n"
    report += "| 代码 | 名称 | 行业 | 夏普比率 | 年化收益 | 最大回撤 |\n"
    report += "|------|------|------|----------|----------|----------|\n"

    for _, row in portfolios['high_sharpe'].iterrows():
        report += f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['sharpe_ratio']:.3f} | {row['annual_return']:.2%} | {row['max_drawdown']:.2%} |\n"

    report += """

---

## 七、研究结论与建议

### 核心发现

1. **风险指标分布特征**
   - A股市场整体夏普比率分布偏左，多数股票风险调整收益不佳
   - 索提诺比率相对夏普比率更能反映真实的投资价值
   - 最大回撤普遍较大，反映A股市场的高波动特性

2. **行业差异显著**
   - 不同行业的风险调整收益差异明显
   - 防守型行业(如银行、公用事业)通常波动较低
   - 科技成长型行业波动大，但部分个股夏普比率较高

3. **市值效应**
   - 小市值股票波动率普遍更高
   - 大市值股票夏普比率相对稳定
   - 中等市值股票可能具有较好的风险收益比

4. **时变性特征**
   - 风险指标呈现明显的周期性
   - 牛市期间夏普比率普遍较高，熊市则相反
   - 需要根据市场环境动态调整策略

### 投资建议

1. **选股策略**
   - 优先选择高夏普比率且最大回撤可控的股票
   - 结合索提诺比率避免"假高收益"陷阱
   - 关注卡玛比率评估长期投资价值

2. **组合配置**
   - 采用多目标优化平衡收益与风险
   - 考虑行业分散降低非系统性风险
   - 市值配置应兼顾流动性与收益

3. **风险管理**
   - 设置最大回撤止损线
   - 定期重新评估风险指标
   - 根据市场环境调整风险敞口

---

## 附录

### 计算参数
- 无风险利率: 3% (年化)
- 交易日数: 252天/年
- 分析周期: 2023-2025年
- 最小交易天数: 200天

### 数据说明
- 数据来源: Tushare
- 收益率: 使用每日涨跌幅(pct_chg)
- 行业分类: 申万一级行业

---

*报告生成时间: {datetime}*
""".format(datetime=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return report


def main():
    """主函数"""
    print("=" * 60)
    print("个股风险调整收益研究分析")
    print("=" * 60)

    analyzer = RiskAdjustedReturnAnalyzer()

    try:
        # 1. 获取数据
        print("\n[1/7] 获取收益率数据...")
        returns_df = analyzer.get_stock_returns('20230101', '20251231', min_days=200)

        # 2. 计算风险指标
        print("\n[2/7] 计算风险指标...")
        risk_df = analyzer.calculate_risk_metrics(returns_df)

        # 3. 获取辅助数据
        print("\n[3/7] 获取辅助数据...")
        stock_info = analyzer.get_stock_info()
        industry_df = analyzer.get_industry_info()
        cap_df = analyzer.get_market_cap()

        # 4. 行业分析
        print("\n[4/7] 行业风险分析...")
        industry_stats, industry_merged = analyzer.analyze_industry_risk(risk_df, industry_df)

        # 5. 市值分析
        print("\n[5/7] 市值与风险分析...")
        cap_stats, cap_merged = analyzer.analyze_marketcap_risk(risk_df, cap_df)

        # 6. 时变性分析
        print("\n[6/7] 风险时变性分析...")
        yearly_df = analyzer.analyze_risk_time_varying()

        # 7. 策略应用
        print("\n[7/7] 策略应用分析...")
        filtered_stocks, top_stocks = analyzer.select_high_sharpe_stocks(risk_df, stock_info)
        composite_valid, composite_top30 = analyzer.risk_adjusted_stock_selection(
            risk_df, stock_info, industry_df, cap_df)
        portfolios = analyzer.portfolio_optimization(risk_df, stock_info)

        # 绘制图表
        analyzer.plot_risk_analysis(risk_df, industry_merged, cap_merged, yearly_df)

        # 汇总结果
        results = {
            'risk_metrics': risk_df,
            'industry_stats': industry_stats,
            'cap_stats': cap_stats,
            'yearly_stats': yearly_df,
            'filtered_stocks': filtered_stocks,
            'top_sharpe_stocks': top_stocks,
            'composite_top30': composite_top30,
            'portfolios': portfolios,
            'corr_sharpe_mv': cap_merged['sharpe_ratio'].corr(cap_merged['total_mv_billion'])
        }

        # 生成报告
        print("\n生成研究报告...")
        report = generate_report(results)

        report_path = f'{REPORT_DIR}/risk_adjusted_return_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存: {report_path}")

        # 保存数据
        risk_df.to_csv(f'{REPORT_DIR}/risk_metrics_data.csv', index=False)
        print(f"风险指标数据已保存: {REPORT_DIR}/risk_metrics_data.csv")

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
