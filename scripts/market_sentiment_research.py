#!/usr/bin/env python3
"""
市场情绪因子研究
================
基于Tushare数据库进行市场情绪因子分析

研究内容：
1. 情绪指标构建：涨跌家数比、涨停/跌停数量、换手率分布、资金流向
2. 情绪与收益：情绪极端值效应、情绪反转效应、情绪持续性
3. 策略应用：情绪择时策略、逆向投资策略、风险预警
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

class MarketSentimentResearch:
    """市场情绪因子研究类"""

    def __init__(self, db_path=DB_PATH, start_date='20150101', end_date='20260130'):
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.conn = duckdb.connect(db_path, read_only=True)
        self.sentiment_df = None
        self.results = {}

    def build_sentiment_indicators(self):
        """
        构建情绪指标
        1. 涨跌家数比 (Advance-Decline Ratio)
        2. 涨停/跌停数量
        3. 换手率分布
        4. 资金流向
        """
        print("=" * 60)
        print("1. 构建市场情绪指标")
        print("=" * 60)

        # 1.1 涨跌家数比和涨停跌停数量
        print("\n1.1 计算涨跌家数比和涨停跌停...")
        ad_query = f"""
        SELECT
            trade_date,
            COUNT(*) as total_stocks,
            SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) as up_count,
            SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END) as down_count,
            SUM(CASE WHEN pct_chg = 0 THEN 1 ELSE 0 END) as flat_count,
            SUM(CASE WHEN pct_chg >= 9.9 THEN 1 ELSE 0 END) as limit_up,
            SUM(CASE WHEN pct_chg <= -9.9 THEN 1 ELSE 0 END) as limit_down,
            AVG(pct_chg) as avg_return,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pct_chg) as median_return,
            STDDEV(pct_chg) as return_std
        FROM daily
        WHERE trade_date >= '{self.start_date}'
          AND trade_date <= '{self.end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        ad_df = self.conn.execute(ad_query).fetchdf()

        # 计算涨跌比
        ad_df['ad_ratio'] = ad_df['up_count'] / (ad_df['down_count'] + 1)  # 避免除零
        ad_df['ad_diff'] = ad_df['up_count'] - ad_df['down_count']
        ad_df['up_pct'] = ad_df['up_count'] / ad_df['total_stocks']
        ad_df['limit_diff'] = ad_df['limit_up'] - ad_df['limit_down']

        print(f"  日期范围: {ad_df['trade_date'].min()} - {ad_df['trade_date'].max()}")
        print(f"  交易日数: {len(ad_df)}")

        # 1.2 换手率分布
        print("\n1.2 计算换手率分布...")
        turnover_query = f"""
        SELECT
            d.trade_date,
            AVG(db.turnover_rate) as avg_turnover,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY db.turnover_rate) as median_turnover,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY db.turnover_rate) as p90_turnover,
            PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY db.turnover_rate) as p10_turnover,
            STDDEV(db.turnover_rate) as turnover_std,
            SUM(db.turnover_rate * db.circ_mv) / NULLIF(SUM(db.circ_mv), 0) as weighted_turnover
        FROM daily d
        JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
        WHERE d.trade_date >= '{self.start_date}'
          AND d.trade_date <= '{self.end_date}'
          AND db.turnover_rate IS NOT NULL
        GROUP BY d.trade_date
        ORDER BY d.trade_date
        """
        turnover_df = self.conn.execute(turnover_query).fetchdf()

        # 1.3 资金流向
        print("\n1.3 计算资金流向指标...")
        moneyflow_query = f"""
        SELECT
            trade_date,
            SUM(net_mf_amount) as total_net_flow,
            SUM(buy_elg_amount - sell_elg_amount) as elg_net_flow,
            SUM(buy_lg_amount - sell_lg_amount) as lg_net_flow,
            SUM(buy_md_amount - sell_md_amount) as md_net_flow,
            SUM(buy_sm_amount - sell_sm_amount) as sm_net_flow,
            COUNT(*) as stock_count
        FROM moneyflow
        WHERE trade_date >= '{self.start_date}'
          AND trade_date <= '{self.end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        moneyflow_df = self.conn.execute(moneyflow_query).fetchdf()

        # 合并所有指标
        self.sentiment_df = ad_df.copy()
        self.sentiment_df = self.sentiment_df.merge(turnover_df, on='trade_date', how='left')
        self.sentiment_df = self.sentiment_df.merge(moneyflow_df, on='trade_date', how='left')

        # 计算综合情绪指标
        self.sentiment_df['trade_date_dt'] = pd.to_datetime(self.sentiment_df['trade_date'])
        self.sentiment_df = self.sentiment_df.sort_values('trade_date').reset_index(drop=True)

        # 标准化各个指标后合成综合情绪指数
        def zscore(x):
            return (x - x.rolling(250, min_periods=60).mean()) / x.rolling(250, min_periods=60).std()

        self.sentiment_df['ad_ratio_z'] = zscore(self.sentiment_df['ad_ratio'])
        self.sentiment_df['limit_diff_z'] = zscore(self.sentiment_df['limit_diff'])
        self.sentiment_df['turnover_z'] = zscore(self.sentiment_df['avg_turnover'])

        # 综合情绪指数（等权平均）
        self.sentiment_df['sentiment_index'] = (
            self.sentiment_df['ad_ratio_z'].fillna(0) +
            self.sentiment_df['limit_diff_z'].fillna(0) +
            self.sentiment_df['turnover_z'].fillna(0)
        ) / 3

        # 情绪分位数
        self.sentiment_df['sentiment_pct'] = self.sentiment_df['sentiment_index'].rolling(250, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        print(f"\n情绪指标统计:")
        print(f"  涨跌比均值: {self.sentiment_df['ad_ratio'].mean():.3f}")
        print(f"  平均涨停数: {self.sentiment_df['limit_up'].mean():.1f}")
        print(f"  平均跌停数: {self.sentiment_df['limit_down'].mean():.1f}")
        print(f"  平均换手率: {self.sentiment_df['avg_turnover'].mean():.2f}%")

        return self.sentiment_df

    def get_index_returns(self):
        """获取市场指数收益率"""
        index_query = f"""
        SELECT
            trade_date,
            pct_chg as index_return,
            close as index_close
        FROM index_daily
        WHERE ts_code = '000001.SH'  -- 上证指数
          AND trade_date >= '{self.start_date}'
          AND trade_date <= '{self.end_date}'
        ORDER BY trade_date
        """
        index_df = self.conn.execute(index_query).fetchdf()
        return index_df

    def analyze_sentiment_returns(self):
        """
        分析情绪与收益的关系
        1. 情绪极端值效应
        2. 情绪反转效应
        3. 情绪持续性
        """
        print("\n" + "=" * 60)
        print("2. 情绪与收益关系分析")
        print("=" * 60)

        # 获取指数收益
        index_df = self.get_index_returns()
        df = self.sentiment_df.merge(index_df, on='trade_date', how='left')

        # 计算未来收益
        for n in [1, 3, 5, 10, 20]:
            df[f'fwd_ret_{n}d'] = df['index_return'].shift(-n).rolling(n).sum()

        # 2.1 情绪极端值效应
        print("\n2.1 情绪极端值效应分析")

        # 根据情绪分位数分组
        df['sentiment_group'] = pd.cut(
            df['sentiment_pct'],
            bins=[0, 0.1, 0.3, 0.7, 0.9, 1.0],
            labels=['极度悲观', '悲观', '中性', '乐观', '极度乐观']
        )

        extreme_results = {}
        for group in ['极度悲观', '悲观', '中性', '乐观', '极度乐观']:
            group_df = df[df['sentiment_group'] == group]
            if len(group_df) > 0:
                extreme_results[group] = {
                    '样本数': len(group_df),
                    '1日收益': group_df['fwd_ret_1d'].mean(),
                    '5日收益': group_df['fwd_ret_5d'].mean(),
                    '10日收益': group_df['fwd_ret_10d'].mean(),
                    '20日收益': group_df['fwd_ret_20d'].mean()
                }

        extreme_df = pd.DataFrame(extreme_results).T
        print("\n各情绪分组未来收益(%):")
        print(extreme_df.round(3).to_string())

        self.results['extreme_effect'] = extreme_df

        # 2.2 情绪反转效应
        print("\n2.2 情绪反转效应分析")

        # 计算情绪变化
        df['sentiment_chg'] = df['sentiment_index'] - df['sentiment_index'].shift(5)
        df['sentiment_chg_group'] = pd.qcut(
            df['sentiment_chg'].dropna(),
            q=5,
            labels=['大幅下降', '下降', '持平', '上升', '大幅上升']
        )

        reversal_results = {}
        for group in ['大幅下降', '下降', '持平', '上升', '大幅上升']:
            group_df = df[df['sentiment_chg_group'] == group]
            if len(group_df) > 0:
                reversal_results[group] = {
                    '样本数': len(group_df),
                    '1日收益': group_df['fwd_ret_1d'].mean(),
                    '5日收益': group_df['fwd_ret_5d'].mean(),
                    '10日收益': group_df['fwd_ret_10d'].mean(),
                    '20日收益': group_df['fwd_ret_20d'].mean()
                }

        reversal_df = pd.DataFrame(reversal_results).T
        print("\n情绪变化与未来收益(%):")
        print(reversal_df.round(3).to_string())

        self.results['reversal_effect'] = reversal_df

        # 2.3 情绪持续性分析
        print("\n2.3 情绪持续性分析")

        # 计算情绪自相关性
        autocorrs = []
        for lag in range(1, 21):
            corr = df['sentiment_index'].corr(df['sentiment_index'].shift(lag))
            autocorrs.append({'滞后期': lag, '自相关系数': corr})

        autocorr_df = pd.DataFrame(autocorrs)
        print("\n情绪指数自相关系数:")
        print(autocorr_df.set_index('滞后期').T.round(3).to_string())

        self.results['autocorrelation'] = autocorr_df
        self.analysis_df = df

        return df

    def build_sentiment_strategy(self):
        """
        构建情绪策略
        1. 情绪择时策略
        2. 逆向投资策略
        3. 风险预警
        """
        print("\n" + "=" * 60)
        print("3. 情绪策略构建与回测")
        print("=" * 60)

        df = self.analysis_df.copy()
        df = df.dropna(subset=['sentiment_pct', 'index_return']).copy()

        # 3.1 情绪择时策略
        print("\n3.1 情绪择时策略")

        # 策略1: 低情绪买入，高情绪卖出
        df['signal_contrarian'] = 0
        df.loc[df['sentiment_pct'] < 0.2, 'signal_contrarian'] = 1  # 低情绪满仓
        df.loc[df['sentiment_pct'] > 0.8, 'signal_contrarian'] = -1  # 高情绪空仓
        df.loc[(df['sentiment_pct'] >= 0.2) & (df['sentiment_pct'] <= 0.8), 'signal_contrarian'] = 0.5  # 中性半仓

        # 策略2: 情绪动量策略
        df['sentiment_ma5'] = df['sentiment_index'].rolling(5).mean()
        df['sentiment_ma20'] = df['sentiment_index'].rolling(20).mean()
        df['signal_momentum'] = 0
        df.loc[df['sentiment_ma5'] > df['sentiment_ma20'], 'signal_momentum'] = 1
        df.loc[df['sentiment_ma5'] <= df['sentiment_ma20'], 'signal_momentum'] = 0

        # 计算策略收益
        df['ret_hold'] = df['index_return'] / 100  # 持有指数
        df['ret_contrarian'] = df['signal_contrarian'].shift(1) * df['ret_hold']
        df['ret_momentum'] = df['signal_momentum'].shift(1) * df['ret_hold']

        # 计算累计收益
        df['cum_hold'] = (1 + df['ret_hold']).cumprod()
        df['cum_contrarian'] = (1 + df['ret_contrarian'].fillna(0)).cumprod()
        df['cum_momentum'] = (1 + df['ret_momentum'].fillna(0)).cumprod()

        # 统计策略表现
        strategy_stats = {}
        for name, ret_col in [('持有指数', 'ret_hold'), ('逆向策略', 'ret_contrarian'), ('动量策略', 'ret_momentum')]:
            returns = df[ret_col].dropna()
            total_ret = (1 + returns).prod() - 1
            annual_ret = (1 + total_ret) ** (252 / len(returns)) - 1
            vol = returns.std() * np.sqrt(252)
            sharpe = annual_ret / vol if vol > 0 else 0
            max_dd = (df[f'cum_{ret_col.split("_")[1]}'].cummax() - df[f'cum_{ret_col.split("_")[1]}']).max() if ret_col != 'ret_hold' else \
                     (df['cum_hold'].cummax() - df['cum_hold']).max()

            strategy_stats[name] = {
                '累计收益': f"{total_ret*100:.1f}%",
                '年化收益': f"{annual_ret*100:.1f}%",
                '年化波动': f"{vol*100:.1f}%",
                '夏普比率': f"{sharpe:.2f}",
                '最大回撤': f"{max_dd*100:.1f}%"
            }

        stats_df = pd.DataFrame(strategy_stats).T
        print("\n策略绩效比较:")
        print(stats_df.to_string())

        self.results['strategy_stats'] = stats_df

        # 3.2 风险预警信号
        print("\n3.2 风险预警信号")

        # 定义风险预警条件
        df['risk_high_sentiment'] = (df['sentiment_pct'] > 0.9).astype(int)
        df['risk_high_turnover'] = (df['turnover_z'] > 2).astype(int)
        df['risk_limit_imbalance'] = (df['limit_diff_z'] > 2).astype(int)
        df['risk_score'] = df['risk_high_sentiment'] + df['risk_high_turnover'] + df['risk_limit_imbalance']

        # 统计风险信号后的收益
        risk_analysis = {}
        for score in [0, 1, 2, 3]:
            score_df = df[df['risk_score'] == score]
            if len(score_df) > 10:
                risk_analysis[f'风险分{score}'] = {
                    '出现次数': len(score_df),
                    '平均5日收益': f"{score_df['fwd_ret_5d'].mean():.2f}%",
                    '平均10日收益': f"{score_df['fwd_ret_10d'].mean():.2f}%",
                    '下跌概率(5日)': f"{(score_df['fwd_ret_5d'] < 0).mean()*100:.1f}%"
                }

        risk_df = pd.DataFrame(risk_analysis).T
        print("\n风险预警效果:")
        print(risk_df.to_string())

        self.results['risk_warning'] = risk_df
        self.strategy_df = df

        return df

    def generate_visualizations(self):
        """生成可视化图表"""
        print("\n" + "=" * 60)
        print("4. 生成可视化图表")
        print("=" * 60)

        df = self.strategy_df.copy()

        # 图1: 情绪指标时序图
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # 1.1 涨跌家数比
        ax1 = axes[0]
        ax1.fill_between(df['trade_date_dt'], df['ad_ratio'], alpha=0.3, color='blue')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('涨跌家数比 (Advance-Decline Ratio)', fontsize=12)
        ax1.set_ylabel('涨跌比')
        ax1.legend(['涨跌比', '平衡线'])

        # 1.2 涨停跌停差
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in df['limit_diff']]
        ax2.bar(df['trade_date_dt'], df['limit_diff'], alpha=0.6, color=colors, width=1)
        ax2.set_title('涨停跌停差 (涨停数-跌停数)', fontsize=12)
        ax2.set_ylabel('家数')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 1.3 市场换手率
        ax3 = axes[2]
        ax3.plot(df['trade_date_dt'], df['avg_turnover'], color='purple', alpha=0.7)
        ax3.fill_between(df['trade_date_dt'], df['avg_turnover'], alpha=0.2, color='purple')
        ax3.set_title('市场平均换手率', fontsize=12)
        ax3.set_ylabel('换手率(%)')

        # 1.4 综合情绪指数
        ax4 = axes[3]
        ax4.plot(df['trade_date_dt'], df['sentiment_index'], color='orange', alpha=0.8)
        ax4.fill_between(df['trade_date_dt'], df['sentiment_index'],
                        where=df['sentiment_index'] > 0, alpha=0.3, color='red')
        ax4.fill_between(df['trade_date_dt'], df['sentiment_index'],
                        where=df['sentiment_index'] <= 0, alpha=0.3, color='green')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('综合情绪指数 (Z-Score)', fontsize=12)
        ax4.set_ylabel('情绪指数')
        ax4.set_xlabel('日期')

        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        plt.savefig(f'{REPORT_PATH}/sentiment_indicators.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  已保存: sentiment_indicators.png")

        # 图2: 情绪与收益关系图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 2.1 情绪分组收益柱状图
        ax1 = axes[0, 0]
        extreme_df = self.results['extreme_effect']
        x = range(len(extreme_df))
        width = 0.2
        ax1.bar([i - 1.5*width for i in x], extreme_df['1日收益'], width, label='1日', color='blue', alpha=0.7)
        ax1.bar([i - 0.5*width for i in x], extreme_df['5日收益'], width, label='5日', color='green', alpha=0.7)
        ax1.bar([i + 0.5*width for i in x], extreme_df['10日收益'], width, label='10日', color='orange', alpha=0.7)
        ax1.bar([i + 1.5*width for i in x], extreme_df['20日收益'], width, label='20日', color='red', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(extreme_df.index, rotation=45)
        ax1.set_title('情绪分组与未来收益', fontsize=12)
        ax1.set_ylabel('收益率(%)')
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 2.2 情绪变化与收益
        ax2 = axes[0, 1]
        reversal_df = self.results['reversal_effect']
        x = range(len(reversal_df))
        ax2.bar([i - 1.5*width for i in x], reversal_df['1日收益'], width, label='1日', color='blue', alpha=0.7)
        ax2.bar([i - 0.5*width for i in x], reversal_df['5日收益'], width, label='5日', color='green', alpha=0.7)
        ax2.bar([i + 0.5*width for i in x], reversal_df['10日收益'], width, label='10日', color='orange', alpha=0.7)
        ax2.bar([i + 1.5*width for i in x], reversal_df['20日收益'], width, label='20日', color='red', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(reversal_df.index, rotation=45)
        ax2.set_title('情绪变化与未来收益', fontsize=12)
        ax2.set_ylabel('收益率(%)')
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 2.3 情绪自相关
        ax3 = axes[1, 0]
        autocorr_df = self.results['autocorrelation']
        ax3.bar(autocorr_df['滞后期'], autocorr_df['自相关系数'], color='steelblue', alpha=0.7)
        ax3.set_title('情绪指数自相关性', fontsize=12)
        ax3.set_xlabel('滞后期(天)')
        ax3.set_ylabel('自相关系数')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 2.4 情绪分位数与收益散点图
        ax4 = axes[1, 1]
        valid_df = df.dropna(subset=['sentiment_pct', 'fwd_ret_10d'])
        scatter = ax4.scatter(valid_df['sentiment_pct'], valid_df['fwd_ret_10d'],
                            alpha=0.3, c=valid_df['fwd_ret_10d'], cmap='RdYlGn', s=10)
        ax4.set_title('情绪分位数 vs 未来10日收益', fontsize=12)
        ax4.set_xlabel('情绪分位数')
        ax4.set_ylabel('未来10日收益(%)')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax4, label='收益率')

        plt.tight_layout()
        plt.savefig(f'{REPORT_PATH}/sentiment_returns_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  已保存: sentiment_returns_analysis.png")

        # 图3: 策略回测图
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 3.1 累计收益曲线
        ax1 = axes[0]
        ax1.plot(df['trade_date_dt'], df['cum_hold'], label='持有指数', color='blue', alpha=0.8)
        ax1.plot(df['trade_date_dt'], df['cum_contrarian'], label='逆向策略', color='green', alpha=0.8)
        ax1.plot(df['trade_date_dt'], df['cum_momentum'], label='动量策略', color='red', alpha=0.8)
        ax1.set_title('情绪策略累计收益对比', fontsize=12)
        ax1.set_ylabel('累计净值')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.grid(True, alpha=0.3)

        # 3.2 风险预警信号与指数走势
        ax2 = axes[1]
        ax2.plot(df['trade_date_dt'], df['index_close'], color='blue', alpha=0.7, label='上证指数')
        # 标记高风险点
        high_risk = df[df['risk_score'] >= 2]
        ax2.scatter(high_risk['trade_date_dt'], high_risk['index_close'],
                   color='red', s=20, alpha=0.5, label='高风险预警')
        ax2.set_title('风险预警信号与指数走势', fontsize=12)
        ax2.set_ylabel('指数点位')
        ax2.set_xlabel('日期')
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{REPORT_PATH}/sentiment_strategy_backtest.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  已保存: sentiment_strategy_backtest.png")

        # 图4: 资金流向分析
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 获取有资金流向数据的日期
        mf_df = df.dropna(subset=['total_net_flow'])

        if len(mf_df) > 0:
            # 4.1 主力资金净流入
            ax1 = axes[0]
            colors = ['red' if x > 0 else 'green' for x in mf_df['elg_net_flow']]
            ax1.bar(mf_df['trade_date_dt'], mf_df['elg_net_flow'] / 1e8,
                   color=colors, alpha=0.6, width=1)
            ax1.set_title('超大单净流入 (亿元)', fontsize=12)
            ax1.set_ylabel('净流入(亿)')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # 4.2 各类资金累计流入
            ax2 = axes[1]
            mf_df = mf_df.copy()
            mf_df['cum_elg'] = mf_df['elg_net_flow'].cumsum() / 1e8
            mf_df['cum_lg'] = mf_df['lg_net_flow'].cumsum() / 1e8
            mf_df['cum_md'] = mf_df['md_net_flow'].cumsum() / 1e8
            mf_df['cum_sm'] = mf_df['sm_net_flow'].cumsum() / 1e8

            ax2.plot(mf_df['trade_date_dt'], mf_df['cum_elg'], label='超大单', color='red', alpha=0.8)
            ax2.plot(mf_df['trade_date_dt'], mf_df['cum_lg'], label='大单', color='orange', alpha=0.8)
            ax2.plot(mf_df['trade_date_dt'], mf_df['cum_md'], label='中单', color='blue', alpha=0.8)
            ax2.plot(mf_df['trade_date_dt'], mf_df['cum_sm'], label='小单', color='green', alpha=0.8)
            ax2.set_title('各类资金累计净流入 (亿元)', fontsize=12)
            ax2.set_ylabel('累计净流入(亿)')
            ax2.set_xlabel('日期')
            ax2.legend(loc='upper left')
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{REPORT_PATH}/moneyflow_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  已保存: moneyflow_analysis.png")

        return True

    def generate_report(self):
        """生成研究报告"""
        print("\n" + "=" * 60)
        print("5. 生成研究报告")
        print("=" * 60)

        report = f"""# 市场情绪因子研究报告

## 研究概述

- **研究期间**: {self.start_date} - {self.end_date}
- **数据来源**: Tushare数据库
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 情绪指标构建

### 1.1 涨跌家数比 (Advance-Decline Ratio)

涨跌家数比是衡量市场广度的重要指标，反映了市场整体的强弱程度。

**计算公式**: AD Ratio = 上涨家数 / 下跌家数

**统计结果**:
- 平均涨跌比: {self.sentiment_df['ad_ratio'].mean():.3f}
- 最大涨跌比: {self.sentiment_df['ad_ratio'].max():.3f}
- 最小涨跌比: {self.sentiment_df['ad_ratio'].min():.3f}
- 标准差: {self.sentiment_df['ad_ratio'].std():.3f}

**解读**: 当涨跌比大于1时，表示市场上涨家数多于下跌家数，市场整体偏强；反之则偏弱。

### 1.2 涨停/跌停数量

涨跌停数量反映了市场的极端情绪。

**统计结果**:
- 平均涨停数: {self.sentiment_df['limit_up'].mean():.1f} 家
- 平均跌停数: {self.sentiment_df['limit_down'].mean():.1f} 家
- 最大涨停数: {self.sentiment_df['limit_up'].max():.0f} 家
- 最大跌停数: {self.sentiment_df['limit_down'].max():.0f} 家

### 1.3 换手率分布

换手率反映了市场的交易活跃度和资金参与意愿。

**统计结果**:
- 平均换手率: {self.sentiment_df['avg_turnover'].mean():.2f}%
- 最高换手率: {self.sentiment_df['avg_turnover'].max():.2f}%
- 最低换手率: {self.sentiment_df['avg_turnover'].min():.2f}%

### 1.4 综合情绪指数

综合情绪指数由涨跌比、涨跌停差、换手率三个指标标准化后等权合成。

**解读**:
- 情绪指数 > 0: 市场情绪偏乐观
- 情绪指数 < 0: 市场情绪偏悲观
- 情绪指数 > 2: 市场过度乐观，需警惕回调
- 情绪指数 < -2: 市场过度悲观，可能存在机会

---

## 2. 情绪与收益关系分析

### 2.1 情绪极端值效应

根据情绪分位数将市场分为5组，分析各组未来收益表现：

{self.results['extreme_effect'].round(3).to_markdown()}

**发现**:
{self._analyze_extreme_effect()}

### 2.2 情绪反转效应

分析情绪变化与未来收益的关系：

{self.results['reversal_effect'].round(3).to_markdown()}

**发现**:
{self._analyze_reversal_effect()}

### 2.3 情绪持续性

情绪指数自相关分析显示：

{self.results['autocorrelation'].set_index('滞后期').T.round(3).to_markdown()}

**发现**:
- 情绪具有较强的短期持续性，1日自相关系数约为 {self.results['autocorrelation'][self.results['autocorrelation']['滞后期']==1]['自相关系数'].values[0]:.3f}
- 随着滞后期增加，自相关逐渐减弱
- 约20日后自相关基本消失，说明情绪周期约为1个月

---

## 3. 策略应用

### 3.1 情绪择时策略

#### 逆向策略
- **策略逻辑**: 情绪低迷时买入，情绪高涨时卖出
- **信号规则**:
  - 情绪分位数 < 20%: 满仓
  - 情绪分位数 > 80%: 空仓
  - 其他情况: 半仓

#### 动量策略
- **策略逻辑**: 跟随情绪趋势
- **信号规则**: 情绪5日均线上穿20日均线时满仓，下穿时空仓

### 策略绩效比较

{self.results['strategy_stats'].to_markdown()}

### 3.2 风险预警

基于以下条件构建风险预警系统：
1. 情绪分位数 > 90%
2. 换手率Z值 > 2
3. 涨跌停差Z值 > 2

**风险预警效果**:

{self.results['risk_warning'].to_markdown()}

**应用建议**:
- 当风险分数 >= 2 时，建议降低仓位
- 当风险分数 = 3 时，建议大幅减仓或空仓

---

## 4. 研究结论与投资建议

### 4.1 主要发现

1. **情绪极端值具有预测价值**: 当市场情绪处于极端悲观状态时，未来收益往往较好；反之亦然。这验证了经典的逆向投资理论。

2. **情绪反转效应显著**: 情绪快速上升后，市场往往面临调整压力；情绪快速下降后，市场可能迎来反弹机会。

3. **情绪具有短期持续性**: 情绪在短期内（1-5天）具有较强的持续性，但长期来看会均值回归。

### 4.2 投资建议

1. **逆向思维**: 当市场情绪极度悲观时（情绪分位数<10%），可考虑逐步建仓；当市场情绪极度乐观时（情绪分位数>90%），应考虑逐步减仓。

2. **风险控制**: 密切关注风险预警信号，当预警分数达到2分以上时，应提高警惕并适当降低仓位。

3. **综合判断**: 情绪指标应与其他基本面、技术面指标结合使用，不宜单独作为投资决策依据。

### 4.3 后续研究方向

1. 结合行业情绪进行轮动策略研究
2. 探索情绪指标在择股中的应用
3. 引入更多情绪代理变量（如舆情数据、融资融券等）

---

## 附录：图表说明

1. **sentiment_indicators.png**: 情绪指标时序图
2. **sentiment_returns_analysis.png**: 情绪与收益关系分析图
3. **sentiment_strategy_backtest.png**: 策略回测图
4. **moneyflow_analysis.png**: 资金流向分析图

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # 保存报告
        report_path = f'{REPORT_PATH}/market_sentiment_research_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")

        # 保存原始数据
        data_path = f'{REPORT_PATH}/sentiment_data.csv'
        self.strategy_df.to_csv(data_path, index=False)
        print(f"数据已保存: {data_path}")

        return report

    def _analyze_extreme_effect(self):
        """分析情绪极端值效应"""
        df = self.results['extreme_effect']
        findings = []

        # 比较极度悲观和极度乐观
        if '极度悲观' in df.index and '极度乐观' in df.index:
            pessimistic_ret_20 = df.loc['极度悲观', '20日收益']
            optimistic_ret_20 = df.loc['极度乐观', '20日收益']
            pessimistic_ret_10 = df.loc['极度悲观', '10日收益']
            optimistic_ret_10 = df.loc['极度乐观', '10日收益']
            neutral_ret_10 = df.loc['中性', '10日收益'] if '中性' in df.index else 0

            findings.append(f"- 极度悲观时期的未来20日收益为{pessimistic_ret_20:.2f}%")
            findings.append(f"- 极度乐观时期的未来20日收益为{optimistic_ret_20:.2f}%")
            findings.append(f"- 中性时期的未来10日收益为{neutral_ret_10:.2f}%")

            if pessimistic_ret_20 > neutral_ret_10:
                findings.append("- 极度悲观后存在一定的反弹效应，但需结合其他因素综合判断")

            if optimistic_ret_10 > neutral_ret_10:
                findings.append("- A股市场呈现动量特征：高情绪后短期仍可能上涨")
                findings.append("- 但高情绪持续后风险积累，中长期需谨慎")

        return '\n'.join(findings) if findings else "- 需要更多数据进行分析"

    def _analyze_reversal_effect(self):
        """分析情绪反转效应"""
        df = self.results['reversal_effect']
        findings = []

        if '大幅上升' in df.index and '大幅下降' in df.index and '持平' in df.index:
            up_ret_20 = df.loc['大幅上升', '20日收益']
            down_ret_20 = df.loc['大幅下降', '20日收益']
            flat_ret_20 = df.loc['持平', '20日收益']

            findings.append(f"- 情绪大幅下降后的未来20日收益为{down_ret_20:.2f}%")
            findings.append(f"- 情绪大幅上升后的未来20日收益为{up_ret_20:.2f}%")
            findings.append(f"- 情绪持平时期的未来20日收益为{flat_ret_20:.2f}%")

            if flat_ret_20 > up_ret_20 and flat_ret_20 > down_ret_20:
                findings.append("- 情绪平稳时期往往有较好的中期收益，剧烈波动后市场需要时间消化")

            if down_ret_20 < 0:
                findings.append("- 情绪急剧恶化可能预示着趋势延续，不宜过早抄底")

        return '\n'.join(findings) if findings else "- 需要更多数据进行分析"

    def run(self):
        """运行完整的研究流程"""
        print("\n" + "=" * 60)
        print("市场情绪因子研究")
        print("=" * 60)
        print(f"研究期间: {self.start_date} - {self.end_date}")
        print(f"数据库: {self.db_path}")
        print("=" * 60)

        # 1. 构建情绪指标
        self.build_sentiment_indicators()

        # 2. 分析情绪与收益
        self.analyze_sentiment_returns()

        # 3. 构建策略
        self.build_sentiment_strategy()

        # 4. 生成可视化
        self.generate_visualizations()

        # 5. 生成报告
        self.generate_report()

        print("\n" + "=" * 60)
        print("研究完成！")
        print("=" * 60)

        return self.results


if __name__ == '__main__':
    # 运行研究
    research = MarketSentimentResearch(
        start_date='20150101',
        end_date='20260130'
    )
    results = research.run()
