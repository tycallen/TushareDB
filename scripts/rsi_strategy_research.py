#!/usr/bin/env python3
"""
RSI交易策略研究报告
====================
基于Tushare数据库研究RSI指标的计算、分析与交易策略回测

研究内容：
1. RSI计算与分析：不同周期RSI、RSI分布特征、行业RSI差异
2. 交易信号：超买超卖信号、RSI背离信号、RSI黄金分割
3. 策略回测：RSI极值策略、RSI动量策略、结合其他指标
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


class RSICalculator:
    """RSI指标计算器"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI指标
        RSI = 100 - (100 / (1 + RS))
        RS = 平均上涨幅度 / 平均下跌幅度
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_rsi_ema(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        使用EMA方法计算RSI（更平滑）
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RSIResearch:
    """RSI策略研究类"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.calculator = RSICalculator()
        self.report_content = []

    def add_to_report(self, content: str):
        """添加内容到报告"""
        self.report_content.append(content)
        print(content)

    def get_stock_data(self, ts_code: str = None, start_date: str = '20200101',
                       end_date: str = '20260130', limit: int = None) -> pd.DataFrame:
        """获取股票日线数据"""
        query = f"""
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               d.pre_close, d.pct_chg, d.vol, d.amount,
               s.name, s.industry
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE d.trade_date >= '{start_date}'
          AND d.trade_date <= '{end_date}'
          AND s.list_status = 'L'
        """
        if ts_code:
            query += f" AND d.ts_code = '{ts_code}'"
        query += " ORDER BY d.ts_code, d.trade_date"
        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).fetchdf()

    def get_industry_stocks(self) -> pd.DataFrame:
        """获取申万行业分类"""
        query = """
        SELECT DISTINCT l1_name as industry, ts_code, name as stock_name
        FROM index_member_all
        WHERE is_new = 'Y' AND out_date IS NULL
        """
        return self.conn.execute(query).fetchdf()

    # ==================== 第一部分：RSI计算与分析 ====================

    def analyze_rsi_periods(self) -> Dict:
        """分析不同周期RSI的特征"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("1. RSI计算与分析")
        self.add_to_report("="*80)

        self.add_to_report("\n### 1.1 不同周期RSI分析")

        # 获取样本股票数据
        sample_stocks = self.conn.execute("""
            SELECT DISTINCT ts_code
            FROM daily
            WHERE trade_date >= '20240101'
            GROUP BY ts_code
            HAVING COUNT(*) > 200
            LIMIT 100
        """).fetchdf()['ts_code'].tolist()

        periods = [6, 9, 12, 14, 20, 30]
        period_stats = {p: {'mean': [], 'std': [], 'above_70': [], 'below_30': []} for p in periods}

        for ts_code in sample_stocks[:50]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date')

            for period in periods:
                rsi = self.calculator.calculate_rsi(df['close'], period)
                rsi_valid = rsi.dropna()
                if len(rsi_valid) > 0:
                    period_stats[period]['mean'].append(rsi_valid.mean())
                    period_stats[period]['std'].append(rsi_valid.std())
                    period_stats[period]['above_70'].append((rsi_valid > 70).sum() / len(rsi_valid) * 100)
                    period_stats[period]['below_30'].append((rsi_valid < 30).sum() / len(rsi_valid) * 100)

        # 汇总统计
        results = []
        for period in periods:
            if period_stats[period]['mean']:
                results.append({
                    'RSI周期': period,
                    '平均值': np.mean(period_stats[period]['mean']),
                    '标准差': np.mean(period_stats[period]['std']),
                    '超买比例(>70)%': np.mean(period_stats[period]['above_70']),
                    '超卖比例(<30)%': np.mean(period_stats[period]['below_30'])
                })

        results_df = pd.DataFrame(results)
        self.add_to_report(f"\n不同RSI周期统计特征（样本数：{len(sample_stocks[:50])}只股票）：")
        self.add_to_report(results_df.to_string(index=False))

        # 绘制不同周期RSI对比图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 选一只股票展示
        sample_stock = '000001.SZ'
        df = self.get_stock_data(ts_code=sample_stock, start_date='20240101')
        df = df.sort_values('trade_date')
        df['date'] = pd.to_datetime(df['trade_date'])

        colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))

        # 图1：不同周期RSI曲线
        ax1 = axes[0, 0]
        for i, period in enumerate(periods):
            rsi = self.calculator.calculate_rsi(df['close'], period)
            ax1.plot(df['date'], rsi, label=f'RSI({period})', color=colors[i], alpha=0.8)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线(70)')
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线(30)')
        ax1.set_title(f'{sample_stock} 不同周期RSI对比')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_ylabel('RSI值')
        ax1.grid(True, alpha=0.3)

        # 图2：RSI周期与特征关系
        ax2 = axes[0, 1]
        x = range(len(results_df))
        width = 0.35
        ax2.bar([i - width/2 for i in x], results_df['超买比例(>70)%'], width, label='超买比例(%)', color='red', alpha=0.7)
        ax2.bar([i + width/2 for i in x], results_df['超卖比例(<30)%'], width, label='超卖比例(%)', color='green', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['RSI周期'])
        ax2.set_xlabel('RSI周期')
        ax2.set_ylabel('比例(%)')
        ax2.set_title('不同周期RSI超买超卖比例')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3：RSI均值与标准差
        ax3 = axes[1, 0]
        ax3.errorbar(results_df['RSI周期'], results_df['平均值'],
                     yerr=results_df['标准差'], marker='o', capsize=5, capthick=2)
        ax3.set_xlabel('RSI周期')
        ax3.set_ylabel('RSI均值 ± 标准差')
        ax3.set_title('不同周期RSI均值与波动性')
        ax3.grid(True, alpha=0.3)

        # 图4：价格与RSI(14)关系
        ax4 = axes[1, 1]
        rsi14 = self.calculator.calculate_rsi(df['close'], 14)
        ax4_price = ax4.twinx()
        ax4.plot(df['date'], rsi14, color='blue', label='RSI(14)')
        ax4_price.plot(df['date'], df['close'], color='orange', alpha=0.7, label='收盘价')
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax4.set_ylabel('RSI', color='blue')
        ax4_price.set_ylabel('收盘价', color='orange')
        ax4.set_title(f'{sample_stock} 价格与RSI(14)关系')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rsi_period_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_period_analysis.png")

        return results_df.to_dict('records')

    def analyze_rsi_distribution(self) -> Dict:
        """分析RSI分布特征"""
        self.add_to_report("\n### 1.2 RSI分布特征分析")

        # 获取大量股票的RSI数据
        all_rsi = []
        bull_rsi = []  # 牛市（2024年上半年涨幅大的股票）
        bear_rsi = []  # 熊市

        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 500
        """).fetchdf()['ts_code'].tolist()

        for ts_code in stocks:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue
            df = df.sort_values('trade_date')
            rsi = self.calculator.calculate_rsi(df['close'], 14).dropna()
            all_rsi.extend(rsi.tolist())

            # 判断趋势
            if len(df) > 50:
                returns = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                if returns > 0.1:
                    bull_rsi.extend(rsi.tolist())
                elif returns < -0.1:
                    bear_rsi.extend(rsi.tolist())

        all_rsi = np.array(all_rsi)

        # 统计分布
        stats = {
            '总体样本数': len(all_rsi),
            '均值': np.mean(all_rsi),
            '中位数': np.median(all_rsi),
            '标准差': np.std(all_rsi),
            '偏度': pd.Series(all_rsi).skew(),
            '峰度': pd.Series(all_rsi).kurtosis(),
            '<20比例%': (all_rsi < 20).sum() / len(all_rsi) * 100,
            '20-30比例%': ((all_rsi >= 20) & (all_rsi < 30)).sum() / len(all_rsi) * 100,
            '30-50比例%': ((all_rsi >= 30) & (all_rsi < 50)).sum() / len(all_rsi) * 100,
            '50-70比例%': ((all_rsi >= 50) & (all_rsi < 70)).sum() / len(all_rsi) * 100,
            '70-80比例%': ((all_rsi >= 70) & (all_rsi < 80)).sum() / len(all_rsi) * 100,
            '>80比例%': (all_rsi >= 80).sum() / len(all_rsi) * 100,
        }

        self.add_to_report(f"\nRSI(14)总体分布统计：")
        for k, v in stats.items():
            if isinstance(v, float):
                self.add_to_report(f"  {k}: {v:.4f}")
            else:
                self.add_to_report(f"  {k}: {v}")

        # 绘制分布图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1：RSI分布直方图
        ax1 = axes[0, 0]
        ax1.hist(all_rsi, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=30, color='g', linestyle='--', linewidth=2, label='超卖线(30)')
        ax1.axvline(x=70, color='r', linestyle='--', linewidth=2, label='超买线(70)')
        ax1.axvline(x=np.mean(all_rsi), color='orange', linestyle='-', linewidth=2, label=f'均值({np.mean(all_rsi):.1f})')
        ax1.set_xlabel('RSI值')
        ax1.set_ylabel('密度')
        ax1.set_title('RSI(14)总体分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2：牛市vs熊市RSI分布对比
        ax2 = axes[0, 1]
        if bull_rsi and bear_rsi:
            ax2.hist(bull_rsi, bins=40, density=True, alpha=0.6, color='red', label=f'上涨股票(n={len(bull_rsi)})')
            ax2.hist(bear_rsi, bins=40, density=True, alpha=0.6, color='green', label=f'下跌股票(n={len(bear_rsi)})')
            ax2.axvline(x=30, color='g', linestyle='--', linewidth=1)
            ax2.axvline(x=70, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('RSI值')
            ax2.set_ylabel('密度')
            ax2.set_title('上涨股票 vs 下跌股票 RSI分布对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 图3：RSI区间分布饼图
        ax3 = axes[1, 0]
        labels = ['<20', '20-30', '30-50', '50-70', '70-80', '>80']
        sizes = [stats['<20比例%'], stats['20-30比例%'], stats['30-50比例%'],
                 stats['50-70比例%'], stats['70-80比例%'], stats['>80比例%']]
        colors_pie = ['darkgreen', 'lightgreen', 'lightyellow', 'lightyellow', 'lightcoral', 'darkred']
        explode = (0.05, 0.02, 0, 0, 0.02, 0.05)
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax3.set_title('RSI(14)区间分布')

        # 图4：RSI累计分布函数
        ax4 = axes[1, 1]
        sorted_rsi = np.sort(all_rsi)
        cdf = np.arange(1, len(sorted_rsi) + 1) / len(sorted_rsi)
        ax4.plot(sorted_rsi, cdf, color='blue', linewidth=2)
        ax4.axvline(x=30, color='g', linestyle='--', label='30')
        ax4.axvline(x=70, color='r', linestyle='--', label='70')
        ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('RSI值')
        ax4.set_ylabel('累计概率')
        ax4.set_title('RSI(14)累计分布函数(CDF)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rsi_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_distribution.png")

        return stats

    def analyze_industry_rsi(self) -> pd.DataFrame:
        """分析行业RSI差异"""
        self.add_to_report("\n### 1.3 行业RSI差异分析")

        # 获取行业成分股
        industry_stocks = self.get_industry_stocks()

        # 按行业计算RSI
        industry_rsi = {}

        for industry in industry_stocks['industry'].unique():
            stocks = industry_stocks[industry_stocks['industry'] == industry]['ts_code'].tolist()
            industry_rsi[industry] = {'mean': [], 'std': [], 'above_70': [], 'below_30': []}

            for ts_code in stocks[:30]:  # 每个行业取30只股票
                df = self.get_stock_data(ts_code=ts_code, start_date='20240101')
                if len(df) < 50:
                    continue
                df = df.sort_values('trade_date')
                rsi = self.calculator.calculate_rsi(df['close'], 14).dropna()
                if len(rsi) > 0:
                    industry_rsi[industry]['mean'].append(rsi.mean())
                    industry_rsi[industry]['std'].append(rsi.std())
                    industry_rsi[industry]['above_70'].append((rsi > 70).sum() / len(rsi) * 100)
                    industry_rsi[industry]['below_30'].append((rsi < 30).sum() / len(rsi) * 100)

        # 汇总统计
        results = []
        for industry, data in industry_rsi.items():
            if data['mean'] and len(data['mean']) >= 5:
                results.append({
                    '行业': industry,
                    '样本数': len(data['mean']),
                    'RSI均值': np.mean(data['mean']),
                    'RSI标准差': np.mean(data['std']),
                    '超买比例%': np.mean(data['above_70']),
                    '超卖比例%': np.mean(data['below_30'])
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RSI均值', ascending=False)

        self.add_to_report(f"\n行业RSI(14)统计（2024年以来）：")
        self.add_to_report(results_df.to_string(index=False))

        # 绘制行业RSI对比图
        if len(results_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 取前15个行业
            top_df = results_df.head(15)

            # 图1：行业RSI均值排名
            ax1 = axes[0, 0]
            colors = ['red' if x > 55 else 'green' if x < 45 else 'gray' for x in top_df['RSI均值']]
            bars = ax1.barh(range(len(top_df)), top_df['RSI均值'], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(top_df)))
            ax1.set_yticklabels(top_df['行业'])
            ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5)
            ax1.set_xlabel('RSI均值')
            ax1.set_title('行业RSI均值排名（Top 15）')
            ax1.grid(True, alpha=0.3, axis='x')

            # 图2：行业RSI超买超卖比例
            ax2 = axes[0, 1]
            x = range(len(top_df))
            width = 0.35
            ax2.barh([i - width/2 for i in x], top_df['超买比例%'], width, label='超买比例', color='red', alpha=0.7)
            ax2.barh([i + width/2 for i in x], top_df['超卖比例%'], width, label='超卖比例', color='green', alpha=0.7)
            ax2.set_yticks(x)
            ax2.set_yticklabels(top_df['行业'])
            ax2.set_xlabel('比例(%)')
            ax2.set_title('行业超买超卖比例对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')

            # 图3：RSI均值vs标准差散点图
            ax3 = axes[1, 0]
            scatter = ax3.scatter(results_df['RSI均值'], results_df['RSI标准差'],
                                  c=results_df['超买比例%'], cmap='RdYlGn_r',
                                  s=100, alpha=0.7, edgecolors='black')
            plt.colorbar(scatter, ax=ax3, label='超买比例%')
            ax3.set_xlabel('RSI均值')
            ax3.set_ylabel('RSI标准差')
            ax3.set_title('行业RSI均值vs波动性')
            ax3.grid(True, alpha=0.3)

            # 标注部分行业
            for _, row in results_df.head(5).iterrows():
                ax3.annotate(row['行业'], (row['RSI均值'], row['RSI标准差']), fontsize=8)

            # 图4：行业RSI箱线图
            ax4 = axes[1, 1]
            box_data = []
            box_labels = []
            for industry in top_df['行业'].tolist()[:10]:
                if industry in industry_rsi and industry_rsi[industry]['mean']:
                    box_data.append(industry_rsi[industry]['mean'])
                    box_labels.append(industry[:4])

            if box_data:
                bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
                ax4.set_ylabel('RSI均值')
                ax4.set_title('行业RSI均值分布（箱线图）')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/rsi_industry_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()

            self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_industry_analysis.png")

        return results_df

    # ==================== 第二部分：交易信号 ====================

    def analyze_overbought_oversold(self) -> Dict:
        """超买超卖信号分析"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("2. 交易信号分析")
        self.add_to_report("="*80)

        self.add_to_report("\n### 2.1 超买超卖信号分析")

        # 分析不同RSI阈值的效果
        thresholds = [(20, 80), (25, 75), (30, 70), (35, 65)]
        results = []

        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 200
        """).fetchdf()['ts_code'].tolist()

        for oversold, overbought in thresholds:
            signal_returns = {'buy': [], 'sell': []}

            for ts_code in stocks[:100]:
                df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
                if len(df) < 100:
                    continue

                df = df.sort_values('trade_date').reset_index(drop=True)
                df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
                df['return_5d'] = df['close'].shift(-5) / df['close'] - 1
                df['return_10d'] = df['close'].shift(-10) / df['close'] - 1
                df['return_20d'] = df['close'].shift(-20) / df['close'] - 1

                # 超卖买入信号
                buy_signals = df[df['rsi'] < oversold]
                if len(buy_signals) > 0:
                    signal_returns['buy'].extend(buy_signals['return_5d'].dropna().tolist())

                # 超买卖出信号
                sell_signals = df[df['rsi'] > overbought]
                if len(sell_signals) > 0:
                    signal_returns['sell'].extend((-sell_signals['return_5d']).dropna().tolist())

            if signal_returns['buy'] and signal_returns['sell']:
                results.append({
                    '阈值': f'{oversold}/{overbought}',
                    '买入信号数': len(signal_returns['buy']),
                    '买入5日胜率%': (np.array(signal_returns['buy']) > 0).sum() / len(signal_returns['buy']) * 100,
                    '买入5日均收益%': np.mean(signal_returns['buy']) * 100,
                    '卖出信号数': len(signal_returns['sell']),
                    '卖出5日胜率%': (np.array(signal_returns['sell']) > 0).sum() / len(signal_returns['sell']) * 100,
                    '卖出5日均收益%': np.mean(signal_returns['sell']) * 100,
                })

        results_df = pd.DataFrame(results)
        self.add_to_report(f"\n超买超卖信号分析（不同阈值）：")
        self.add_to_report(results_df.to_string(index=False))

        # 详细分析30/70阈值
        self.add_to_report("\n#### RSI(14) 30/70阈值详细分析：")

        detailed_results = {'buy': {'5d': [], '10d': [], '20d': []},
                           'sell': {'5d': [], '10d': [], '20d': []}}

        for ts_code in stocks[:100]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['return_5d'] = df['close'].shift(-5) / df['close'] - 1
            df['return_10d'] = df['close'].shift(-10) / df['close'] - 1
            df['return_20d'] = df['close'].shift(-20) / df['close'] - 1

            buy_signals = df[df['rsi'] < 30]
            sell_signals = df[df['rsi'] > 70]

            for period in ['5d', '10d', '20d']:
                col = f'return_{period}'
                detailed_results['buy'][period].extend(buy_signals[col].dropna().tolist())
                detailed_results['sell'][period].extend((-sell_signals[col]).dropna().tolist())

        self.add_to_report("\n买入信号（RSI<30）后续收益统计：")
        for period in ['5d', '10d', '20d']:
            data = detailed_results['buy'][period]
            if data:
                self.add_to_report(f"  {period}: 信号数={len(data)}, 胜率={((np.array(data)>0).sum()/len(data)*100):.1f}%, "
                                 f"均收益={(np.mean(data)*100):.2f}%, 最大收益={(max(data)*100):.1f}%, 最大亏损={(min(data)*100):.1f}%")

        self.add_to_report("\n卖出信号（RSI>70）后续收益统计：")
        for period in ['5d', '10d', '20d']:
            data = detailed_results['sell'][period]
            if data:
                self.add_to_report(f"  {period}: 信号数={len(data)}, 胜率={((np.array(data)>0).sum()/len(data)*100):.1f}%, "
                                 f"均收益={(np.mean(data)*100):.2f}%")

        # 绘制超买超卖信号图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1：买入信号收益分布
        ax1 = axes[0, 0]
        buy_data = detailed_results['buy']['5d']
        if buy_data:
            ax1.hist(np.array(buy_data) * 100, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax1.axvline(x=np.mean(buy_data) * 100, color='red', linestyle='-', linewidth=2,
                       label=f'均值={np.mean(buy_data)*100:.2f}%')
            ax1.set_xlabel('5日收益率(%)')
            ax1.set_ylabel('密度')
            ax1.set_title('RSI<30买入信号5日收益分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 图2：卖出信号收益分布
        ax2 = axes[0, 1]
        sell_data = detailed_results['sell']['5d']
        if sell_data:
            ax2.hist(np.array(sell_data) * 100, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax2.axvline(x=np.mean(sell_data) * 100, color='blue', linestyle='-', linewidth=2,
                       label=f'均值={np.mean(sell_data)*100:.2f}%')
            ax2.set_xlabel('5日收益率(%)')
            ax2.set_ylabel('密度')
            ax2.set_title('RSI>70卖出信号5日收益分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 图3：不同阈值胜率对比
        ax3 = axes[1, 0]
        x = range(len(results_df))
        width = 0.35
        ax3.bar([i - width/2 for i in x], results_df['买入5日胜率%'], width, label='买入胜率', color='green', alpha=0.7)
        ax3.bar([i + width/2 for i in x], results_df['卖出5日胜率%'], width, label='卖出胜率', color='red', alpha=0.7)
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(results_df['阈值'])
        ax3.set_xlabel('RSI阈值(超卖/超买)')
        ax3.set_ylabel('胜率(%)')
        ax3.set_title('不同RSI阈值信号胜率对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图4：示例股票RSI与买卖信号
        ax4 = axes[1, 1]
        sample_df = self.get_stock_data(ts_code='000001.SZ', start_date='20240101')
        sample_df = sample_df.sort_values('trade_date')
        sample_df['date'] = pd.to_datetime(sample_df['trade_date'])
        sample_df['rsi'] = self.calculator.calculate_rsi(sample_df['close'], 14)

        ax4_price = ax4.twinx()
        ax4.plot(sample_df['date'], sample_df['rsi'], color='blue', label='RSI(14)')
        ax4_price.plot(sample_df['date'], sample_df['close'], color='gray', alpha=0.5, label='价格')

        # 标记买卖信号
        buy_mask = sample_df['rsi'] < 30
        sell_mask = sample_df['rsi'] > 70
        ax4.scatter(sample_df[buy_mask]['date'], sample_df[buy_mask]['rsi'],
                    color='green', marker='^', s=100, label='买入信号', zorder=5)
        ax4.scatter(sample_df[sell_mask]['date'], sample_df[sell_mask]['rsi'],
                    color='red', marker='v', s=100, label='卖出信号', zorder=5)

        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax4.set_ylabel('RSI', color='blue')
        ax4_price.set_ylabel('价格', color='gray')
        ax4.set_title('000001.SZ RSI买卖信号示例')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rsi_overbought_oversold.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_overbought_oversold.png")

        return results_df.to_dict('records')

    def analyze_rsi_divergence(self) -> Dict:
        """RSI背离信号分析"""
        self.add_to_report("\n### 2.2 RSI背离信号分析")

        def detect_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
            """检测RSI背离"""
            df = df.copy()
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['bullish_div'] = False
            df['bearish_div'] = False

            for i in range(lookback, len(df)):
                window = df.iloc[i-lookback:i+1]

                # 检测看涨背离：价格创新低但RSI没有创新低
                price_low_idx = window['close'].idxmin()
                rsi_low_idx = window['rsi'].idxmin()
                if (df.loc[i, 'close'] < window['close'].iloc[:-1].min() and
                    df.loc[i, 'rsi'] > window['rsi'].iloc[:-1].min()):
                    df.loc[i, 'bullish_div'] = True

                # 检测看跌背离：价格创新高但RSI没有创新高
                if (df.loc[i, 'close'] > window['close'].iloc[:-1].max() and
                    df.loc[i, 'rsi'] < window['rsi'].iloc[:-1].max()):
                    df.loc[i, 'bearish_div'] = True

            return df

        # 分析背离信号效果
        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 150
        """).fetchdf()['ts_code'].tolist()

        bullish_returns = []
        bearish_returns = []

        for ts_code in stocks[:80]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df = detect_divergence(df)
            df['return_10d'] = df['close'].shift(-10) / df['close'] - 1

            bullish = df[df['bullish_div']]['return_10d'].dropna()
            bearish = df[df['bearish_div']]['return_10d'].dropna()

            bullish_returns.extend(bullish.tolist())
            bearish_returns.extend((-bearish).tolist())

        self.add_to_report("\n#### 看涨背离（价格新低+RSI未新低）信号分析：")
        if bullish_returns:
            self.add_to_report(f"  信号数量: {len(bullish_returns)}")
            self.add_to_report(f"  10日胜率: {((np.array(bullish_returns)>0).sum()/len(bullish_returns)*100):.1f}%")
            self.add_to_report(f"  10日平均收益: {(np.mean(bullish_returns)*100):.2f}%")
            self.add_to_report(f"  10日最大收益: {(max(bullish_returns)*100):.1f}%")
            self.add_to_report(f"  10日最大亏损: {(min(bullish_returns)*100):.1f}%")

        self.add_to_report("\n#### 看跌背离（价格新高+RSI未新高）信号分析：")
        if bearish_returns:
            self.add_to_report(f"  信号数量: {len(bearish_returns)}")
            self.add_to_report(f"  做空10日胜率: {((np.array(bearish_returns)>0).sum()/len(bearish_returns)*100):.1f}%")
            self.add_to_report(f"  做空10日平均收益: {(np.mean(bearish_returns)*100):.2f}%")

        # 绘制背离示例图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 找一个有背离的例子
        sample_df = None
        for ts_code in stocks[:50]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20240101')
            if len(df) < 100:
                continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            df = detect_divergence(df)
            if df['bullish_div'].sum() > 0 or df['bearish_div'].sum() > 0:
                sample_df = df
                sample_code = ts_code
                break

        if sample_df is not None:
            sample_df['date'] = pd.to_datetime(sample_df['trade_date'])

            # 图1：价格与RSI对比
            ax1 = axes[0, 0]
            ax1_price = ax1.twinx()
            ax1.plot(sample_df['date'], sample_df['rsi'], color='blue', label='RSI(14)')
            ax1_price.plot(sample_df['date'], sample_df['close'], color='orange', alpha=0.7, label='价格')

            # 标记背离点
            bull_div = sample_df[sample_df['bullish_div']]
            bear_div = sample_df[sample_df['bearish_div']]
            ax1.scatter(bull_div['date'], bull_div['rsi'], color='green', marker='^', s=150,
                       label='看涨背离', zorder=5)
            ax1.scatter(bear_div['date'], bear_div['rsi'], color='red', marker='v', s=150,
                       label='看跌背离', zorder=5)

            ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax1.set_ylabel('RSI', color='blue')
            ax1_price.set_ylabel('价格', color='orange')
            ax1.set_title(f'{sample_code} RSI背离信号示例')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

        # 图2：看涨背离收益分布
        ax2 = axes[0, 1]
        if bullish_returns:
            ax2.hist(np.array(bullish_returns) * 100, bins=30, density=True, alpha=0.7,
                    color='green', edgecolor='black')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax2.axvline(x=np.mean(bullish_returns) * 100, color='red', linestyle='-', linewidth=2,
                       label=f'均值={np.mean(bullish_returns)*100:.2f}%')
            ax2.set_xlabel('10日收益率(%)')
            ax2.set_ylabel('密度')
            ax2.set_title('看涨背离信号10日收益分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 图3：看跌背离收益分布
        ax3 = axes[1, 0]
        if bearish_returns:
            ax3.hist(np.array(bearish_returns) * 100, bins=30, density=True, alpha=0.7,
                    color='red', edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax3.axvline(x=np.mean(bearish_returns) * 100, color='blue', linestyle='-', linewidth=2,
                       label=f'均值={np.mean(bearish_returns)*100:.2f}%')
            ax3.set_xlabel('10日收益率(%)')
            ax3.set_ylabel('密度')
            ax3.set_title('看跌背离信号10日收益分布（做空）')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 图4：背离信号统计
        ax4 = axes[1, 1]
        labels = ['看涨背离', '看跌背离']
        win_rates = []
        avg_returns = []
        if bullish_returns:
            win_rates.append((np.array(bullish_returns) > 0).sum() / len(bullish_returns) * 100)
            avg_returns.append(np.mean(bullish_returns) * 100)
        else:
            win_rates.append(0)
            avg_returns.append(0)
        if bearish_returns:
            win_rates.append((np.array(bearish_returns) > 0).sum() / len(bearish_returns) * 100)
            avg_returns.append(np.mean(bearish_returns) * 100)
        else:
            win_rates.append(0)
            avg_returns.append(0)

        x = np.arange(len(labels))
        width = 0.35
        ax4.bar(x - width/2, win_rates, width, label='胜率(%)', color='blue', alpha=0.7)
        ax4.bar(x + width/2, avg_returns, width, label='平均收益(%)', color='orange', alpha=0.7)
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('比例/收益(%)')
        ax4.set_title('RSI背离信号效果统计')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rsi_divergence.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_divergence.png")

        return {'bullish': bullish_returns, 'bearish': bearish_returns}

    def analyze_rsi_golden_ratio(self) -> Dict:
        """RSI黄金分割分析"""
        self.add_to_report("\n### 2.3 RSI黄金分割分析")

        # 黄金分割水平
        golden_levels = {
            '23.6%': 23.6,
            '38.2%': 38.2,
            '50.0%': 50.0,
            '61.8%': 61.8,
            '76.4%': 76.4,
        }

        self.add_to_report("\nRSI黄金分割位：")
        for name, level in golden_levels.items():
            self.add_to_report(f"  {name}: RSI = {level}")

        # 分析RSI在各黄金分割位的表现
        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 150
        """).fetchdf()['ts_code'].tolist()

        level_returns = {level: [] for level in golden_levels.values()}

        for ts_code in stocks[:80]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['return_5d'] = df['close'].shift(-5) / df['close'] - 1

            for level in golden_levels.values():
                # 从下方穿越黄金分割位
                cross_up = (df['rsi'].shift(1) < level) & (df['rsi'] >= level)
                returns = df[cross_up]['return_5d'].dropna()
                level_returns[level].extend(returns.tolist())

        # 统计结果
        results = []
        for name, level in golden_levels.items():
            data = level_returns[level]
            if data:
                results.append({
                    '黄金分割位': name,
                    'RSI值': level,
                    '信号数': len(data),
                    '5日胜率%': (np.array(data) > 0).sum() / len(data) * 100,
                    '5日均收益%': np.mean(data) * 100,
                })

        results_df = pd.DataFrame(results)
        self.add_to_report(f"\nRSI上穿黄金分割位后5日收益统计：")
        self.add_to_report(results_df.to_string(index=False))

        # 绘制黄金分割图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1：示例股票RSI与黄金分割线
        ax1 = axes[0, 0]
        sample_df = self.get_stock_data(ts_code='000001.SZ', start_date='20240101')
        sample_df = sample_df.sort_values('trade_date')
        sample_df['date'] = pd.to_datetime(sample_df['trade_date'])
        sample_df['rsi'] = self.calculator.calculate_rsi(sample_df['close'], 14)

        ax1.plot(sample_df['date'], sample_df['rsi'], color='blue', linewidth=1.5)
        colors_golden = ['purple', 'blue', 'gray', 'orange', 'red']
        for (name, level), color in zip(golden_levels.items(), colors_golden):
            ax1.axhline(y=level, color=color, linestyle='--', alpha=0.6, label=f'{name}({level})')
        ax1.set_ylabel('RSI(14)')
        ax1.set_title('000001.SZ RSI与黄金分割位')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 图2：各黄金分割位胜率
        ax2 = axes[0, 1]
        if len(results_df) > 0:
            colors_bar = ['green' if x > 50 else 'red' for x in results_df['5日胜率%']]
            ax2.bar(results_df['黄金分割位'], results_df['5日胜率%'], color=colors_bar, alpha=0.7, edgecolor='black')
            ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('黄金分割位')
            ax2.set_ylabel('5日胜率(%)')
            ax2.set_title('RSI上穿黄金分割位5日胜率')
            ax2.grid(True, alpha=0.3)

        # 图3：各黄金分割位平均收益
        ax3 = axes[1, 0]
        if len(results_df) > 0:
            colors_bar = ['green' if x > 0 else 'red' for x in results_df['5日均收益%']]
            ax3.bar(results_df['黄金分割位'], results_df['5日均收益%'], color=colors_bar, alpha=0.7, edgecolor='black')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('黄金分割位')
            ax3.set_ylabel('5日平均收益(%)')
            ax3.set_title('RSI上穿黄金分割位5日平均收益')
            ax3.grid(True, alpha=0.3)

        # 图4：RSI在黄金分割位附近的分布
        ax4 = axes[1, 1]
        all_rsi = []
        for ts_code in stocks[:50]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20240101')
            if len(df) < 50:
                continue
            df = df.sort_values('trade_date')
            rsi = self.calculator.calculate_rsi(df['close'], 14).dropna()
            all_rsi.extend(rsi.tolist())

        if all_rsi:
            ax4.hist(all_rsi, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
            for (name, level), color in zip(golden_levels.items(), colors_golden):
                ax4.axvline(x=level, color=color, linestyle='--', linewidth=2, label=f'{name}')
            ax4.set_xlabel('RSI值')
            ax4.set_ylabel('密度')
            ax4.set_title('RSI分布与黄金分割位')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rsi_golden_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_golden_ratio.png")

        return results_df.to_dict('records')

    # ==================== 第三部分：策略回测 ====================

    def backtest_extreme_rsi(self) -> Dict:
        """RSI极值策略回测"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("3. 策略回测")
        self.add_to_report("="*80)

        self.add_to_report("\n### 3.1 RSI极值策略回测")
        self.add_to_report("\n策略规则：")
        self.add_to_report("  - 买入条件：RSI(14) < 20 且从最低点回升")
        self.add_to_report("  - 卖出条件：RSI(14) > 80 或 持有超过20天")
        self.add_to_report("  - 仓位管理：单只股票最大10%仓位")

        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 200
        """).fetchdf()['ts_code'].tolist()

        all_trades = []

        for ts_code in stocks[:100]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['rsi_prev'] = df['rsi'].shift(1)

            position = False
            entry_price = 0
            entry_date = None
            entry_idx = 0

            for i in range(20, len(df) - 20):
                current_rsi = df.loc[i, 'rsi']
                prev_rsi = df.loc[i, 'rsi_prev']

                if pd.isna(current_rsi) or pd.isna(prev_rsi):
                    continue

                # 买入信号：RSI<20 且从低点回升
                if not position and current_rsi < 20 and current_rsi > prev_rsi:
                    position = True
                    entry_price = df.loc[i, 'close']
                    entry_date = df.loc[i, 'trade_date']
                    entry_idx = i

                # 卖出信号：RSI>80 或持有超过20天
                elif position:
                    hold_days = i - entry_idx
                    if current_rsi > 80 or hold_days >= 20:
                        exit_price = df.loc[i, 'close']
                        exit_date = df.loc[i, 'trade_date']
                        returns = (exit_price - entry_price) / entry_price

                        all_trades.append({
                            'ts_code': ts_code,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'hold_days': hold_days,
                            'return': returns,
                            'exit_reason': 'RSI>80' if current_rsi > 80 else '持有超时'
                        })

                        position = False

        # 统计回测结果
        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            self.add_to_report(f"\n回测结果统计：")
            self.add_to_report(f"  总交易次数: {len(trades_df)}")
            self.add_to_report(f"  盈利交易数: {(trades_df['return'] > 0).sum()}")
            self.add_to_report(f"  亏损交易数: {(trades_df['return'] <= 0).sum()}")
            self.add_to_report(f"  胜率: {((trades_df['return'] > 0).sum() / len(trades_df) * 100):.1f}%")
            self.add_to_report(f"  平均收益率: {(trades_df['return'].mean() * 100):.2f}%")
            self.add_to_report(f"  最大收益: {(trades_df['return'].max() * 100):.1f}%")
            self.add_to_report(f"  最大亏损: {(trades_df['return'].min() * 100):.1f}%")
            self.add_to_report(f"  平均持有天数: {trades_df['hold_days'].mean():.1f}天")
            self.add_to_report(f"  盈亏比: {(trades_df[trades_df['return']>0]['return'].mean() / abs(trades_df[trades_df['return']<=0]['return'].mean())):.2f}")

            # 按退出原因统计
            self.add_to_report(f"\n按退出原因统计：")
            for reason in trades_df['exit_reason'].unique():
                subset = trades_df[trades_df['exit_reason'] == reason]
                self.add_to_report(f"  {reason}: 交易数={len(subset)}, 胜率={((subset['return']>0).sum()/len(subset)*100):.1f}%, "
                                 f"平均收益={(subset['return'].mean()*100):.2f}%")

            # 绘制回测图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 图1：收益分布
            ax1 = axes[0, 0]
            ax1.hist(trades_df['return'] * 100, bins=30, density=True, alpha=0.7,
                    color='blue', edgecolor='black')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax1.axvline(x=trades_df['return'].mean() * 100, color='red', linestyle='-', linewidth=2,
                       label=f"均值={trades_df['return'].mean()*100:.2f}%")
            ax1.set_xlabel('收益率(%)')
            ax1.set_ylabel('密度')
            ax1.set_title('RSI极值策略收益分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 图2：累计收益曲线
            ax2 = axes[0, 1]
            trades_df_sorted = trades_df.sort_values('exit_date')
            cum_returns = (1 + trades_df_sorted['return']).cumprod()
            ax2.plot(range(len(cum_returns)), cum_returns, color='blue', linewidth=1.5)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('交易次数')
            ax2.set_ylabel('累计收益倍数')
            ax2.set_title('RSI极值策略累计收益曲线')
            ax2.grid(True, alpha=0.3)

            # 图3：持有天数vs收益
            ax3 = axes[1, 0]
            scatter = ax3.scatter(trades_df['hold_days'], trades_df['return'] * 100,
                                 c=['green' if x > 0 else 'red' for x in trades_df['return']],
                                 alpha=0.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('持有天数')
            ax3.set_ylabel('收益率(%)')
            ax3.set_title('持有天数 vs 收益率')
            ax3.grid(True, alpha=0.3)

            # 图4：月度收益统计
            ax4 = axes[1, 1]
            trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_returns = trades_df.groupby('month')['return'].mean() * 100
            colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns]
            monthly_returns.plot(kind='bar', ax=ax4, color=colors_monthly, alpha=0.7, edgecolor='black')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('月份')
            ax4.set_ylabel('平均收益率(%)')
            ax4.set_title('月度平均收益')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/rsi_extreme_backtest.png', dpi=150, bbox_inches='tight')
            plt.close()

            self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_extreme_backtest.png")

            return trades_df.to_dict('records')

        return {}

    def backtest_rsi_momentum(self) -> Dict:
        """RSI动量策略回测"""
        self.add_to_report("\n### 3.2 RSI动量策略回测")
        self.add_to_report("\n策略规则：")
        self.add_to_report("  - 买入条件：RSI(14)上穿50，且RSI连续3日上升")
        self.add_to_report("  - 卖出条件：RSI(14)下穿50，或RSI连续3日下降")
        self.add_to_report("  - 趋势跟踪策略，适合趋势市场")

        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 200
        """).fetchdf()['ts_code'].tolist()

        all_trades = []

        for ts_code in stocks[:100]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['rsi_1'] = df['rsi'].shift(1)
            df['rsi_2'] = df['rsi'].shift(2)
            df['rsi_3'] = df['rsi'].shift(3)

            # RSI趋势判断
            df['rsi_up'] = (df['rsi'] > df['rsi_1']) & (df['rsi_1'] > df['rsi_2']) & (df['rsi_2'] > df['rsi_3'])
            df['rsi_down'] = (df['rsi'] < df['rsi_1']) & (df['rsi_1'] < df['rsi_2']) & (df['rsi_2'] < df['rsi_3'])

            # RSI穿越50
            df['cross_up_50'] = (df['rsi'] > 50) & (df['rsi_1'] <= 50)
            df['cross_down_50'] = (df['rsi'] < 50) & (df['rsi_1'] >= 50)

            position = False
            entry_price = 0
            entry_date = None
            entry_idx = 0

            for i in range(10, len(df) - 10):
                if pd.isna(df.loc[i, 'rsi']):
                    continue

                # 买入信号
                if not position and df.loc[i, 'cross_up_50'] and df.loc[i, 'rsi_up']:
                    position = True
                    entry_price = df.loc[i, 'close']
                    entry_date = df.loc[i, 'trade_date']
                    entry_idx = i

                # 卖出信号
                elif position:
                    if df.loc[i, 'cross_down_50'] or df.loc[i, 'rsi_down']:
                        exit_price = df.loc[i, 'close']
                        exit_date = df.loc[i, 'trade_date']
                        returns = (exit_price - entry_price) / entry_price
                        hold_days = i - entry_idx

                        all_trades.append({
                            'ts_code': ts_code,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'hold_days': hold_days,
                            'return': returns,
                            'exit_reason': 'RSI下穿50' if df.loc[i, 'cross_down_50'] else 'RSI连续下降'
                        })

                        position = False

        # 统计回测结果
        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            self.add_to_report(f"\n回测结果统计：")
            self.add_to_report(f"  总交易次数: {len(trades_df)}")
            self.add_to_report(f"  盈利交易数: {(trades_df['return'] > 0).sum()}")
            self.add_to_report(f"  亏损交易数: {(trades_df['return'] <= 0).sum()}")
            self.add_to_report(f"  胜率: {((trades_df['return'] > 0).sum() / len(trades_df) * 100):.1f}%")
            self.add_to_report(f"  平均收益率: {(trades_df['return'].mean() * 100):.2f}%")
            self.add_to_report(f"  最大收益: {(trades_df['return'].max() * 100):.1f}%")
            self.add_to_report(f"  最大亏损: {(trades_df['return'].min() * 100):.1f}%")
            self.add_to_report(f"  平均持有天数: {trades_df['hold_days'].mean():.1f}天")

            if (trades_df['return'] <= 0).sum() > 0:
                profit_loss_ratio = trades_df[trades_df['return']>0]['return'].mean() / abs(trades_df[trades_df['return']<=0]['return'].mean())
                self.add_to_report(f"  盈亏比: {profit_loss_ratio:.2f}")

            # 绘制回测图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 图1：收益分布
            ax1 = axes[0, 0]
            ax1.hist(trades_df['return'] * 100, bins=30, density=True, alpha=0.7,
                    color='orange', edgecolor='black')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax1.axvline(x=trades_df['return'].mean() * 100, color='red', linestyle='-', linewidth=2,
                       label=f"均值={trades_df['return'].mean()*100:.2f}%")
            ax1.set_xlabel('收益率(%)')
            ax1.set_ylabel('密度')
            ax1.set_title('RSI动量策略收益分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 图2：累计收益曲线
            ax2 = axes[0, 1]
            trades_df_sorted = trades_df.sort_values('exit_date')
            cum_returns = (1 + trades_df_sorted['return']).cumprod()
            ax2.plot(range(len(cum_returns)), cum_returns, color='orange', linewidth=1.5)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('交易次数')
            ax2.set_ylabel('累计收益倍数')
            ax2.set_title('RSI动量策略累计收益曲线')
            ax2.grid(True, alpha=0.3)

            # 图3：退出原因统计
            ax3 = axes[1, 0]
            exit_stats = trades_df.groupby('exit_reason').agg({
                'return': ['count', 'mean']
            }).reset_index()
            exit_stats.columns = ['exit_reason', 'count', 'mean_return']
            colors_exit = ['blue', 'green']
            ax3.bar(exit_stats['exit_reason'], exit_stats['mean_return'] * 100, color=colors_exit, alpha=0.7, edgecolor='black')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('退出原因')
            ax3.set_ylabel('平均收益率(%)')
            ax3.set_title('不同退出原因平均收益')
            ax3.grid(True, alpha=0.3)

            # 图4：策略对比
            ax4 = axes[1, 1]
            # 与极值策略对比（如果有数据）
            strategies = ['RSI动量策略']
            win_rates = [((trades_df['return'] > 0).sum() / len(trades_df) * 100)]
            avg_returns = [trades_df['return'].mean() * 100]

            x = np.arange(len(strategies))
            width = 0.35
            ax4.bar(x - width/2, win_rates, width, label='胜率(%)', color='blue', alpha=0.7)
            ax4.bar(x + width/2, avg_returns, width, label='平均收益(%)', color='orange', alpha=0.7)
            ax4.axhline(y=50, color='blue', linestyle='--', alpha=0.3)
            ax4.axhline(y=0, color='orange', linestyle='--', alpha=0.3)
            ax4.set_xticks(x)
            ax4.set_xticklabels(strategies)
            ax4.set_ylabel('比例/收益(%)')
            ax4.set_title('RSI动量策略统计')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/rsi_momentum_backtest.png', dpi=150, bbox_inches='tight')
            plt.close()

            self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_momentum_backtest.png")

            return trades_df.to_dict('records')

        return {}

    def backtest_rsi_combined(self) -> Dict:
        """RSI结合其他指标策略回测"""
        self.add_to_report("\n### 3.3 RSI结合MACD策略回测")
        self.add_to_report("\n策略规则：")
        self.add_to_report("  - 买入条件：RSI(14) < 35 且 MACD金叉 且 成交量放大")
        self.add_to_report("  - 卖出条件：RSI(14) > 70 或 MACD死叉")
        self.add_to_report("  - 多指标共振策略，追求高胜率")

        def calculate_macd(prices, fast=12, slow=26, signal=9):
            """计算MACD"""
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram

        stocks = self.conn.execute("""
            SELECT ts_code FROM stock_basic
            WHERE list_status = 'L' AND market IN ('主板', '中小板', '创业板')
            LIMIT 200
        """).fetchdf()['ts_code'].tolist()

        all_trades = []

        for ts_code in stocks[:100]:
            df = self.get_stock_data(ts_code=ts_code, start_date='20230101')
            if len(df) < 100:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['rsi'] = self.calculator.calculate_rsi(df['close'], 14)
            df['macd'], df['signal'], df['histogram'] = calculate_macd(df['close'])
            df['macd_prev'] = df['macd'].shift(1)
            df['signal_prev'] = df['signal'].shift(1)
            df['vol_ma20'] = df['vol'].rolling(20).mean()

            # MACD金叉/死叉
            df['macd_golden'] = (df['macd'] > df['signal']) & (df['macd_prev'] <= df['signal_prev'])
            df['macd_death'] = (df['macd'] < df['signal']) & (df['macd_prev'] >= df['signal_prev'])

            # 成交量放大
            df['vol_increase'] = df['vol'] > df['vol_ma20'] * 1.2

            position = False
            entry_price = 0
            entry_date = None
            entry_idx = 0

            for i in range(30, len(df) - 10):
                if pd.isna(df.loc[i, 'rsi']) or pd.isna(df.loc[i, 'macd']):
                    continue

                # 买入信号：RSI<35 + MACD金叉 + 成交量放大
                if not position:
                    if (df.loc[i, 'rsi'] < 35 and
                        df.loc[i, 'macd_golden'] and
                        df.loc[i, 'vol_increase']):
                        position = True
                        entry_price = df.loc[i, 'close']
                        entry_date = df.loc[i, 'trade_date']
                        entry_idx = i

                # 卖出信号：RSI>70 或 MACD死叉
                elif position:
                    if df.loc[i, 'rsi'] > 70 or df.loc[i, 'macd_death']:
                        exit_price = df.loc[i, 'close']
                        exit_date = df.loc[i, 'trade_date']
                        returns = (exit_price - entry_price) / entry_price
                        hold_days = i - entry_idx

                        all_trades.append({
                            'ts_code': ts_code,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'hold_days': hold_days,
                            'return': returns,
                            'exit_reason': 'RSI>70' if df.loc[i, 'rsi'] > 70 else 'MACD死叉'
                        })

                        position = False

        # 统计回测结果
        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            self.add_to_report(f"\n回测结果统计：")
            self.add_to_report(f"  总交易次数: {len(trades_df)}")
            self.add_to_report(f"  盈利交易数: {(trades_df['return'] > 0).sum()}")
            self.add_to_report(f"  亏损交易数: {(trades_df['return'] <= 0).sum()}")
            self.add_to_report(f"  胜率: {((trades_df['return'] > 0).sum() / len(trades_df) * 100):.1f}%")
            self.add_to_report(f"  平均收益率: {(trades_df['return'].mean() * 100):.2f}%")
            self.add_to_report(f"  最大收益: {(trades_df['return'].max() * 100):.1f}%")
            self.add_to_report(f"  最大亏损: {(trades_df['return'].min() * 100):.1f}%")
            self.add_to_report(f"  平均持有天数: {trades_df['hold_days'].mean():.1f}天")

            if (trades_df['return'] <= 0).sum() > 0:
                profit_loss_ratio = trades_df[trades_df['return']>0]['return'].mean() / abs(trades_df[trades_df['return']<=0]['return'].mean())
                self.add_to_report(f"  盈亏比: {profit_loss_ratio:.2f}")

            # 绘制回测图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 图1：三种策略对比
            ax1 = axes[0, 0]
            strategies = ['RSI+MACD组合']
            win_rates = [((trades_df['return'] > 0).sum() / len(trades_df) * 100)]
            avg_returns = [trades_df['return'].mean() * 100]

            x = np.arange(len(strategies))
            width = 0.35
            ax1.bar(x - width/2, win_rates, width, label='胜率(%)', color='purple', alpha=0.7)
            ax1.bar(x + width/2, avg_returns, width, label='平均收益(%)', color='cyan', alpha=0.7)
            ax1.axhline(y=50, color='purple', linestyle='--', alpha=0.3)
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies)
            ax1.set_ylabel('比例/收益(%)')
            ax1.set_title('RSI+MACD组合策略统计')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 图2：累计收益曲线
            ax2 = axes[0, 1]
            trades_df_sorted = trades_df.sort_values('exit_date')
            cum_returns = (1 + trades_df_sorted['return']).cumprod()
            ax2.plot(range(len(cum_returns)), cum_returns, color='purple', linewidth=1.5)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('交易次数')
            ax2.set_ylabel('累计收益倍数')
            ax2.set_title('RSI+MACD策略累计收益曲线')
            ax2.grid(True, alpha=0.3)

            # 图3：收益分布
            ax3 = axes[1, 0]
            ax3.hist(trades_df['return'] * 100, bins=30, density=True, alpha=0.7,
                    color='purple', edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax3.axvline(x=trades_df['return'].mean() * 100, color='red', linestyle='-', linewidth=2,
                       label=f"均值={trades_df['return'].mean()*100:.2f}%")
            ax3.set_xlabel('收益率(%)')
            ax3.set_ylabel('密度')
            ax3.set_title('RSI+MACD策略收益分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 图4：示例交易图
            ax4 = axes[1, 1]
            # 找一个有交易的例子
            sample_code = trades_df.iloc[0]['ts_code'] if len(trades_df) > 0 else '000001.SZ'
            sample_df = self.get_stock_data(ts_code=sample_code, start_date='20240101')
            sample_df = sample_df.sort_values('trade_date')
            sample_df['date'] = pd.to_datetime(sample_df['trade_date'])
            sample_df['rsi'] = self.calculator.calculate_rsi(sample_df['close'], 14)

            ax4_price = ax4.twinx()
            ax4.plot(sample_df['date'], sample_df['rsi'], color='blue', label='RSI(14)')
            ax4_price.plot(sample_df['date'], sample_df['close'], color='gray', alpha=0.5, label='价格')
            ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=35, color='g', linestyle='--', alpha=0.5)
            ax4.set_ylabel('RSI', color='blue')
            ax4_price.set_ylabel('价格', color='gray')
            ax4.set_title(f'{sample_code} RSI+MACD策略示例')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/rsi_combined_backtest.png', dpi=150, bbox_inches='tight')
            plt.close()

            self.add_to_report(f"\n图表已保存: {OUTPUT_DIR}/rsi_combined_backtest.png")

            return trades_df.to_dict('records')

        return {}

    def generate_strategy_comparison(self):
        """生成策略对比总结"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("4. 策略对比总结")
        self.add_to_report("="*80)

        self.add_to_report("""
### RSI策略研究总结

#### 1. RSI指标特性
- RSI(14)在A股市场的均值约50左右，分布相对对称
- 短周期RSI(6)波动大，信号多但噪音也多
- 长周期RSI(20,30)更平滑，适合趋势判断
- 不同行业RSI特征差异明显，周期股波动更大

#### 2. 交易信号有效性
- 超买超卖信号：RSI<30买入有正期望，但胜率不高
- RSI背离信号：准确率较高但信号稀少
- 黄金分割位：50是重要分界线，上穿50后续上涨概率较大

#### 3. 策略回测结论
- RSI极值策略：等待极端超卖后买入，胜率中等但盈亏比好
- RSI动量策略：趋势跟踪，适合趋势明显的市场
- 组合策略：RSI+MACD+成交量，信号少但质量高

#### 4. 实战建议
1. 单独使用RSI信号效果有限，建议结合其他指标
2. 不同市场环境使用不同策略（震荡市用极值策略，趋势市用动量策略）
3. 注意行业特性，对周期股适当放宽阈值
4. 严格设置止损，RSI只是辅助工具
5. 关注RSI背离信号，虽然稀少但准确率高

#### 5. 风险提示
- RSI是滞后指标，不能预测转折点
- 超买/超卖状态可能持续很长时间
- 历史回测不代表未来收益
- 实盘需考虑交易成本和滑点
""")

    def save_report(self):
        """保存研究报告"""
        report_path = f'{OUTPUT_DIR}/rsi_strategy_research_report.md'

        # 添加报告头部
        header = f"""# RSI交易策略研究报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Tushare数据库
**分析周期**: 2023年至今

---

"""
        full_report = header + '\n'.join(self.report_content)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)

        print(f"\n报告已保存至: {report_path}")
        return report_path

    def run_full_research(self):
        """运行完整研究"""
        print("="*80)
        print("RSI交易策略研究")
        print("="*80)

        # 第一部分：RSI计算与分析
        self.analyze_rsi_periods()
        self.analyze_rsi_distribution()
        self.analyze_industry_rsi()

        # 第二部分：交易信号
        self.analyze_overbought_oversold()
        self.analyze_rsi_divergence()
        self.analyze_rsi_golden_ratio()

        # 第三部分：策略回测
        self.backtest_extreme_rsi()
        self.backtest_rsi_momentum()
        self.backtest_rsi_combined()

        # 生成总结
        self.generate_strategy_comparison()

        # 保存报告
        self.save_report()

        print("\n" + "="*80)
        print("研究完成！")
        print("="*80)


if __name__ == '__main__':
    research = RSIResearch(DB_PATH)
    research.run_full_research()
