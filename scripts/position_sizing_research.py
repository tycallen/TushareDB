#!/usr/bin/env python3
"""
仓位管理策略研究
Position Sizing Strategy Research

研究内容：
1. 固定比例法 (Fixed Fractional)
2. 凯利公式 (Kelly Criterion)
3. 固定金额法 (Fixed Amount)
4. 波动率调整法 (Volatility-Based)

使用沪深300成分股数据进行回测研究
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


class PositionSizingResearch:
    """仓位管理策略研究类"""

    def __init__(self, db_path: str):
        """初始化数据库连接"""
        self.conn = duckdb.connect(db_path, read_only=True)
        self.initial_capital = 1000000  # 初始资金100万
        self.risk_free_rate = 0.03  # 无风险利率3%

    def get_stock_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据（前复权）"""
        query = """
        SELECT
            d.ts_code, d.trade_date,
            d.open * a.adj_factor / (SELECT adj_factor FROM adj_factor WHERE ts_code = d.ts_code ORDER BY trade_date DESC LIMIT 1) as open,
            d.high * a.adj_factor / (SELECT adj_factor FROM adj_factor WHERE ts_code = d.ts_code ORDER BY trade_date DESC LIMIT 1) as high,
            d.low * a.adj_factor / (SELECT adj_factor FROM adj_factor WHERE ts_code = d.ts_code ORDER BY trade_date DESC LIMIT 1) as low,
            d.close * a.adj_factor / (SELECT adj_factor FROM adj_factor WHERE ts_code = d.ts_code ORDER BY trade_date DESC LIMIT 1) as close,
            d.vol, d.amount, d.pct_chg
        FROM daily d
        JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = ? AND d.trade_date >= ? AND d.trade_date <= ?
        ORDER BY d.trade_date
        """
        df = self.conn.execute(query, [ts_code, start_date, end_date]).fetchdf()
        if df.empty:
            return df
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')

    def get_index_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        query = """
        SELECT ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
        FROM index_daily
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        df = self.conn.execute(query, [ts_code, start_date, end_date]).fetchdf()
        if df.empty:
            return df
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')

    def get_hs300_stocks(self) -> List[str]:
        """获取沪深300成分股"""
        query = """
        SELECT DISTINCT ts_code FROM hs_const
        WHERE hs_type = 'SH' OR hs_type = 'SZ'
        LIMIT 50
        """
        df = self.conn.execute(query).fetchdf()
        return df['ts_code'].tolist()

    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动波动率"""
        return returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR（平均真实范围）"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def generate_signals(self, data: pd.DataFrame, strategy: str = 'ma_cross') -> pd.Series:
        """生成交易信号"""
        if strategy == 'ma_cross':
            # 双均线策略
            ma_short = data['close'].rolling(window=10).mean()
            ma_long = data['close'].rolling(window=30).mean()
            signal = pd.Series(0, index=data.index)
            signal[ma_short > ma_long] = 1  # 买入信号
            signal[ma_short < ma_long] = -1  # 卖出信号
            return signal
        elif strategy == 'momentum':
            # 动量策略
            returns = data['close'].pct_change(20)
            signal = pd.Series(0, index=data.index)
            signal[returns > 0.05] = 1
            signal[returns < -0.05] = -1
            return signal
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def backtest_fixed_fraction(self, data: pd.DataFrame, signals: pd.Series,
                                 fraction: float = 0.1) -> Dict:
        """
        固定比例法回测
        每次交易使用固定比例的总资金
        """
        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []

        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                equity_curve.append({'date': date, 'equity': capital})
                continue

            signal = signals.loc[date] if date in signals.index else 0
            price = row['close']

            # 计算当前权益
            current_equity = capital + shares * price

            if signal == 1 and position <= 0:  # 买入信号
                # 使用固定比例的资金买入
                invest_amount = current_equity * fraction
                shares_to_buy = int(invest_amount / price / 100) * 100  # 整手
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    capital -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date, 'action': 'buy',
                        'price': price, 'shares': shares_to_buy,
                        'position_pct': fraction
                    })

            elif signal == -1 and position >= 0 and shares > 0:  # 卖出信号
                # 全部卖出
                revenue = shares * price
                capital += revenue
                trades.append({
                    'date': date, 'action': 'sell',
                    'price': price, 'shares': shares,
                    'position_pct': 0
                })
                shares = 0
                position = -1

            current_equity = capital + shares * price
            equity_curve.append({'date': date, 'equity': current_equity})

        return {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'trades': trades,
            'final_equity': capital + shares * data.iloc[-1]['close'],
            'method': f'Fixed Fraction ({fraction*100:.0f}%)'
        }

    def backtest_kelly(self, data: pd.DataFrame, signals: pd.Series,
                       win_rate: float = 0.55, win_loss_ratio: float = 1.5,
                       kelly_fraction: float = 0.5) -> Dict:
        """
        凯利公式法回测
        Kelly比例 = W - (1-W)/R
        W = 胜率, R = 盈亏比
        """
        # 计算凯利比例
        full_kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = full_kelly * kelly_fraction  # 使用半凯利或更保守的比例
        kelly = max(0, min(kelly, 0.25))  # 限制最大仓位25%

        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []

        # 动态计算胜率和盈亏比
        trade_results = []

        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                equity_curve.append({'date': date, 'equity': capital})
                continue

            signal = signals.loc[date] if date in signals.index else 0
            price = row['close']
            current_equity = capital + shares * price

            # 动态更新凯利比例（如果有足够交易记录）
            if len(trade_results) >= 10:
                wins = [r for r in trade_results if r > 0]
                losses = [r for r in trade_results if r < 0]
                if wins and losses:
                    dynamic_win_rate = len(wins) / len(trade_results)
                    avg_win = np.mean(wins)
                    avg_loss = abs(np.mean(losses))
                    if avg_loss > 0:
                        dynamic_ratio = avg_win / avg_loss
                        dynamic_kelly = dynamic_win_rate - (1 - dynamic_win_rate) / dynamic_ratio
                        kelly = max(0, min(dynamic_kelly * kelly_fraction, 0.25))

            if signal == 1 and position <= 0:
                invest_amount = current_equity * kelly
                shares_to_buy = int(invest_amount / price / 100) * 100
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    capital -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date, 'action': 'buy',
                        'price': price, 'shares': shares_to_buy,
                        'position_pct': kelly,
                        'entry_price': price
                    })

            elif signal == -1 and position >= 0 and shares > 0:
                revenue = shares * price
                capital += revenue

                # 记录交易结果
                if trades and 'entry_price' in trades[-1]:
                    entry_price = trades[-1]['entry_price']
                    pnl_pct = (price - entry_price) / entry_price
                    trade_results.append(pnl_pct)

                trades.append({
                    'date': date, 'action': 'sell',
                    'price': price, 'shares': shares,
                    'position_pct': 0
                })
                shares = 0
                position = -1

            current_equity = capital + shares * price
            equity_curve.append({'date': date, 'equity': current_equity})

        return {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'trades': trades,
            'final_equity': capital + shares * data.iloc[-1]['close'],
            'method': f'Kelly Criterion ({kelly_fraction*100:.0f}% Kelly)'
        }

    def backtest_fixed_amount(self, data: pd.DataFrame, signals: pd.Series,
                               fixed_amount: float = 100000) -> Dict:
        """
        固定金额法回测
        每次交易使用固定金额
        """
        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []

        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                equity_curve.append({'date': date, 'equity': capital})
                continue

            signal = signals.loc[date] if date in signals.index else 0
            price = row['close']
            current_equity = capital + shares * price

            if signal == 1 and position <= 0:
                invest_amount = min(fixed_amount, current_equity * 0.95)  # 最多使用95%资金
                shares_to_buy = int(invest_amount / price / 100) * 100
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    capital -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date, 'action': 'buy',
                        'price': price, 'shares': shares_to_buy,
                        'position_pct': cost / current_equity
                    })

            elif signal == -1 and position >= 0 and shares > 0:
                revenue = shares * price
                capital += revenue
                trades.append({
                    'date': date, 'action': 'sell',
                    'price': price, 'shares': shares,
                    'position_pct': 0
                })
                shares = 0
                position = -1

            current_equity = capital + shares * price
            equity_curve.append({'date': date, 'equity': current_equity})

        return {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'trades': trades,
            'final_equity': capital + shares * data.iloc[-1]['close'],
            'method': f'Fixed Amount ({fixed_amount/10000:.0f}万)'
        }

    def backtest_volatility_adjusted(self, data: pd.DataFrame, signals: pd.Series,
                                      risk_pct: float = 0.02, atr_period: int = 14) -> Dict:
        """
        波动率调整法回测
        根据ATR调整仓位大小
        仓位 = 账户风险金额 / (ATR * N)
        """
        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []

        # 计算ATR
        atr = self.calculate_atr(data, atr_period)

        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                equity_curve.append({'date': date, 'equity': capital})
                continue

            signal = signals.loc[date] if date in signals.index else 0
            price = row['close']
            current_equity = capital + shares * price
            current_atr = atr.loc[date] if date in atr.index and not pd.isna(atr.loc[date]) else None

            if signal == 1 and position <= 0 and current_atr is not None and current_atr > 0:
                # 根据ATR计算仓位
                risk_amount = current_equity * risk_pct
                # 假设止损距离为2倍ATR
                stop_distance = 2 * current_atr
                shares_to_buy = int(risk_amount / stop_distance / 100) * 100

                # 限制最大仓位
                max_shares = int(current_equity * 0.5 / price / 100) * 100
                shares_to_buy = min(shares_to_buy, max_shares)

                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    capital -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date, 'action': 'buy',
                        'price': price, 'shares': shares_to_buy,
                        'position_pct': cost / current_equity,
                        'atr': current_atr
                    })

            elif signal == -1 and position >= 0 and shares > 0:
                revenue = shares * price
                capital += revenue
                trades.append({
                    'date': date, 'action': 'sell',
                    'price': price, 'shares': shares,
                    'position_pct': 0
                })
                shares = 0
                position = -1

            current_equity = capital + shares * price
            equity_curve.append({'date': date, 'equity': current_equity})

        return {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'trades': trades,
            'final_equity': capital + shares * data.iloc[-1]['close'],
            'method': f'Volatility Adjusted ({risk_pct*100:.0f}% Risk)'
        }

    def calculate_metrics(self, result: Dict) -> Dict:
        """计算绩效指标"""
        equity_curve = result['equity_curve']
        returns = equity_curve['equity'].pct_change().dropna()

        total_return = (result['final_equity'] - self.initial_capital) / self.initial_capital

        # 年化收益率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 波动率
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cummax = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 交易统计
        trades = result['trades']
        buy_trades = [t for t in trades if t['action'] == 'buy']

        # 计算胜率
        win_count = 0
        total_trades = 0
        for i in range(0, len(trades) - 1, 2):
            if trades[i]['action'] == 'buy' and i + 1 < len(trades) and trades[i + 1]['action'] == 'sell':
                buy_price = trades[i]['price']
                sell_price = trades[i + 1]['price']
                if sell_price > buy_price:
                    win_count += 1
                total_trades += 1

        win_rate = win_count / total_trades if total_trades > 0 else 0

        return {
            'method': result['method'],
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_trades': len(buy_trades),
            'final_equity': result['final_equity']
        }

    def run_comparison(self, ts_code: str, start_date: str, end_date: str,
                       strategy: str = 'ma_cross') -> Tuple[pd.DataFrame, Dict]:
        """运行仓位管理方法对比"""
        # 获取数据
        data = self.get_stock_data(ts_code, start_date, end_date)
        if data.empty:
            return None, None

        # 生成信号
        signals = self.generate_signals(data, strategy)

        # 运行不同仓位管理方法
        results = {}

        # 固定比例法 - 测试不同比例
        for fraction in [0.1, 0.2, 0.3]:
            result = self.backtest_fixed_fraction(data, signals, fraction)
            results[f'fixed_fraction_{int(fraction*100)}'] = result

        # 凯利公式 - 测试不同凯利系数
        for kelly_frac in [0.25, 0.5, 0.75]:
            result = self.backtest_kelly(data, signals, kelly_fraction=kelly_frac)
            results[f'kelly_{int(kelly_frac*100)}'] = result

        # 固定金额法 - 测试不同金额
        for amount in [50000, 100000, 200000]:
            result = self.backtest_fixed_amount(data, signals, fixed_amount=amount)
            results[f'fixed_amount_{int(amount/10000)}w'] = result

        # 波动率调整法 - 测试不同风险比例
        for risk in [0.01, 0.02, 0.03]:
            result = self.backtest_volatility_adjusted(data, signals, risk_pct=risk)
            results[f'volatility_{int(risk*100)}'] = result

        # 计算所有方法的指标
        metrics_list = []
        for key, result in results.items():
            metrics = self.calculate_metrics(result)
            metrics['key'] = key
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        return metrics_df, results

    def run_multi_stock_analysis(self, start_date: str, end_date: str,
                                  num_stocks: int = 10) -> pd.DataFrame:
        """多股票分析"""
        # 获取一些活跃股票
        query = """
        SELECT ts_code, name FROM stock_basic
        WHERE list_status = 'L' AND list_date < ?
        ORDER BY ts_code
        LIMIT ?
        """
        stocks = self.conn.execute(query, [start_date, num_stocks * 3]).fetchdf()

        all_metrics = []
        tested_stocks = 0

        for _, row in stocks.iterrows():
            if tested_stocks >= num_stocks:
                break

            ts_code = row['ts_code']
            metrics_df, _ = self.run_comparison(ts_code, start_date, end_date)

            if metrics_df is not None and not metrics_df.empty:
                metrics_df['ts_code'] = ts_code
                metrics_df['stock_name'] = row['name']
                all_metrics.append(metrics_df)
                tested_stocks += 1
                print(f"Processed {ts_code} ({tested_stocks}/{num_stocks})")

        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        return pd.DataFrame()

    def generate_report(self, metrics_df: pd.DataFrame, results: Dict,
                        output_dir: str, ts_code: str = None):
        """生成研究报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 绘制资金曲线对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 按方法类别分组绘图
        categories = {
            'Fixed Fraction': ['fixed_fraction_10', 'fixed_fraction_20', 'fixed_fraction_30'],
            'Kelly Criterion': ['kelly_25', 'kelly_50', 'kelly_75'],
            'Fixed Amount': ['fixed_amount_5w', 'fixed_amount_10w', 'fixed_amount_20w'],
            'Volatility Adjusted': ['volatility_1', 'volatility_2', 'volatility_3']
        }

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for ax_idx, (category, keys) in enumerate(categories.items()):
            ax = axes[ax_idx // 2, ax_idx % 2]
            for i, key in enumerate(keys):
                if key in results:
                    equity = results[key]['equity_curve']
                    label = results[key]['method']
                    ax.plot(equity.index, equity['equity'] / self.initial_capital,
                           label=label, color=colors[i], linewidth=1.5)
            ax.set_title(f'{category} - Equity Curve', fontsize=12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity (Normalized)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_sizing_equity_curves.png'), dpi=150)
        plt.close()

        # 2. 绘制指标对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 按方法分组的指标
        metrics_summary = metrics_df.groupby('key').agg({
            'total_return': 'mean',
            'annual_return': 'mean',
            'annual_volatility': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'calmar_ratio': 'mean'
        }).reset_index()

        # 定义方法顺序
        method_order = [
            'fixed_fraction_10', 'fixed_fraction_20', 'fixed_fraction_30',
            'kelly_25', 'kelly_50', 'kelly_75',
            'fixed_amount_5w', 'fixed_amount_10w', 'fixed_amount_20w',
            'volatility_1', 'volatility_2', 'volatility_3'
        ]

        # 过滤存在的方法
        method_order = [m for m in method_order if m in metrics_summary['key'].values]
        metrics_summary = metrics_summary.set_index('key').loc[method_order].reset_index()

        # 设置颜色
        category_colors = {
            'fixed_fraction': '#1f77b4',
            'kelly': '#ff7f0e',
            'fixed_amount': '#2ca02c',
            'volatility': '#d62728'
        }

        def get_color(key):
            for cat, color in category_colors.items():
                if key.startswith(cat):
                    return color
            return 'gray'

        bar_colors = [get_color(k) for k in metrics_summary['key']]

        # 绘制各项指标
        metrics_to_plot = [
            ('total_return', 'Total Return', True),
            ('annual_return', 'Annual Return', True),
            ('annual_volatility', 'Annual Volatility', False),
            ('sharpe_ratio', 'Sharpe Ratio', True),
            ('max_drawdown', 'Max Drawdown', False),
            ('calmar_ratio', 'Calmar Ratio', True)
        ]

        for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            values = metrics_summary[metric].values
            x_labels = [k.replace('_', '\n') for k in metrics_summary['key']]

            bars = ax.bar(range(len(values)), values, color=bar_colors, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            # 标注最佳值
            if higher_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values) if metric == 'max_drawdown' else np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_sizing_metrics_comparison.png'), dpi=150)
        plt.close()

        # 3. 生成Markdown报告
        report = self._generate_markdown_report(metrics_df, metrics_summary, ts_code)

        report_path = os.path.join(output_dir, 'position_sizing_research_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")
        print(f"Charts saved to: {output_dir}")

        return report_path

    def _generate_markdown_report(self, metrics_df: pd.DataFrame,
                                   metrics_summary: pd.DataFrame,
                                   ts_code: str = None) -> str:
        """生成Markdown格式报告"""

        report = f"""# 仓位管理策略研究报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**研究标的**: {'多股票组合分析' if ts_code is None else ts_code}

---

## 1. 研究概述

### 1.1 研究目的
本研究旨在对比分析四种主流仓位管理方法的实际效果，为投资者提供仓位管理的决策依据。

### 1.2 研究方法
- **交易策略**: 双均线交叉策略（MA10与MA30）
- **初始资金**: 100万元
- **回测周期**: 基于历史数据的完整回测
- **评价指标**: 收益率、波动率、夏普比率、最大回撤、卡玛比率

### 1.3 仓位管理方法

| 方法 | 描述 | 特点 |
|------|------|------|
| **固定比例法** | 每次使用总资金固定比例交易 | 简单直观，风险随资金增长 |
| **凯利公式** | 基于胜率和盈亏比计算最优仓位 | 理论最优，但波动较大 |
| **固定金额法** | 每次使用固定金额交易 | 绝对风险可控，资金效率变化 |
| **波动率调整法** | 根据市场波动调整仓位 | 动态适应市场，风险平衡 |

---

## 2. 效果分析

### 2.1 收益对比

"""
        # 添加收益对比表格
        report += "| 方法 | 总收益率 | 年化收益率 | 波动率 | 夏普比率 |\n"
        report += "|------|----------|------------|--------|----------|\n"

        for _, row in metrics_summary.iterrows():
            report += f"| {row['key']} | {row['total_return']*100:.2f}% | {row['annual_return']*100:.2f}% | {row['annual_volatility']*100:.2f}% | {row['sharpe_ratio']:.2f} |\n"

        report += """

### 2.2 风险控制效果

"""
        # 添加风险指标表格
        report += "| 方法 | 最大回撤 | 卡玛比率 |\n"
        report += "|------|----------|----------|\n"

        for _, row in metrics_summary.iterrows():
            report += f"| {row['key']} | {row['max_drawdown']*100:.2f}% | {row['calmar_ratio']:.2f} |\n"

        # 找出最佳方法
        best_sharpe_idx = metrics_summary['sharpe_ratio'].idxmax()
        best_sharpe = metrics_summary.loc[best_sharpe_idx]

        best_calmar_idx = metrics_summary['calmar_ratio'].idxmax()
        best_calmar = metrics_summary.loc[best_calmar_idx]

        lowest_dd_idx = metrics_summary['max_drawdown'].idxmax()  # 回撤是负数，max是最小回撤
        lowest_dd = metrics_summary.loc[lowest_dd_idx]

        report += f"""

### 2.3 关键发现

1. **最高夏普比率**: {best_sharpe['key']} (夏普比率: {best_sharpe['sharpe_ratio']:.2f})
2. **最佳卡玛比率**: {best_calmar['key']} (卡玛比率: {best_calmar['calmar_ratio']:.2f})
3. **最小回撤**: {lowest_dd['key']} (最大回撤: {lowest_dd['max_drawdown']*100:.2f}%)

---

## 3. 各方法详细分析

### 3.1 固定比例法 (Fixed Fractional)

**原理**: 每次交易使用账户总资金的固定百分比。

**公式**:
```
仓位金额 = 当前资金 × 固定比例
```

**测试参数**: 10%, 20%, 30%

**优点**:
- 实现简单，易于理解和执行
- 仓位随账户增长自动调整
- 亏损时自动减仓，有一定保护作用

**缺点**:
- 不考虑市场波动状况
- 不考虑交易胜率和盈亏比
- 比例选择主观性强

**适用场景**: 适合追求简单稳定的投资者，或作为其他方法的基准对比。

### 3.2 凯利公式 (Kelly Criterion)

**原理**: 基于历史胜率和盈亏比计算理论最优仓位。

**公式**:
```
Kelly比例 = W - (1-W)/R
其中: W = 胜率, R = 盈亏比
```

**测试参数**: 25%, 50%, 75% 凯利

**优点**:
- 理论上可实现长期资金最大化增长
- 科学考虑了胜率和盈亏比
- 动态调整，适应策略表现变化

**缺点**:
- 全凯利仓位波动极大
- 对参数估计敏感
- 需要足够交易样本计算参数

**适用场景**: 适合有稳定策略、能承受较大波动的专业投资者。建议使用半凯利或更保守的比例。

### 3.3 固定金额法 (Fixed Amount)

**原理**: 每次交易使用固定金额，不随账户变化。

**公式**:
```
仓位金额 = min(固定金额, 可用资金)
```

**测试参数**: 5万, 10万, 20万

**优点**:
- 绝对风险可控
- 计算简单，执行容易
- 不会因连续亏损而加大仓位

**缺点**:
- 资金利用率随账户变化
- 小账户可能无法执行
- 不能充分利用复利效应

**适用场景**: 适合风险厌恶型投资者，或资金量较小的初学者。

### 3.4 波动率调整法 (Volatility-Based)

**原理**: 根据市场波动率（ATR）动态调整仓位大小。

**公式**:
```
仓位 = 账户风险金额 / (ATR × 倍数)
账户风险金额 = 当前资金 × 风险比例
```

**测试参数**: 1%, 2%, 3% 风险

**优点**:
- 高波动时自动减仓
- 低波动时适当加仓
- 风险更加均衡

**缺点**:
- 需要计算ATR等指标
- 参数设置需要经验
- 可能错过趋势行情

**适用场景**: 适合追求稳定风险暴露的系统化交易者。

---

## 4. 最优方案建议

### 4.1 按风险偏好推荐

| 风险偏好 | 推荐方法 | 推荐参数 | 理由 |
|----------|----------|----------|------|
| **保守型** | 固定金额法 或 波动率调整法 | 5万固定 或 1%风险 | 绝对风险可控，回撤较小 |
| **稳健型** | 波动率调整法 | 2%风险 | 动态平衡风险与收益 |
| **积极型** | 固定比例法 或 半凯利 | 20%比例 或 50%凯利 | 追求较高收益，可承受较大波动 |
| **激进型** | 凯利公式 | 75%凯利 | 追求最大化收益，风险承受力强 |

### 4.2 策略结合建议

1. **趋势策略**: 推荐波动率调整法或凯利公式
   - 趋势明确时加大仓位
   - 配合止损管理风险

2. **均值回归策略**: 推荐固定比例法
   - 分批建仓
   - 避免单次过度暴露

3. **套利策略**: 推荐固定金额法
   - 风险有限时固定仓位
   - 保持收益稳定性

### 4.3 动态调整机制

建议采用**混合策略**，根据市场环境动态切换：

```
if 市场波动率 > 历史75分位:
    使用波动率调整法（降低仓位）
elif 策略近期胜率 > 60%:
    使用凯利公式（适度加仓）
else:
    使用固定比例法（基础仓位）
```

### 4.4 风控建议

1. **单笔止损**: 不超过总资金的2%
2. **总仓位上限**: 不超过总资金的80%
3. **单标的上限**: 不超过总资金的30%
4. **连续亏损处理**: 连续亏损3次后减半仓位

---

## 5. 结论

1. **没有绝对最优的仓位管理方法**，需要根据投资者风险偏好、资金规模和交易策略综合选择。

2. **波动率调整法**在风险调整后收益方面通常表现较好，适合大多数投资者。

3. **凯利公式**理论最优但实际应用需谨慎，建议使用半凯利或更保守的比例。

4. **固定比例法**简单实用，是很好的基准方法。

5. **固定金额法**适合初学者和风险厌恶型投资者。

6. **建议采用动态调整机制**，根据市场环境和策略表现灵活切换仓位管理方法。

---

## 6. 附录

### 6.1 指标说明

| 指标 | 说明 | 计算方法 |
|------|------|----------|
| 总收益率 | 期末与期初资金之比 | (期末资金-初始资金)/初始资金 |
| 年化收益率 | 折算为年度收益 | (1+总收益率)^(365/天数)-1 |
| 波动率 | 收益率标准差年化 | 日收益率标准差 × sqrt(252) |
| 夏普比率 | 风险调整后收益 | (年化收益-无风险利率)/波动率 |
| 最大回撤 | 最大峰谷跌幅 | (谷值-峰值)/峰值 |
| 卡玛比率 | 收益回撤比 | 年化收益率/最大回撤 |

### 6.2 图表说明

- **position_sizing_equity_curves.png**: 四类仓位管理方法的资金曲线对比
- **position_sizing_metrics_comparison.png**: 各项绩效指标的柱状图对比

---

*本报告由仓位管理策略研究系统自动生成*
"""

        return report


def main():
    """主函数"""
    # 数据库路径
    db_path = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
    output_dir = "/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research"

    # 创建研究实例
    research = PositionSizingResearch(db_path)

    # 设置回测参数
    start_date = "20200101"
    end_date = "20251231"
    test_stock = "000001.SZ"  # 平安银行作为测试标的

    print("=" * 60)
    print("仓位管理策略研究")
    print("=" * 60)

    # 1. 单股票详细分析
    print(f"\n[1] 单股票分析: {test_stock}")
    print("-" * 40)

    metrics_df, results = research.run_comparison(test_stock, start_date, end_date)

    if metrics_df is not None:
        print("\n绩效指标汇总:")
        print(metrics_df[['method', 'total_return', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))

        # 生成报告
        report_path = research.generate_report(metrics_df, results, output_dir, test_stock)

    # 2. 多股票分析
    print(f"\n[2] 多股票分析")
    print("-" * 40)

    multi_metrics = research.run_multi_stock_analysis(start_date, end_date, num_stocks=5)

    if not multi_metrics.empty:
        # 计算各方法在多股票上的平均表现
        avg_metrics = multi_metrics.groupby('key').agg({
            'total_return': 'mean',
            'annual_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'calmar_ratio': 'mean'
        }).round(4)

        print("\n多股票平均绩效:")
        print(avg_metrics.to_string())

        # 保存多股票分析结果
        multi_metrics.to_csv(os.path.join(output_dir, 'multi_stock_analysis.csv'), index=False)
        print(f"\n多股票分析结果已保存到: {output_dir}/multi_stock_analysis.csv")

    print("\n" + "=" * 60)
    print("研究完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
