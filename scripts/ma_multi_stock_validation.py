#!/usr/bin/env python3
"""
均线交易系统多股票验证
=====================
对多只股票进行均线策略的交叉验证，验证策略的普适性

Author: Claude AI Assistant
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'


class MultiStockMAValidation:
    """多股票均线策略验证"""

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_sample_stocks(self, n: int = 50) -> List[str]:
        """获取样本股票列表"""
        query = """
        SELECT DISTINCT d.ts_code
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE s.list_status = 'L'
          AND d.trade_date >= '20200101'
          AND d.trade_date <= '20251231'
        GROUP BY d.ts_code
        HAVING COUNT(*) >= 1200
        ORDER BY d.ts_code
        LIMIT ?
        """
        result = self.conn.execute(query, [n]).fetchall()
        return [r[0] for r in result]

    def load_stock_data(self, ts_code: str) -> pd.DataFrame:
        """加载股票数据"""
        query = f"""
        SELECT
            d.trade_date,
            d.close,
            d.vol,
            COALESCE(a.adj_factor, 1.0) as adj_factor
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = '{ts_code}'
          AND d.trade_date >= '20200101'
          AND d.trade_date <= '20251231'
        ORDER BY d.trade_date
        """
        df = self.conn.execute(query).fetchdf()

        if len(df) == 0:
            return df

        latest_adj = df['adj_factor'].iloc[-1]
        df['adj_close'] = df['close'] * df['adj_factor'] / latest_adj
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)

        return df

    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均线"""
        return prices.rolling(window=period).mean()

    def test_dual_ma_strategy(self, df: pd.DataFrame,
                               short_period: int,
                               long_period: int) -> Dict:
        """测试双均线策略"""
        if len(df) < long_period + 50:
            return None

        short_ma = self.calculate_ma(df['adj_close'], short_period)
        long_ma = self.calculate_ma(df['adj_close'], long_period)

        signal = (short_ma > long_ma).astype(int)
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns

        # 买入持有收益
        buyhold_returns = returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 买入持有
        bh_total_return = (1 + buyhold_returns).prod() - 1
        bh_annual_return = (1 + bh_total_return) ** (252 / len(df)) - 1

        trades = (signal.diff().abs() > 0).sum()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'bh_total_return': bh_total_return,
            'bh_annual_return': bh_annual_return,
            'outperformance': total_return - bh_total_return
        }

    def run_validation(self, short_period: int = 10, long_period: int = 30):
        """运行多股票验证"""
        print("正在获取样本股票...")
        stocks = self.get_sample_stocks(100)
        print(f"共获取 {len(stocks)} 只股票")

        results = []
        for i, ts_code in enumerate(stocks):
            if (i + 1) % 20 == 0:
                print(f"  正在处理第 {i+1}/{len(stocks)} 只股票...")

            df = self.load_stock_data(ts_code)
            if len(df) < long_period + 50:
                continue

            result = self.test_dual_ma_strategy(df, short_period, long_period)
            if result:
                result['ts_code'] = ts_code
                results.append(result)

        return pd.DataFrame(results)

    def run_parameter_validation(self, param_pairs: List[Tuple[int, int]]):
        """对不同参数组合进行验证"""
        print("正在获取样本股票...")
        stocks = self.get_sample_stocks(50)
        print(f"共获取 {len(stocks)} 只股票")

        all_results = {}
        for short_p, long_p in param_pairs:
            print(f"\n测试参数组合: MA{short_p}/MA{long_p}")
            results = []

            for i, ts_code in enumerate(stocks):
                df = self.load_stock_data(ts_code)
                if len(df) < long_p + 50:
                    continue

                result = self.test_dual_ma_strategy(df, short_p, long_p)
                if result:
                    results.append(result)

            if results:
                all_results[f'{short_p}/{long_p}'] = pd.DataFrame(results)

        return all_results


def main():
    """主函数"""
    validator = MultiStockMAValidation(DB_PATH)

    print("=" * 60)
    print("均线策略多股票交叉验证")
    print("=" * 60)

    # 测试多个参数组合
    param_pairs = [
        (5, 20), (5, 30), (5, 50),
        (10, 30), (10, 60),
        (20, 60), (20, 120)
    ]

    all_results = validator.run_parameter_validation(param_pairs)

    # 汇总结果
    summary = []
    for params, df in all_results.items():
        summary.append({
            '参数组合': params,
            '股票数量': len(df),
            '平均收益率': df['total_return'].mean() * 100,
            '收益率中位数': df['total_return'].median() * 100,
            '平均年化收益': df['annual_return'].mean() * 100,
            '平均夏普比率': df['sharpe_ratio'].mean(),
            '平均最大回撤': df['max_drawdown'].mean() * 100,
            '跑赢比例': (df['outperformance'] > 0).sum() / len(df) * 100,
            '平均超额收益': df['outperformance'].mean() * 100
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 60)
    print("多股票验证结果汇总")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # 保存结果
    summary_df.to_csv(f'{REPORT_DIR}/ma_multi_stock_validation.csv', index=False)

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 平均收益率对比
    ax1 = axes[0, 0]
    x = range(len(summary_df))
    ax1.bar(x, summary_df['平均收益率'], color='steelblue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['参数组合'], rotation=45)
    ax1.set_ylabel('平均收益率 (%)')
    ax1.set_title('不同参数组合的平均收益率')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # 夏普比率对比
    ax2 = axes[0, 1]
    ax2.bar(x, summary_df['平均夏普比率'], color='green', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['参数组合'], rotation=45)
    ax2.set_ylabel('平均夏普比率')
    ax2.set_title('不同参数组合的平均夏普比率')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 跑赢买入持有比例
    ax3 = axes[1, 0]
    ax3.bar(x, summary_df['跑赢比例'], color='orange', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df['参数组合'], rotation=45)
    ax3.set_ylabel('跑赢买入持有比例 (%)')
    ax3.set_title('策略跑赢买入持有的股票比例')
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%基准线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 最大回撤对比
    ax4 = axes[1, 1]
    ax4.bar(x, abs(summary_df['平均最大回撤']), color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary_df['参数组合'], rotation=45)
    ax4.set_ylabel('平均最大回撤 (%, 绝对值)')
    ax4.set_title('不同参数组合的平均最大回撤')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{REPORT_DIR}/ma_multi_stock_validation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 生成验证报告
    report = []
    report.append("# 均线策略多股票交叉验证报告\n")
    report.append(f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**验证股票数量**: {len(all_results[list(all_results.keys())[0]])} 只\n")
    report.append(f"**数据范围**: 2020-01-01 至 2025-12-31\n")
    report.append("\n## 参数组合验证结果\n")
    report.append(summary_df.to_markdown(index=False, floatfmt='.2f'))
    report.append("\n\n## 分析结论\n")

    # 找出最优参数
    best_sharpe_idx = summary_df['平均夏普比率'].idxmax()
    best_outperform_idx = summary_df['跑赢比例'].idxmax()
    best_return_idx = summary_df['平均收益率'].idxmax()

    report.append(f"1. **最高夏普比率参数组合**: {summary_df.loc[best_sharpe_idx, '参数组合']}，夏普比率 {summary_df.loc[best_sharpe_idx, '平均夏普比率']:.2f}\n")
    report.append(f"2. **最高跑赢比例参数组合**: {summary_df.loc[best_outperform_idx, '参数组合']}，跑赢比例 {summary_df.loc[best_outperform_idx, '跑赢比例']:.1f}%\n")
    report.append(f"3. **最高平均收益参数组合**: {summary_df.loc[best_return_idx, '参数组合']}，平均收益 {summary_df.loc[best_return_idx, '平均收益率']:.2f}%\n")

    report.append("\n## 主要发现\n")
    report.append("- 在多股票验证中，均线策略的效果因参数选择而异\n")
    report.append("- 较长周期的均线组合通常具有更低的交易频率和更稳定的表现\n")
    report.append("- 并非所有股票都适合使用均线策略，策略选择需要结合个股特性\n")

    report.append("\n![多股票验证结果](ma_multi_stock_validation.png)\n")

    with open(f'{REPORT_DIR}/ma_multi_stock_validation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n报告已保存到: {REPORT_DIR}/ma_multi_stock_validation_report.md")
    print(f"图片已保存到: {REPORT_DIR}/ma_multi_stock_validation.png")


if __name__ == '__main__':
    main()
