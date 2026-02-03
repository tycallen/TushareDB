#!/usr/bin/env python3
"""
ST股票深度分析
================
补充研究：
1. ST股票历史收益分析
2. ST vs 非ST股票对比
3. 摘帽效应量化分析
4. ST股票估值特征
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
REPORT_DIR = "/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research"

os.makedirs(REPORT_DIR, exist_ok=True)


class STDeepAnalysis:
    """ST股票深度分析"""

    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_lines = []

    def add_line(self, line):
        self.report_lines.append(line)

    def get_st_stocks_current(self):
        """获取当前ST股票（仅上市状态）"""
        query = """
        SELECT ts_code, name, industry, area, list_date, market
        FROM stock_basic
        WHERE name LIKE '%ST%' AND list_status = 'L'
        ORDER BY ts_code
        """
        return self.conn.execute(query).fetchdf()

    def get_non_st_stocks_sample(self, n=100):
        """获取非ST股票样本"""
        query = f"""
        SELECT ts_code, name, industry, area, list_date, market
        FROM stock_basic
        WHERE name NOT LIKE '%ST%' AND list_status = 'L'
        ORDER BY RANDOM()
        LIMIT {n}
        """
        return self.conn.execute(query).fetchdf()

    def analyze_st_returns_2024(self):
        """分析2024年ST股票收益表现"""
        # 获取当前ST股票
        st_stocks = self.get_st_stocks_current()
        st_codes = st_stocks['ts_code'].tolist()

        if not st_codes:
            return None, None

        codes_str = "', '".join(st_codes)

        # 获取2024年首日和末日价格
        query = f"""
        WITH first_day AS (
            SELECT ts_code, close as first_close
            FROM daily
            WHERE ts_code IN ('{codes_str}')
            AND trade_date >= '20240101'
            AND trade_date <= '20240115'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) = 1
        ),
        last_day AS (
            SELECT ts_code, close as last_close
            FROM daily
            WHERE ts_code IN ('{codes_str}')
            AND trade_date >= '20241201'
            AND trade_date <= '20241231'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) = 1
        )
        SELECT f.ts_code, f.first_close, l.last_close,
               (l.last_close - f.first_close) / f.first_close * 100 as return_pct
        FROM first_day f
        JOIN last_day l ON f.ts_code = l.ts_code
        WHERE f.first_close > 0
        """
        st_returns = self.conn.execute(query).fetchdf()

        # 获取同期大盘收益
        index_query = """
        WITH first_day AS (
            SELECT close as first_close
            FROM index_daily
            WHERE ts_code = '000001.SH'
            AND trade_date >= '20240101'
            ORDER BY trade_date
            LIMIT 1
        ),
        last_day AS (
            SELECT close as last_close
            FROM index_daily
            WHERE ts_code = '000001.SH'
            AND trade_date >= '20241201'
            AND trade_date <= '20241231'
            ORDER BY trade_date DESC
            LIMIT 1
        )
        SELECT f.first_close, l.last_close,
               (l.last_close - f.first_close) / f.first_close * 100 as return_pct
        FROM first_day f, last_day l
        """
        index_return = self.conn.execute(index_query).fetchdf()

        return st_returns, index_return

    def analyze_st_vs_non_st(self):
        """对比ST与非ST股票的表现"""
        # ST股票2024年统计
        st_query = """
        SELECT
            sb.ts_code,
            AVG(d.pct_chg) as avg_return,
            STDDEV(d.pct_chg) as volatility,
            SUM(CASE WHEN d.pct_chg >= 4.9 THEN 1 ELSE 0 END) as limit_up_days,
            SUM(CASE WHEN d.pct_chg <= -4.9 THEN 1 ELSE 0 END) as limit_down_days,
            AVG(d.vol) as avg_volume,
            COUNT(*) as trading_days
        FROM stock_basic sb
        JOIN daily d ON sb.ts_code = d.ts_code
        WHERE sb.name LIKE '%ST%'
        AND sb.list_status = 'L'
        AND d.trade_date >= '20240101'
        AND d.trade_date <= '20241231'
        GROUP BY sb.ts_code
        """
        st_stats = self.conn.execute(st_query).fetchdf()

        # 非ST股票2024年统计（样本）
        non_st_query = """
        SELECT
            sb.ts_code,
            AVG(d.pct_chg) as avg_return,
            STDDEV(d.pct_chg) as volatility,
            SUM(CASE WHEN d.pct_chg >= 9.9 THEN 1 ELSE 0 END) as limit_up_days,
            SUM(CASE WHEN d.pct_chg <= -9.9 THEN 1 ELSE 0 END) as limit_down_days,
            AVG(d.vol) as avg_volume,
            COUNT(*) as trading_days
        FROM stock_basic sb
        JOIN daily d ON sb.ts_code = d.ts_code
        WHERE sb.name NOT LIKE '%ST%'
        AND sb.list_status = 'L'
        AND d.trade_date >= '20240101'
        AND d.trade_date <= '20241231'
        GROUP BY sb.ts_code
        """
        non_st_stats = self.conn.execute(non_st_query).fetchdf()

        return st_stats, non_st_stats

    def analyze_st_valuation(self):
        """分析ST股票估值特征"""
        # 获取最新交易日
        latest_date_query = "SELECT MAX(trade_date) FROM daily_basic"
        latest_date = self.conn.execute(latest_date_query).fetchone()[0]

        # ST股票估值
        st_val_query = f"""
        SELECT
            sb.ts_code, sb.name, sb.industry,
            db.pe_ttm, db.pb, db.total_mv, db.circ_mv,
            db.turnover_rate
        FROM stock_basic sb
        JOIN daily_basic db ON sb.ts_code = db.ts_code
        WHERE sb.name LIKE '%ST%'
        AND sb.list_status = 'L'
        AND db.trade_date = '{latest_date}'
        AND db.total_mv IS NOT NULL
        """
        st_valuation = self.conn.execute(st_val_query).fetchdf()

        # 非ST股票估值（对比）
        non_st_val_query = f"""
        SELECT
            AVG(db.pe_ttm) as avg_pe,
            AVG(db.pb) as avg_pb,
            AVG(db.total_mv) as avg_mv,
            AVG(db.turnover_rate) as avg_turnover
        FROM stock_basic sb
        JOIN daily_basic db ON sb.ts_code = db.ts_code
        WHERE sb.name NOT LIKE '%ST%'
        AND sb.list_status = 'L'
        AND db.trade_date = '{latest_date}'
        AND db.total_mv IS NOT NULL
        AND db.pe_ttm > 0 AND db.pe_ttm < 500
        """
        non_st_avg = self.conn.execute(non_st_val_query).fetchdf()

        return st_valuation, non_st_avg

    def analyze_st_market_cap(self):
        """分析ST股票市值分布"""
        latest_date_query = "SELECT MAX(trade_date) FROM daily_basic"
        latest_date = self.conn.execute(latest_date_query).fetchone()[0]

        query = f"""
        SELECT
            sb.ts_code, sb.name,
            db.total_mv / 10000 as total_mv_yi,  -- 转换为亿元
            db.circ_mv / 10000 as circ_mv_yi
        FROM stock_basic sb
        JOIN daily_basic db ON sb.ts_code = db.ts_code
        WHERE sb.name LIKE '%ST%'
        AND sb.list_status = 'L'
        AND db.trade_date = '{latest_date}'
        AND db.total_mv IS NOT NULL
        ORDER BY db.total_mv DESC
        """
        return self.conn.execute(query).fetchdf()

    def analyze_industry_st_ratio(self):
        """分析各行业ST股票占比"""
        query = """
        WITH industry_stats AS (
            SELECT
                industry,
                COUNT(*) as total,
                SUM(CASE WHEN name LIKE '%ST%' THEN 1 ELSE 0 END) as st_count
            FROM stock_basic
            WHERE list_status = 'L' AND industry IS NOT NULL
            GROUP BY industry
            HAVING COUNT(*) >= 10
        )
        SELECT
            industry,
            total as 行业总数,
            st_count as ST数量,
            ROUND(st_count * 100.0 / total, 2) as ST占比
        FROM industry_stats
        ORDER BY st_count DESC
        LIMIT 20
        """
        return self.conn.execute(query).fetchdf()

    def create_comprehensive_charts(self, st_returns, st_stats, non_st_stats,
                                    st_valuation, market_cap_data, industry_ratio):
        """创建综合图表"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # 1. ST股票2024年收益分布
        ax1 = axes[0, 0]
        if st_returns is not None and not st_returns.empty:
            returns = st_returns['return_pct'].dropna()
            ax1.hist(returns, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
            ax1.axvline(returns.mean(), color='blue', linestyle='--', linewidth=2,
                       label=f'平均: {returns.mean():.1f}%')
            ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2,
                       label=f'中位数: {returns.median():.1f}%')
            ax1.axvline(0, color='black', linestyle='-', linewidth=1)
            ax1.set_xlabel('年度收益率 (%)')
            ax1.set_ylabel('股票数量')
            ax1.set_title('ST股票2024年收益分布', fontsize=14, fontweight='bold')
            ax1.legend()

        # 2. ST vs 非ST 波动率对比
        ax2 = axes[0, 1]
        st_vol = st_stats['volatility'].dropna()
        non_st_vol = non_st_stats['volatility'].dropna()
        bp = ax2.boxplot([st_vol, non_st_vol], labels=['ST股票', '非ST股票'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#3498db')
        ax2.set_ylabel('日涨跌幅标准差 (%)')
        ax2.set_title('ST vs 非ST 波动率对比', fontsize=14, fontweight='bold')

        # 3. ST股票市值分布
        ax3 = axes[1, 0]
        if not market_cap_data.empty:
            mv_data = market_cap_data['total_mv_yi'].dropna()
            # 分组
            bins = [0, 10, 20, 30, 50, 100, 500, 1000]
            labels = ['<10亿', '10-20亿', '20-30亿', '30-50亿', '50-100亿', '100-500亿', '>500亿']
            mv_groups = pd.cut(mv_data, bins=bins, labels=labels, right=False)
            mv_counts = mv_groups.value_counts().sort_index()
            ax3.bar(mv_counts.index, mv_counts.values, color='#9b59b6')
            ax3.set_xlabel('总市值')
            ax3.set_ylabel('股票数量')
            ax3.set_title('ST股票市值分布', fontsize=14, fontweight='bold')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. 行业ST占比
        ax4 = axes[1, 1]
        if not industry_ratio.empty:
            top_industries = industry_ratio.head(15)
            ax4.barh(range(len(top_industries)), top_industries['ST占比'].values,
                    color='#f39c12')
            ax4.set_yticks(range(len(top_industries)))
            ax4.set_yticklabels(top_industries['industry'].values)
            ax4.set_xlabel('ST股票占比 (%)')
            ax4.set_title('行业ST股票占比 (Top 15)', fontsize=14, fontweight='bold')
            ax4.invert_yaxis()
            for i, v in enumerate(top_industries['ST占比'].values):
                ax4.text(v + 0.2, i, f'{v:.1f}%', va='center')

        # 5. ST vs 非ST 涨停/跌停对比
        ax5 = axes[2, 0]
        st_limit_up = st_stats['limit_up_days'].mean()
        st_limit_down = st_stats['limit_down_days'].mean()
        non_st_limit_up = non_st_stats['limit_up_days'].mean()
        non_st_limit_down = non_st_stats['limit_down_days'].mean()

        x = np.arange(2)
        width = 0.35
        bars1 = ax5.bar(x - width/2, [st_limit_up, non_st_limit_up], width,
                       label='涨停天数', color='#e74c3c')
        bars2 = ax5.bar(x + width/2, [st_limit_down, non_st_limit_down], width,
                       label='跌停天数', color='#27ae60')
        ax5.set_ylabel('平均天数')
        ax5.set_title('2024年平均涨跌停天数对比', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['ST股票', '非ST股票'])
        ax5.legend()

        # 6. ST股票PE分布
        ax6 = axes[2, 1]
        if not st_valuation.empty:
            pe_data = st_valuation['pe_ttm'].dropna()
            pe_data = pe_data[(pe_data > -200) & (pe_data < 200)]  # 过滤极端值
            if not pe_data.empty:
                ax6.hist(pe_data, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
                ax6.axvline(pe_data.mean(), color='red', linestyle='--',
                           label=f'平均: {pe_data.mean():.1f}')
                ax6.axvline(0, color='black', linestyle='-', linewidth=1)
                ax6.set_xlabel('PE (TTM)')
                ax6.set_ylabel('股票数量')
                ax6.set_title('ST股票PE分布', fontsize=14, fontweight='bold')
                ax6.legend()

        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}/st_deep_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_deep_report(self):
        """生成深度分析报告"""
        print("=" * 60)
        print("ST股票深度分析")
        print("=" * 60)

        # 1. 收益分析
        print("\n[1/5] 分析ST股票2024年收益...")
        st_returns, index_return = self.analyze_st_returns_2024()

        self.add_line("# ST股票深度分析报告\n")
        self.add_line(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.add_line("\n## 一、2024年收益表现\n")

        if st_returns is not None and not st_returns.empty:
            returns = st_returns['return_pct'].dropna()
            positive_count = (returns > 0).sum()
            negative_count = (returns < 0).sum()

            self.add_line(f"**ST股票收益统计（2024年）：**\n")
            self.add_line(f"- 样本数量：{len(returns)}只\n")
            self.add_line(f"- 平均收益：{returns.mean():.2f}%\n")
            self.add_line(f"- 中位数收益：{returns.median():.2f}%\n")
            self.add_line(f"- 最高收益：{returns.max():.2f}%\n")
            self.add_line(f"- 最低收益：{returns.min():.2f}%\n")
            self.add_line(f"- 上涨股票：{positive_count}只 ({positive_count/len(returns)*100:.1f}%)\n")
            self.add_line(f"- 下跌股票：{negative_count}只 ({negative_count/len(returns)*100:.1f}%)\n")

            if not index_return.empty:
                self.add_line(f"\n**同期上证指数收益：** {index_return['return_pct'].iloc[0]:.2f}%\n")
                self.add_line(f"\n**超额收益（ST vs 大盘）：** {returns.mean() - index_return['return_pct'].iloc[0]:.2f}%\n")

        # 2. ST vs 非ST对比
        print("[2/5] 对比ST与非ST股票...")
        st_stats, non_st_stats = self.analyze_st_vs_non_st()

        self.add_line("\n## 二、ST vs 非ST股票对比\n")
        self.add_line("\n| 指标 | ST股票 | 非ST股票 |\n")
        self.add_line("|------|--------|----------|\n")
        self.add_line(f"| 平均日涨跌幅 | {st_stats['avg_return'].mean():.3f}% | {non_st_stats['avg_return'].mean():.3f}% |\n")
        self.add_line(f"| 波动率(标准差) | {st_stats['volatility'].mean():.2f}% | {non_st_stats['volatility'].mean():.2f}% |\n")
        self.add_line(f"| 平均涨停天数 | {st_stats['limit_up_days'].mean():.1f}天 | {non_st_stats['limit_up_days'].mean():.1f}天 |\n")
        self.add_line(f"| 平均跌停天数 | {st_stats['limit_down_days'].mean():.1f}天 | {non_st_stats['limit_down_days'].mean():.1f}天 |\n")

        # 3. 估值分析
        print("[3/5] 分析ST股票估值...")
        st_valuation, non_st_avg = self.analyze_st_valuation()

        self.add_line("\n## 三、估值特征分析\n")

        if not st_valuation.empty:
            pe_valid = st_valuation['pe_ttm'].dropna()
            pe_valid = pe_valid[(pe_valid > -200) & (pe_valid < 200)]
            pb_valid = st_valuation['pb'].dropna()
            pb_valid = pb_valid[(pb_valid > 0) & (pb_valid < 20)]

            self.add_line(f"**ST股票估值统计：**\n")
            if not pe_valid.empty:
                self.add_line(f"- 平均PE(TTM)：{pe_valid.mean():.1f}\n")
                self.add_line(f"- PE中位数：{pe_valid.median():.1f}\n")
            if not pb_valid.empty:
                self.add_line(f"- 平均PB：{pb_valid.mean():.2f}\n")
                self.add_line(f"- PB中位数：{pb_valid.median():.2f}\n")

            if not non_st_avg.empty:
                self.add_line(f"\n**非ST股票估值对比：**\n")
                self.add_line(f"- 非ST平均PE(TTM)：{non_st_avg['avg_pe'].iloc[0]:.1f}\n")
                self.add_line(f"- 非ST平均PB：{non_st_avg['avg_pb'].iloc[0]:.2f}\n")

        # 4. 市值分布
        print("[4/5] 分析市值分布...")
        market_cap_data = self.analyze_st_market_cap()

        self.add_line("\n## 四、市值分布\n")

        if not market_cap_data.empty:
            mv = market_cap_data['total_mv_yi'].dropna()
            self.add_line(f"**ST股票市值统计（亿元）：**\n")
            self.add_line(f"- 平均市值：{mv.mean():.1f}亿\n")
            self.add_line(f"- 中位数市值：{mv.median():.1f}亿\n")
            self.add_line(f"- 最大市值：{mv.max():.1f}亿\n")
            self.add_line(f"- 最小市值：{mv.min():.1f}亿\n")

            # 市值分组
            small_cap = (mv < 30).sum()
            mid_cap = ((mv >= 30) & (mv < 100)).sum()
            large_cap = (mv >= 100).sum()

            self.add_line(f"\n**市值分组：**\n")
            self.add_line(f"- 小市值(<30亿)：{small_cap}只 ({small_cap/len(mv)*100:.1f}%)\n")
            self.add_line(f"- 中市值(30-100亿)：{mid_cap}只 ({mid_cap/len(mv)*100:.1f}%)\n")
            self.add_line(f"- 大市值(>100亿)：{large_cap}只 ({large_cap/len(mv)*100:.1f}%)\n")

            # Top 10 市值最大的ST股票
            self.add_line("\n**市值最大的ST股票 (Top 10)：**\n")
            self.add_line("| 代码 | 名称 | 总市值(亿) |\n")
            self.add_line("|------|------|------------|\n")
            for _, row in market_cap_data.head(10).iterrows():
                self.add_line(f"| {row['ts_code']} | {row['name']} | {row['total_mv_yi']:.1f} |\n")

        # 5. 行业分析
        print("[5/5] 分析行业ST比例...")
        industry_ratio = self.analyze_industry_st_ratio()

        self.add_line("\n## 五、行业ST股票占比\n")
        self.add_line("\n| 行业 | 行业总数 | ST数量 | ST占比 |\n")
        self.add_line("|------|----------|--------|--------|\n")
        for _, row in industry_ratio.head(15).iterrows():
            self.add_line(f"| {row['industry']} | {row['行业总数']} | {row['ST数量']} | {row['ST占比']}% |\n")

        # 生成图表
        print("\n正在生成图表...")
        self.create_comprehensive_charts(st_returns, st_stats, non_st_stats,
                                         st_valuation, market_cap_data, industry_ratio)

        self.add_line(f"\n![深度分析图表](st_deep_analysis.png)\n")

        # 添加策略性结论
        self.add_line("\n## 六、策略性结论\n")
        self.add_line("""
### 6.1 摘帽博弈要点

**最佳布局时机：**
1. 年报披露季前3个月（1-3月）
2. 公司发布业绩预盈公告后
3. 重大重组方案获批后

**摘帽成功概率提升因子：**
- 连续两季度盈利
- 净资产转正
- 主营业务恢复
- 审计意见改善

### 6.2 风险规避清单

**高危特征（应立即规避）：**
1. 面值退市风险（股价低于1元）
2. 连续20个交易日市值低于3亿
3. 审计意见为否定或无法表示
4. 被证监会立案调查

**中危特征（需谨慎）：**
1. 连续3年亏损
2. 大股东高比例质押
3. 诉讼缠身
4. 核心业务停滞

### 6.3 困境反转策略执行

**买入信号：**
1. 国资/战投入主公告
2. 债务重整方案获批
3. 核心资产出售回款
4. 新业务实质性突破

**持有条件：**
- 设置时间止损（6个月未见转机则退出）
- 严格仓位控制（单票不超过3%）
- 动态跟踪基本面变化
""")

        # 保存报告
        report_path = f'{REPORT_DIR}/st_deep_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(self.report_lines))

        print(f"\n深度分析报告已保存至：{report_path}")
        print(f"图表已保存至：{REPORT_DIR}/st_deep_analysis.png")

    def close(self):
        self.conn.close()


def main():
    analysis = STDeepAnalysis(DB_PATH)
    try:
        analysis.generate_deep_report()
        print("\n" + "=" * 60)
        print("深度分析完成！")
        print("=" * 60)
    finally:
        analysis.close()


if __name__ == "__main__":
    main()
