"""
金融股特征研究分析脚本

本脚本用于分析A股金融板块的分类、估值、周期性和风险特征
数据来源：Tushare DuckDB数据库

使用方法：
    python financial_stocks_analysis.py
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FinancialStockAnalyzer:
    """金融股分析器"""

    def __init__(self, db_path: str):
        """初始化分析器

        Args:
            db_path: DuckDB数据库路径
        """
        self.db_path = db_path
        self.conn = None
        self.financial_stocks = None

    def connect(self):
        """连接数据库"""
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def get_financial_stocks(self) -> pd.DataFrame:
        """获取所有金融股列表"""
        self.financial_stocks = self.conn.execute('''
            SELECT DISTINCT ima.ts_code, ima.name, ima.l1_name, ima.l2_name, ima.l3_name,
                CASE
                    WHEN ima.l1_name = '银行' THEN '银行'
                    WHEN ima.l3_name = '证券Ⅲ' THEN '券商'
                    WHEN ima.l3_name = '保险Ⅲ' THEN '保险'
                    WHEN ima.l2_name = '多元金融' THEN '多元金融'
                    ELSE '其他'
                END as category,
                CASE
                    WHEN ima.l1_name = '银行' THEN ima.l2_name
                    WHEN ima.l3_name = '证券Ⅲ' THEN '证券'
                    WHEN ima.l3_name = '保险Ⅲ' THEN '保险'
                    WHEN ima.l2_name = '多元金融' THEN ima.l3_name
                    ELSE ima.l3_name
                END as sub_type
            FROM index_member_all ima
            WHERE (ima.l1_name = '银行' OR ima.l1_name = '非银金融') AND ima.is_new = 'Y'
            AND ima.name NOT LIKE '%退市%'
        ''').fetchdf()
        return self.financial_stocks

    def get_valuation(self) -> pd.DataFrame:
        """获取最新估值数据"""
        valuation = self.conn.execute('''
            WITH latest_date AS (
                SELECT MAX(trade_date) as max_date FROM daily_basic
            )
            SELECT d.ts_code, d.trade_date, d.pe, d.pe_ttm, d.pb,
                   d.dv_ratio, d.dv_ttm, d.total_mv, d.circ_mv
            FROM daily_basic d, latest_date l
            WHERE d.trade_date = l.max_date
        ''').fetchdf()
        return valuation

    def analyze_classification(self):
        """分析金融股分类"""
        print("=" * 80)
        print("金融股分类统计")
        print("=" * 80)

        stocks = self.get_financial_stocks()

        # 按大类统计
        category_stats = stocks.groupby('category').size().reset_index(name='count')
        print("\n各类金融股数量:")
        print(category_stats.to_string(index=False))

        # 银行股细分
        bank_stocks = stocks[stocks['category'] == '银行']
        bank_types = bank_stocks.groupby('sub_type').size().reset_index(name='count')
        print("\n银行股分类:")
        print(bank_types.to_string(index=False))

        # 多元金融细分
        div_stocks = stocks[stocks['category'] == '多元金融']
        div_types = div_stocks.groupby('sub_type').size().reset_index(name='count')
        print("\n多元金融分类:")
        print(div_types.to_string(index=False))

        return stocks

    def analyze_valuation(self):
        """分析估值水平"""
        print("\n" + "=" * 80)
        print("金融股估值分析")
        print("=" * 80)

        stocks = self.get_financial_stocks()
        valuation = self.get_valuation()

        merged = stocks.merge(valuation, on='ts_code', how='left')
        merged = merged.dropna(subset=['pb'])

        # 按类别统计估值
        for cat in ['银行', '券商', '保险', '多元金融']:
            cat_data = merged[merged['category'] == cat]
            if len(cat_data) > 0:
                print(f"\n【{cat}】({len(cat_data)}只)")
                print(f"  PB: 中位数={cat_data['pb'].median():.2f}, 均值={cat_data['pb'].mean():.2f}")
                print(f"  PE_TTM: 中位数={cat_data['pe_ttm'].median():.2f}")
                print(f"  股息率: 中位数={cat_data['dv_ttm'].median():.2f}%")

        return merged

    def analyze_beta(self):
        """分析行业Beta"""
        print("\n" + "=" * 80)
        print("行业Beta分析")
        print("=" * 80)

        # 获取申万金融指数和上证指数
        sw_data = self.conn.execute('''
            SELECT ts_code, trade_date, pct_change
            FROM sw_daily
            WHERE ts_code IN ('801780.SI', '801790.SI')
            ORDER BY trade_date
        ''').fetchdf()

        sh_data = self.conn.execute('''
            SELECT ts_code, trade_date, pct_chg as pct_change
            FROM index_daily
            WHERE ts_code = '000001.SH'
            AND trade_date >= '20210104'
            ORDER BY trade_date
        ''').fetchdf()

        # 转为宽表
        pivot_sw = sw_data.pivot(index='trade_date', columns='ts_code', values='pct_change')
        sh_pct = sh_data.set_index('trade_date')['pct_change']
        pivot_sw['000001.SH'] = sh_pct

        # 计算Beta
        def calculate_beta(returns, market_returns):
            common_idx = returns.dropna().index.intersection(market_returns.dropna().index)
            if len(common_idx) < 30:
                return np.nan
            cov = np.cov(returns[common_idx], market_returns[common_idx])[0, 1]
            var = np.var(market_returns[common_idx])
            return cov / var if var != 0 else np.nan

        market_returns = pivot_sw['000001.SH']

        if '801780.SI' in pivot_sw.columns:
            beta_bank = calculate_beta(pivot_sw['801780.SI'], market_returns)
            print(f"\n银行指数Beta: {beta_bank:.4f}")

        if '801790.SI' in pivot_sw.columns:
            beta_nonbank = calculate_beta(pivot_sw['801790.SI'], market_returns)
            print(f"非银金融Beta: {beta_nonbank:.4f}")

    def analyze_risk(self):
        """分析风险指标"""
        print("\n" + "=" * 80)
        print("风险分析")
        print("=" * 80)

        stocks = self.get_financial_stocks()
        stock_list = stocks['ts_code'].tolist()

        # 获取日收益率数据
        daily_returns = self.conn.execute(f'''
            SELECT ts_code, trade_date, pct_chg
            FROM daily
            WHERE ts_code IN ({','.join([f"'{s}'" for s in stock_list])})
            AND trade_date >= '20240101'
            ORDER BY ts_code, trade_date
        ''').fetchdf()

        # 计算风险指标
        risk_stats = []
        for ts_code in stock_list:
            stock_data = daily_returns[daily_returns['ts_code'] == ts_code]
            if len(stock_data) > 30:
                returns = stock_data['pct_chg'].dropna()
                risk_stats.append({
                    'ts_code': ts_code,
                    'daily_vol': returns.std(),
                    'annual_vol': returns.std() * np.sqrt(250),
                    'var_95': np.percentile(returns, 5)
                })

        risk_df = pd.DataFrame(risk_stats)
        risk_df = risk_df.merge(stocks[['ts_code', 'name', 'category']], on='ts_code')

        # 按类别统计
        for cat in ['银行', '券商', '保险', '多元金融']:
            cat_data = risk_df[risk_df['category'] == cat]
            if len(cat_data) > 0:
                print(f"\n【{cat}】({len(cat_data)}只)")
                print(f"  年化波动率: 中位数={cat_data['annual_vol'].median():.2f}%")
                print(f"  95% VaR: 中位数={cat_data['var_95'].median():.2f}%")

        return risk_df

    def find_low_valuation_stocks(self):
        """筛选低估值股票"""
        print("\n" + "=" * 80)
        print("低估值股票筛选")
        print("=" * 80)

        stocks = self.get_financial_stocks()
        valuation = self.get_valuation()
        merged = stocks.merge(valuation, on='ts_code', how='left')
        merged = merged.dropna(subset=['pb'])

        # 银行股：PB<0.6, 股息率>4%
        bank_low = merged[(merged['category'] == '银行') &
                          (merged['pb'] < 0.6) &
                          (merged['dv_ttm'] > 4)]
        bank_low = bank_low.sort_values('pb')

        print(f"\n低估值银行股 (PB<0.6, 股息率>4%): {len(bank_low)}只")
        cols = ['ts_code', 'name', 'sub_type', 'pb', 'pe_ttm', 'dv_ttm']
        print(bank_low[cols].to_string(index=False))

        # 综合评分
        merged['div_score'] = merged['dv_ttm'].rank(pct=True)
        merged['pb_score'] = merged['pb'].rank(pct=True, ascending=False)
        merged['value_score'] = (merged['div_score'] + merged['pb_score']) / 2

        top_value = merged.nlargest(10, 'value_score')
        print(f"\n综合低估值TOP10:")
        cols = ['ts_code', 'name', 'category', 'pb', 'dv_ttm', 'value_score']
        print(top_value[cols].to_string(index=False))

        return top_value

    def run_full_analysis(self):
        """运行完整分析"""
        self.connect()
        try:
            self.analyze_classification()
            self.analyze_valuation()
            self.analyze_beta()
            self.analyze_risk()
            self.find_low_valuation_stocks()
        finally:
            self.close()


def main():
    """主函数"""
    import os

    # 数据库路径
    db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tushare.db')
    db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        return

    print(f"数据库路径: {db_path}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    analyzer = FinancialStockAnalyzer(db_path)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
