#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务异常检测系统 - Financial Anomaly Detection System
=====================================================

基于 DuckDB 数据库的全面财务风险分析工具

功能模块:
1. 盈余质量分析 (Earnings Quality Analysis)
2. 财务造假预警模型 (Fraud Detection Models)
   - Beneish M-Score
   - Altman Z-Score
   - Piotroski F-Score
3. 异常模式检测 (Anomaly Pattern Detection)
4. 历史案例分析 (Historical Case Study)
5. 当前风险排查 (Current Risk Scanning)

作者: Financial Analytics System
日期: 2024
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
REPORT_DIR = Path("/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class FinancialAnomalyDetector:
    """财务异常检测系统"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.results = {}

    def close(self):
        self.conn.close()

    def _execute_query(self, query: str) -> pd.DataFrame:
        """执行SQL查询并返回DataFrame"""
        try:
            return self.conn.execute(query).fetchdf()
        except Exception as e:
            print(f"查询错误: {e}")
            return pd.DataFrame()

    # ========== 1. 盈余质量分析 ==========

    def analyze_earnings_quality(self, latest_period: str = '20231231') -> dict:
        """
        盈余质量分析
        - 应计利润 vs 经营现金流
        - 非经常性损益占比
        - 收入确认异常检测
        """
        print("\n" + "="*60)
        print("1. 盈余质量分析 (Earnings Quality Analysis)")
        print("="*60)

        results = {}

        # 1.1 应计利润与经营现金流背离分析
        query_accrual = f"""
        WITH financial_data AS (
            SELECT
                i.ts_code,
                s.name,
                s.industry,
                i.end_date,
                i.n_income_attr_p as net_income,  -- 归母净利润
                c.n_cashflow_act as ocf,  -- 经营活动现金流净额
                i.revenue,
                i.total_revenue
            FROM income i
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code
                AND i.end_date = c.end_date
                AND i.report_type = c.report_type
            LEFT JOIN stock_basic s ON i.ts_code = s.ts_code
            WHERE i.report_type = '1'
                AND i.end_date = '{latest_period}'
                AND i.n_income_attr_p IS NOT NULL
                AND c.n_cashflow_act IS NOT NULL
                AND i.n_income_attr_p > 0  -- 盈利公司
        )
        SELECT
            ts_code,
            name,
            industry,
            net_income,
            ocf,
            revenue,
            -- 应计利润 = 净利润 - 经营现金流
            (net_income - ocf) as accruals,
            -- 应计比率 = 应计利润 / 净利润
            CASE WHEN net_income != 0 THEN (net_income - ocf) / net_income * 100 ELSE 0 END as accrual_ratio,
            -- 现金收入比 = 经营现金流 / 净利润
            CASE WHEN net_income != 0 THEN ocf / net_income * 100 ELSE 0 END as ocf_to_income_ratio
        FROM financial_data
        WHERE net_income > 10000000  -- 过滤净利润大于1000万的公司
        ORDER BY accrual_ratio DESC
        """

        df_accrual = self._execute_query(query_accrual)

        # 找出高应计比率的公司 (应计利润占净利润超过100%)
        high_accrual = df_accrual[df_accrual['accrual_ratio'] > 100].head(50)
        results['high_accrual_companies'] = high_accrual

        print(f"\n1.1 高应计比率公司 (应计利润/净利润 > 100%):")
        print(f"    发现 {len(df_accrual[df_accrual['accrual_ratio'] > 100])} 家公司")
        if len(high_accrual) > 0:
            print(f"\n    前10家高风险公司:")
            for _, row in high_accrual.head(10).iterrows():
                print(f"    {row['ts_code']} {row['name']}: 应计比率={row['accrual_ratio']:.1f}%")

        # 1.2 非经常性损益占比分析
        query_nonrecurring = f"""
        SELECT
            f.ts_code,
            s.name,
            s.industry,
            f.end_date,
            f.extra_item,  -- 非经常性损益
            f.profit_dedt,  -- 扣非净利润
            i.n_income_attr_p as net_income,
            -- 非经常性损益占比
            CASE WHEN i.n_income_attr_p != 0 AND i.n_income_attr_p IS NOT NULL
                 THEN (i.n_income_attr_p - f.profit_dedt) / ABS(i.n_income_attr_p) * 100
                 ELSE NULL END as nonrecurring_ratio
        FROM fina_indicator_vip f
        LEFT JOIN income i ON f.ts_code = i.ts_code AND f.end_date = i.end_date AND i.report_type = '1'
        LEFT JOIN stock_basic s ON f.ts_code = s.ts_code
        WHERE f.end_date = '{latest_period}'
            AND i.n_income_attr_p > 10000000  -- 净利润大于1000万
            AND f.profit_dedt IS NOT NULL
        ORDER BY nonrecurring_ratio DESC NULLS LAST
        """

        df_nonrecurring = self._execute_query(query_nonrecurring)

        # 非经常性损益占比超过50%的公司
        high_nonrecurring = df_nonrecurring[
            (df_nonrecurring['nonrecurring_ratio'].notna()) &
            (df_nonrecurring['nonrecurring_ratio'] > 50)
        ].head(50)
        results['high_nonrecurring_companies'] = high_nonrecurring

        print(f"\n1.2 非经常性损益占比高的公司 (> 50%):")
        print(f"    发现 {len(df_nonrecurring[(df_nonrecurring['nonrecurring_ratio'].notna()) & (df_nonrecurring['nonrecurring_ratio'] > 50)])} 家公司")

        # 1.3 收入确认异常 - 收入增长与现金流增长背离
        query_revenue_anomaly = f"""
        WITH current_data AS (
            SELECT
                i.ts_code,
                i.revenue as curr_revenue,
                c.c_fr_sale_sg as curr_cash_from_sales
            FROM income i
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code
                AND i.end_date = c.end_date
                AND i.report_type = c.report_type
            WHERE i.report_type = '1' AND i.end_date = '{latest_period}'
        ),
        prev_data AS (
            SELECT
                i.ts_code,
                i.revenue as prev_revenue,
                c.c_fr_sale_sg as prev_cash_from_sales
            FROM income i
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code
                AND i.end_date = c.end_date
                AND i.report_type = c.report_type
            WHERE i.report_type = '1' AND i.end_date = '{str(int(latest_period) - 10000)}'
        )
        SELECT
            curr.ts_code,
            s.name,
            s.industry,
            curr.curr_revenue,
            prev.prev_revenue,
            curr.curr_cash_from_sales,
            prev.prev_cash_from_sales,
            CASE WHEN prev.prev_revenue > 0 THEN (curr.curr_revenue - prev.prev_revenue) / prev.prev_revenue * 100 ELSE NULL END as revenue_growth,
            CASE WHEN prev.prev_cash_from_sales > 0 THEN (curr.curr_cash_from_sales - prev.prev_cash_from_sales) / prev.prev_cash_from_sales * 100 ELSE NULL END as cash_growth,
            CASE WHEN prev.prev_revenue > 0 AND prev.prev_cash_from_sales > 0
                 THEN ((curr.curr_revenue - prev.prev_revenue) / prev.prev_revenue - (curr.curr_cash_from_sales - prev.prev_cash_from_sales) / prev.prev_cash_from_sales) * 100
                 ELSE NULL END as divergence
        FROM current_data curr
        LEFT JOIN prev_data prev ON curr.ts_code = prev.ts_code
        LEFT JOIN stock_basic s ON curr.ts_code = s.ts_code
        WHERE curr.curr_revenue > 100000000  -- 收入大于1亿
            AND prev.prev_revenue > 100000000
        ORDER BY divergence DESC NULLS LAST
        """

        df_revenue_anomaly = self._execute_query(query_revenue_anomaly)

        # 收入增长远超现金流增长的公司
        revenue_cash_divergence = df_revenue_anomaly[
            (df_revenue_anomaly['divergence'].notna()) &
            (df_revenue_anomaly['divergence'] > 30)  # 背离超过30个百分点
        ].head(50)
        results['revenue_cash_divergence'] = revenue_cash_divergence

        print(f"\n1.3 收入与销售现金流背离的公司 (背离 > 30%):")
        print(f"    发现 {len(df_revenue_anomaly[(df_revenue_anomaly['divergence'].notna()) & (df_revenue_anomaly['divergence'] > 30)])} 家公司")

        return results

    # ========== 2. 财务造假预警模型 ==========

    def calculate_beneish_mscore(self, latest_period: str = '20231231') -> pd.DataFrame:
        """
        Beneish M-Score 模型 - 盈余操纵检测

        M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
                  + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

        M-Score > -1.78 表示可能存在盈余操纵

        指标说明:
        - DSRI (Days Sales in Receivables Index): 应收账款周转天数指数
        - GMI (Gross Margin Index): 毛利率指数
        - AQI (Asset Quality Index): 资产质量指数
        - SGI (Sales Growth Index): 销售增长指数
        - DEPI (Depreciation Index): 折旧指数
        - SGAI (SG&A Index): 销管费用指数
        - TATA (Total Accruals to Total Assets): 应计利润占总资产比
        - LVGI (Leverage Index): 杠杆指数
        """
        print("\n" + "="*60)
        print("2.1 Beneish M-Score 模型 (盈余操纵检测)")
        print("="*60)

        prev_period = str(int(latest_period) - 10000)

        query = f"""
        WITH curr AS (
            SELECT
                i.ts_code,
                i.revenue,
                i.total_revenue,
                i.oper_cost,
                i.sell_exp,
                i.admin_exp,
                i.n_income_attr_p as net_income,
                b.accounts_receiv,
                b.total_assets,
                b.total_cur_assets,
                b.fix_assets,
                b.total_liab,
                b.total_cur_liab,
                c.n_cashflow_act as ocf,
                c.depr_fa_coga_dpba as depreciation
            FROM income i
            LEFT JOIN balancesheet b ON i.ts_code = b.ts_code AND i.end_date = b.end_date AND i.report_type = b.report_type
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code AND i.end_date = c.end_date AND i.report_type = c.report_type
            WHERE i.report_type = '1' AND i.end_date = '{latest_period}'
        ),
        prev AS (
            SELECT
                i.ts_code,
                i.revenue as prev_revenue,
                i.oper_cost as prev_oper_cost,
                i.sell_exp as prev_sell_exp,
                i.admin_exp as prev_admin_exp,
                b.accounts_receiv as prev_ar,
                b.total_assets as prev_total_assets,
                b.total_cur_assets as prev_total_cur_assets,
                b.fix_assets as prev_fix_assets,
                b.total_liab as prev_total_liab,
                c.depr_fa_coga_dpba as prev_depreciation
            FROM income i
            LEFT JOIN balancesheet b ON i.ts_code = b.ts_code AND i.end_date = b.end_date AND i.report_type = b.report_type
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code AND i.end_date = c.end_date AND i.report_type = c.report_type
            WHERE i.report_type = '1' AND i.end_date = '{prev_period}'
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.revenue,
            c.net_income,
            c.total_assets,

            -- DSRI: 应收账款周转天数指数
            CASE WHEN p.prev_ar > 0 AND c.revenue > 0 AND p.prev_revenue > 0
                 THEN (c.accounts_receiv / c.revenue) / (p.prev_ar / p.prev_revenue)
                 ELSE 1 END as DSRI,

            -- GMI: 毛利率指数 (上期毛利率/本期毛利率)
            CASE WHEN c.revenue > 0 AND p.prev_revenue > 0 AND c.oper_cost IS NOT NULL AND p.prev_oper_cost IS NOT NULL
                 THEN ((p.prev_revenue - p.prev_oper_cost) / p.prev_revenue) /
                      NULLIF((c.revenue - c.oper_cost) / c.revenue, 0)
                 ELSE 1 END as GMI,

            -- AQI: 资产质量指数
            CASE WHEN c.total_assets > 0 AND p.prev_total_assets > 0
                 THEN (1 - (c.total_cur_assets + c.fix_assets) / c.total_assets) /
                      NULLIF(1 - (p.prev_total_cur_assets + p.prev_fix_assets) / p.prev_total_assets, 0)
                 ELSE 1 END as AQI,

            -- SGI: 销售增长指数
            CASE WHEN p.prev_revenue > 0 THEN c.revenue / p.prev_revenue ELSE 1 END as SGI,

            -- DEPI: 折旧指数
            CASE WHEN c.fix_assets > 0 AND p.prev_fix_assets > 0 AND c.depreciation IS NOT NULL AND p.prev_depreciation IS NOT NULL
                 THEN (p.prev_depreciation / (p.prev_depreciation + p.prev_fix_assets)) /
                      NULLIF((c.depreciation / (c.depreciation + c.fix_assets)), 0)
                 ELSE 1 END as DEPI,

            -- SGAI: 销管费用指数
            CASE WHEN c.revenue > 0 AND p.prev_revenue > 0
                 THEN ((c.sell_exp + c.admin_exp) / c.revenue) /
                      NULLIF((p.prev_sell_exp + p.prev_admin_exp) / p.prev_revenue, 0)
                 ELSE 1 END as SGAI,

            -- TATA: 应计利润占总资产比
            CASE WHEN c.total_assets > 0 AND c.net_income IS NOT NULL AND c.ocf IS NOT NULL
                 THEN (c.net_income - c.ocf) / c.total_assets
                 ELSE 0 END as TATA,

            -- LVGI: 杠杆指数
            CASE WHEN c.total_assets > 0 AND p.prev_total_assets > 0
                 THEN (c.total_liab / c.total_assets) /
                      NULLIF(p.prev_total_liab / p.prev_total_assets, 0)
                 ELSE 1 END as LVGI

        FROM curr c
        LEFT JOIN prev p ON c.ts_code = p.ts_code
        LEFT JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE c.revenue > 100000000  -- 过滤收入大于1亿的公司
            AND c.total_assets > 100000000
        """

        df = self._execute_query(query)

        if len(df) > 0:
            # 计算 M-Score
            # 处理异常值，限制在合理范围内
            for col in ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI']:
                df[col] = df[col].clip(-10, 10)
            df['TATA'] = df['TATA'].clip(-1, 1)

            df['M_Score'] = (
                -4.84
                + 0.92 * df['DSRI']
                + 0.528 * df['GMI']
                + 0.404 * df['AQI']
                + 0.892 * df['SGI']
                + 0.115 * df['DEPI']
                - 0.172 * df['SGAI']
                + 4.679 * df['TATA']
                - 0.327 * df['LVGI']
            )

            # 过滤有效结果
            df = df[df['M_Score'].notna()].copy()

            # 标记高风险公司 (M-Score > -1.78)
            df['manipulation_risk'] = df['M_Score'] > -1.78

            high_risk = df[df['manipulation_risk']].sort_values('M_Score', ascending=False)

            print(f"\n分析公司总数: {len(df)}")
            print(f"疑似盈余操纵公司 (M-Score > -1.78): {len(high_risk)} 家 ({len(high_risk)/len(df)*100:.1f}%)")

            if len(high_risk) > 0:
                print(f"\n前20家高风险公司:")
                for _, row in high_risk.head(20).iterrows():
                    print(f"    {row['ts_code']} {row['name']}: M-Score={row['M_Score']:.2f}")

            self.results['beneish_mscore'] = df

        return df

    def calculate_altman_zscore(self, latest_period: str = '20231231') -> pd.DataFrame:
        """
        Altman Z-Score 模型 - 破产预测

        原始模型 (制造业):
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        改进模型 (非制造业):
        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

        X1 = 营运资本 / 总资产
        X2 = 留存收益 / 总资产
        X3 = EBIT / 总资产
        X4 = 股东权益市值 / 总负债
        X5 = 销售收入 / 总资产

        Z-Score 判断标准:
        - Z > 2.99: 安全区域
        - 1.81 < Z < 2.99: 灰色区域
        - Z < 1.81: 危险区域
        """
        print("\n" + "="*60)
        print("2.2 Altman Z-Score 模型 (破产预测)")
        print("="*60)

        query = f"""
        SELECT
            b.ts_code,
            s.name,
            s.industry,
            b.total_assets,
            b.total_cur_assets,
            b.total_cur_liab,
            b.total_liab,
            b.total_hldr_eqy_exc_min_int as equity,
            b.undistr_porfit as retained_earnings,
            b.surplus_rese,
            i.revenue,
            i.operate_profit,
            i.n_income_attr_p as net_income,
            i.fin_exp as interest_expense,
            -- 营运资本
            (b.total_cur_assets - b.total_cur_liab) as working_capital,
            -- EBIT (息税前利润)
            (i.operate_profit + COALESCE(i.fin_exp, 0)) as ebit
        FROM balancesheet b
        LEFT JOIN income i ON b.ts_code = i.ts_code AND b.end_date = i.end_date AND b.report_type = i.report_type
        LEFT JOIN stock_basic s ON b.ts_code = s.ts_code
        WHERE b.report_type = '1' AND b.end_date = '{latest_period}'
            AND b.total_assets > 100000000
            AND b.total_liab > 0
        """

        df = self._execute_query(query)

        if len(df) > 0:
            # 计算各项指标
            # X1: 营运资本/总资产
            df['X1'] = df['working_capital'] / df['total_assets']

            # X2: 留存收益/总资产 (使用未分配利润+盈余公积)
            df['X2'] = (df['retained_earnings'].fillna(0) + df['surplus_rese'].fillna(0)) / df['total_assets']

            # X3: EBIT/总资产
            df['X3'] = df['ebit'].fillna(0) / df['total_assets']

            # X4: 股东权益/总负债 (简化版，使用账面价值)
            df['X4'] = df['equity'].fillna(0) / df['total_liab']

            # X5: 收入/总资产
            df['X5'] = df['revenue'].fillna(0) / df['total_assets']

            # 计算Z-Score (使用改进模型)
            df['Z_Score'] = (
                6.56 * df['X1'].clip(-5, 5)
                + 3.26 * df['X2'].clip(-5, 5)
                + 6.72 * df['X3'].clip(-2, 2)
                + 1.05 * df['X4'].clip(0, 10)
            )

            # 分类
            df['risk_zone'] = pd.cut(
                df['Z_Score'],
                bins=[-np.inf, 1.1, 2.6, np.inf],
                labels=['危险区域', '灰色区域', '安全区域']
            )

            # 过滤有效结果
            df = df[df['Z_Score'].notna()].copy()

            danger_zone = df[df['risk_zone'] == '危险区域'].sort_values('Z_Score')
            gray_zone = df[df['risk_zone'] == '灰色区域'].sort_values('Z_Score')

            print(f"\n分析公司总数: {len(df)}")
            print(f"危险区域 (Z < 1.1): {len(danger_zone)} 家 ({len(danger_zone)/len(df)*100:.1f}%)")
            print(f"灰色区域 (1.1 < Z < 2.6): {len(gray_zone)} 家 ({len(gray_zone)/len(df)*100:.1f}%)")
            print(f"安全区域 (Z > 2.6): {len(df[df['risk_zone'] == '安全区域'])} 家")

            if len(danger_zone) > 0:
                print(f"\n危险区域前20家公司:")
                for _, row in danger_zone.head(20).iterrows():
                    print(f"    {row['ts_code']} {row['name']}: Z-Score={row['Z_Score']:.2f}")

            self.results['altman_zscore'] = df

        return df

    def calculate_piotroski_fscore(self, latest_period: str = '20231231') -> pd.DataFrame:
        """
        Piotroski F-Score 模型 - 财务健康评估

        9项指标，每项1分，总分0-9分:

        盈利能力 (4分):
        1. ROA > 0
        2. 经营现金流 > 0
        3. ROA 增长
        4. 应计利润 < 0 (即现金流 > 净利润)

        杠杆与流动性 (3分):
        5. 长期负债比下降
        6. 流动比率上升
        7. 无新股发行

        运营效率 (2分):
        8. 毛利率上升
        9. 资产周转率上升

        F-Score >= 8: 财务健康
        F-Score <= 2: 财务困境
        """
        print("\n" + "="*60)
        print("2.3 Piotroski F-Score 模型 (财务健康评估)")
        print("="*60)

        prev_period = str(int(latest_period) - 10000)

        query = f"""
        WITH curr AS (
            SELECT
                i.ts_code,
                i.revenue,
                i.oper_cost,
                i.n_income_attr_p as net_income,
                b.total_assets,
                b.total_ncl as long_term_debt,
                b.total_cur_assets,
                b.total_cur_liab,
                b.total_share,
                c.n_cashflow_act as ocf
            FROM income i
            LEFT JOIN balancesheet b ON i.ts_code = b.ts_code AND i.end_date = b.end_date AND i.report_type = b.report_type
            LEFT JOIN cashflow c ON i.ts_code = c.ts_code AND i.end_date = c.end_date AND i.report_type = c.report_type
            WHERE i.report_type = '1' AND i.end_date = '{latest_period}'
        ),
        prev AS (
            SELECT
                i.ts_code,
                i.revenue as prev_revenue,
                i.oper_cost as prev_oper_cost,
                i.n_income_attr_p as prev_net_income,
                b.total_assets as prev_total_assets,
                b.total_ncl as prev_long_term_debt,
                b.total_cur_assets as prev_cur_assets,
                b.total_cur_liab as prev_cur_liab,
                b.total_share as prev_total_share
            FROM income i
            LEFT JOIN balancesheet b ON i.ts_code = b.ts_code AND i.end_date = b.end_date AND i.report_type = b.report_type
            WHERE i.report_type = '1' AND i.end_date = '{prev_period}'
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.revenue,
            c.net_income,
            c.total_assets,
            c.ocf,

            -- ROA
            CASE WHEN c.total_assets > 0 THEN c.net_income / c.total_assets ELSE 0 END as ROA,
            CASE WHEN p.prev_total_assets > 0 THEN p.prev_net_income / p.prev_total_assets ELSE 0 END as prev_ROA,

            -- 毛利率
            CASE WHEN c.revenue > 0 THEN (c.revenue - c.oper_cost) / c.revenue ELSE 0 END as gross_margin,
            CASE WHEN p.prev_revenue > 0 THEN (p.prev_revenue - p.prev_oper_cost) / p.prev_revenue ELSE 0 END as prev_gross_margin,

            -- 流动比率
            CASE WHEN c.total_cur_liab > 0 THEN c.total_cur_assets / c.total_cur_liab ELSE 0 END as current_ratio,
            CASE WHEN p.prev_cur_liab > 0 THEN p.prev_cur_assets / p.prev_cur_liab ELSE 0 END as prev_current_ratio,

            -- 长期负债比率
            CASE WHEN c.total_assets > 0 THEN COALESCE(c.long_term_debt, 0) / c.total_assets ELSE 0 END as debt_ratio,
            CASE WHEN p.prev_total_assets > 0 THEN COALESCE(p.prev_long_term_debt, 0) / p.prev_total_assets ELSE 0 END as prev_debt_ratio,

            -- 资产周转率
            CASE WHEN c.total_assets > 0 THEN c.revenue / c.total_assets ELSE 0 END as asset_turnover,
            CASE WHEN p.prev_total_assets > 0 THEN p.prev_revenue / p.prev_total_assets ELSE 0 END as prev_asset_turnover,

            -- 股本变化
            c.total_share,
            p.prev_total_share

        FROM curr c
        LEFT JOIN prev p ON c.ts_code = p.ts_code
        LEFT JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE c.total_assets > 100000000
        """

        df = self._execute_query(query)

        if len(df) > 0:
            # 计算9项得分
            # 盈利能力
            df['F1_ROA_positive'] = (df['ROA'] > 0).astype(int)
            df['F2_OCF_positive'] = (df['ocf'] > 0).astype(int)
            df['F3_ROA_increase'] = (df['ROA'] > df['prev_ROA']).astype(int)
            df['F4_accrual_negative'] = (df['ocf'] > df['net_income']).astype(int)

            # 杠杆与流动性
            df['F5_debt_decrease'] = (df['debt_ratio'] < df['prev_debt_ratio']).astype(int)
            df['F6_current_increase'] = (df['current_ratio'] > df['prev_current_ratio']).astype(int)
            df['F7_no_dilution'] = ((df['total_share'].fillna(0) <= df['prev_total_share'].fillna(0)) |
                                    (df['prev_total_share'].isna())).astype(int)

            # 运营效率
            df['F8_margin_increase'] = (df['gross_margin'] > df['prev_gross_margin']).astype(int)
            df['F9_turnover_increase'] = (df['asset_turnover'] > df['prev_asset_turnover']).astype(int)

            # 计算总分
            df['F_Score'] = (
                df['F1_ROA_positive'] + df['F2_OCF_positive'] +
                df['F3_ROA_increase'] + df['F4_accrual_negative'] +
                df['F5_debt_decrease'] + df['F6_current_increase'] +
                df['F7_no_dilution'] + df['F8_margin_increase'] +
                df['F9_turnover_increase']
            )

            # 分类
            low_score = df[df['F_Score'] <= 2].sort_values('F_Score')
            high_score = df[df['F_Score'] >= 8].sort_values('F_Score', ascending=False)

            print(f"\n分析公司总数: {len(df)}")
            print(f"财务困境 (F-Score <= 2): {len(low_score)} 家 ({len(low_score)/len(df)*100:.1f}%)")
            print(f"财务健康 (F-Score >= 8): {len(high_score)} 家 ({len(high_score)/len(df)*100:.1f}%)")

            # 分数分布
            score_dist = df['F_Score'].value_counts().sort_index()
            print(f"\nF-Score 分布:")
            for score, count in score_dist.items():
                print(f"    {int(score)}分: {count} 家 ({count/len(df)*100:.1f}%)")

            if len(low_score) > 0:
                print(f"\n财务困境公司 (前20家):")
                for _, row in low_score.head(20).iterrows():
                    print(f"    {row['ts_code']} {row['name']}: F-Score={int(row['F_Score'])}")

            self.results['piotroski_fscore'] = df

        return df

    # ========== 3. 异常模式检测 ==========

    def detect_anomaly_patterns(self, latest_period: str = '20231231') -> dict:
        """
        异常模式检测
        - 存货异常增长
        - 应收账款异常
        - 毛利率异常波动
        - 费用率异常
        """
        print("\n" + "="*60)
        print("3. 异常模式检测 (Anomaly Pattern Detection)")
        print("="*60)

        results = {}
        prev_period = str(int(latest_period) - 10000)

        # 3.1 存货异常增长
        query_inventory = f"""
        WITH curr AS (
            SELECT
                b.ts_code,
                b.inventories,
                b.total_assets,
                i.revenue,
                i.oper_cost
            FROM balancesheet b
            LEFT JOIN income i ON b.ts_code = i.ts_code AND b.end_date = i.end_date AND b.report_type = i.report_type
            WHERE b.report_type = '1' AND b.end_date = '{latest_period}'
        ),
        prev AS (
            SELECT
                b.ts_code,
                b.inventories as prev_inv,
                i.revenue as prev_revenue
            FROM balancesheet b
            LEFT JOIN income i ON b.ts_code = i.ts_code AND b.end_date = i.end_date AND b.report_type = i.report_type
            WHERE b.report_type = '1' AND b.end_date = '{prev_period}'
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.inventories,
            p.prev_inv,
            c.revenue,
            p.prev_revenue,
            -- 存货增长率
            CASE WHEN p.prev_inv > 0 THEN (c.inventories - p.prev_inv) / p.prev_inv * 100 ELSE NULL END as inv_growth,
            -- 收入增长率
            CASE WHEN p.prev_revenue > 0 THEN (c.revenue - p.prev_revenue) / p.prev_revenue * 100 ELSE NULL END as rev_growth,
            -- 存货/收入比
            CASE WHEN c.revenue > 0 THEN c.inventories / c.revenue * 100 ELSE NULL END as inv_to_revenue,
            -- 存货增长与收入增长的背离
            CASE WHEN p.prev_inv > 0 AND p.prev_revenue > 0
                 THEN ((c.inventories - p.prev_inv) / p.prev_inv - (c.revenue - p.prev_revenue) / p.prev_revenue) * 100
                 ELSE NULL END as inv_rev_divergence
        FROM curr c
        LEFT JOIN prev p ON c.ts_code = p.ts_code
        LEFT JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE c.inventories > 10000000  -- 存货大于1000万
            AND c.revenue > 100000000
        ORDER BY inv_rev_divergence DESC NULLS LAST
        """

        df_inventory = self._execute_query(query_inventory)

        # 存货增长远超收入增长
        inv_anomaly = df_inventory[
            (df_inventory['inv_rev_divergence'].notna()) &
            (df_inventory['inv_rev_divergence'] > 30)
        ].head(50)
        results['inventory_anomaly'] = inv_anomaly

        print(f"\n3.1 存货异常增长 (存货增速 - 收入增速 > 30%):")
        print(f"    发现 {len(df_inventory[(df_inventory['inv_rev_divergence'].notna()) & (df_inventory['inv_rev_divergence'] > 30)])} 家公司")

        # 3.2 应收账款异常
        query_ar = f"""
        WITH curr AS (
            SELECT
                b.ts_code,
                b.accounts_receiv,
                b.notes_receiv,
                (COALESCE(b.accounts_receiv, 0) + COALESCE(b.notes_receiv, 0)) as total_ar,
                i.revenue
            FROM balancesheet b
            LEFT JOIN income i ON b.ts_code = i.ts_code AND b.end_date = i.end_date AND b.report_type = i.report_type
            WHERE b.report_type = '1' AND b.end_date = '{latest_period}'
        ),
        prev AS (
            SELECT
                b.ts_code,
                (COALESCE(b.accounts_receiv, 0) + COALESCE(b.notes_receiv, 0)) as prev_ar,
                i.revenue as prev_revenue
            FROM balancesheet b
            LEFT JOIN income i ON b.ts_code = i.ts_code AND b.end_date = i.end_date AND b.report_type = i.report_type
            WHERE b.report_type = '1' AND b.end_date = '{prev_period}'
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.total_ar,
            p.prev_ar,
            c.revenue,
            p.prev_revenue,
            -- 应收账款增长率
            CASE WHEN p.prev_ar > 0 THEN (c.total_ar - p.prev_ar) / p.prev_ar * 100 ELSE NULL END as ar_growth,
            -- 收入增长率
            CASE WHEN p.prev_revenue > 0 THEN (c.revenue - p.prev_revenue) / p.prev_revenue * 100 ELSE NULL END as rev_growth,
            -- 应收账款周转天数
            CASE WHEN c.revenue > 0 THEN c.total_ar / c.revenue * 365 ELSE NULL END as ar_days,
            -- 应收账款增长与收入增长的背离
            CASE WHEN p.prev_ar > 0 AND p.prev_revenue > 0
                 THEN ((c.total_ar - p.prev_ar) / p.prev_ar - (c.revenue - p.prev_revenue) / p.prev_revenue) * 100
                 ELSE NULL END as ar_rev_divergence
        FROM curr c
        LEFT JOIN prev p ON c.ts_code = p.ts_code
        LEFT JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE c.total_ar > 10000000
            AND c.revenue > 100000000
        ORDER BY ar_rev_divergence DESC NULLS LAST
        """

        df_ar = self._execute_query(query_ar)

        ar_anomaly = df_ar[
            (df_ar['ar_rev_divergence'].notna()) &
            (df_ar['ar_rev_divergence'] > 30)
        ].head(50)
        results['ar_anomaly'] = ar_anomaly

        print(f"\n3.2 应收账款异常 (应收增速 - 收入增速 > 30%):")
        print(f"    发现 {len(df_ar[(df_ar['ar_rev_divergence'].notna()) & (df_ar['ar_rev_divergence'] > 30)])} 家公司")

        # 3.3 毛利率异常波动
        query_margin = f"""
        WITH margins AS (
            SELECT
                i.ts_code,
                i.end_date,
                i.revenue,
                i.oper_cost,
                CASE WHEN i.revenue > 0 THEN (i.revenue - i.oper_cost) / i.revenue * 100 ELSE NULL END as gross_margin
            FROM income i
            WHERE i.report_type = '1'
                AND i.end_date >= '{str(int(latest_period) - 30000)}'  -- 最近3年
                AND i.revenue > 100000000
        ),
        stats AS (
            SELECT
                ts_code,
                AVG(gross_margin) as avg_margin,
                STDDEV(gross_margin) as std_margin,
                COUNT(*) as periods
            FROM margins
            WHERE gross_margin IS NOT NULL
            GROUP BY ts_code
            HAVING COUNT(*) >= 3
        )
        SELECT
            m.ts_code,
            s.name,
            s.industry,
            m.gross_margin as current_margin,
            st.avg_margin,
            st.std_margin,
            st.periods,
            -- Z-score
            CASE WHEN st.std_margin > 0 THEN (m.gross_margin - st.avg_margin) / st.std_margin ELSE 0 END as margin_zscore
        FROM margins m
        LEFT JOIN stats st ON m.ts_code = st.ts_code
        LEFT JOIN stock_basic s ON m.ts_code = s.ts_code
        WHERE m.end_date = '{latest_period}'
            AND st.std_margin IS NOT NULL
        ORDER BY ABS(margin_zscore) DESC
        """

        df_margin = self._execute_query(query_margin)

        margin_anomaly = df_margin[
            (df_margin['margin_zscore'].notna()) &
            (df_margin['margin_zscore'].abs() > 2)
        ].head(50)
        results['margin_anomaly'] = margin_anomaly

        print(f"\n3.3 毛利率异常波动 (|Z-score| > 2):")
        print(f"    发现 {len(df_margin[(df_margin['margin_zscore'].notna()) & (df_margin['margin_zscore'].abs() > 2)])} 家公司")

        # 3.4 费用率异常
        query_expense = f"""
        WITH curr AS (
            SELECT
                i.ts_code,
                i.revenue,
                i.sell_exp,
                i.admin_exp,
                i.fin_exp,
                CASE WHEN i.revenue > 0 THEN (COALESCE(i.sell_exp, 0) + COALESCE(i.admin_exp, 0)) / i.revenue * 100 ELSE NULL END as expense_ratio
            FROM income i
            WHERE i.report_type = '1' AND i.end_date = '{latest_period}'
        ),
        prev AS (
            SELECT
                i.ts_code,
                CASE WHEN i.revenue > 0 THEN (COALESCE(i.sell_exp, 0) + COALESCE(i.admin_exp, 0)) / i.revenue * 100 ELSE NULL END as prev_expense_ratio
            FROM income i
            WHERE i.report_type = '1' AND i.end_date = '{prev_period}'
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.revenue,
            c.sell_exp,
            c.admin_exp,
            c.expense_ratio,
            p.prev_expense_ratio,
            -- 费用率变化
            (c.expense_ratio - p.prev_expense_ratio) as expense_ratio_change
        FROM curr c
        LEFT JOIN prev p ON c.ts_code = p.ts_code
        LEFT JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE c.revenue > 100000000
            AND c.expense_ratio IS NOT NULL
            AND p.prev_expense_ratio IS NOT NULL
        ORDER BY ABS(expense_ratio_change) DESC
        """

        df_expense = self._execute_query(query_expense)

        expense_anomaly = df_expense[
            (df_expense['expense_ratio_change'].notna()) &
            (df_expense['expense_ratio_change'].abs() > 10)
        ].head(50)
        results['expense_anomaly'] = expense_anomaly

        print(f"\n3.4 费用率异常波动 (|变化| > 10%):")
        print(f"    发现 {len(df_expense[(df_expense['expense_ratio_change'].notna()) & (df_expense['expense_ratio_change'].abs() > 10)])} 家公司")

        self.results['anomaly_patterns'] = results

        return results

    # ========== 4. 历史案例分析 ==========

    def analyze_historical_cases(self, latest_period: str = '20231231') -> dict:
        """
        历史案例分析
        - 识别ST和退市公司
        - 分析其财务特征
        """
        print("\n" + "="*60)
        print("4. 历史案例分析 (Historical Case Study)")
        print("="*60)

        results = {}

        # 4.1 当前ST公司统计
        query_st = """
        SELECT
            ts_code,
            name,
            industry,
            list_date,
            market,
            CASE
                WHEN name LIKE '*ST%' THEN '退市风险警示 (*ST)'
                WHEN name LIKE 'ST%' THEN '特别处理 (ST)'
                ELSE '正常'
            END as st_type
        FROM stock_basic
        WHERE name LIKE '%ST%' AND list_status = 'L'
        ORDER BY st_type, ts_code
        """

        df_st = self._execute_query(query_st)
        results['st_companies'] = df_st

        st_count = len(df_st[df_st['st_type'].str.contains('ST')])
        star_st_count = len(df_st[df_st['st_type'].str.contains('\\*ST')])

        print(f"\n4.1 当前ST公司统计:")
        print(f"    *ST (退市风险): {star_st_count} 家")
        print(f"    ST (特别处理): {st_count - star_st_count} 家")
        print(f"    合计: {st_count} 家")

        # 4.2 退市公司统计
        query_delisted = """
        SELECT
            ts_code,
            name,
            industry,
            list_date,
            market
        FROM stock_basic
        WHERE list_status = 'D'
        ORDER BY ts_code
        """

        df_delisted = self._execute_query(query_delisted)
        results['delisted_companies'] = df_delisted

        print(f"\n4.2 已退市公司: {len(df_delisted)} 家")

        # 4.3 ST公司的财务特征分析
        query_st_financials = f"""
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            f.roe,
            f.roa,
            f.current_ratio,
            f.quick_ratio,
            f.gross_margin,
            f.netprofit_margin,
            f.ar_turn,
            f.assets_turn,
            i.n_income_attr_p as net_income,
            c.n_cashflow_act as ocf
        FROM stock_basic s
        LEFT JOIN fina_indicator_vip f ON s.ts_code = f.ts_code AND f.end_date = '{latest_period}'
        LEFT JOIN income i ON s.ts_code = i.ts_code AND i.end_date = '{latest_period}' AND i.report_type = '1'
        LEFT JOIN cashflow c ON s.ts_code = c.ts_code AND c.end_date = '{latest_period}' AND c.report_type = '1'
        WHERE s.name LIKE '%ST%' AND s.list_status = 'L'
        """

        df_st_fin = self._execute_query(query_st_financials)
        results['st_financials'] = df_st_fin

        if len(df_st_fin) > 0:
            print(f"\n4.3 ST公司财务特征 (基于 {latest_period} 数据):")
            print(f"    ROE中位数: {df_st_fin['roe'].median():.2f}%")
            print(f"    ROA中位数: {df_st_fin['roa'].median():.2f}%")
            print(f"    流动比率中位数: {df_st_fin['current_ratio'].median():.2f}")
            print(f"    亏损公司占比: {(df_st_fin['net_income'] < 0).mean()*100:.1f}%")
            print(f"    经营现金流为负占比: {(df_st_fin['ocf'] < 0).mean()*100:.1f}%")

        self.results['historical_cases'] = results

        return results

    # ========== 5. 综合风险评分 ==========

    def calculate_composite_risk_score(self, latest_period: str = '20231231') -> pd.DataFrame:
        """
        计算综合风险评分
        整合所有模型的结果，给出最终风险评级
        """
        print("\n" + "="*60)
        print("5. 综合风险评分 (Composite Risk Score)")
        print("="*60)

        # 获取各模型结果
        if 'beneish_mscore' not in self.results:
            self.calculate_beneish_mscore(latest_period)
        if 'altman_zscore' not in self.results:
            self.calculate_altman_zscore(latest_period)
        if 'piotroski_fscore' not in self.results:
            self.calculate_piotroski_fscore(latest_period)

        # 合并结果
        df_m = self.results.get('beneish_mscore', pd.DataFrame())
        df_z = self.results.get('altman_zscore', pd.DataFrame())
        df_f = self.results.get('piotroski_fscore', pd.DataFrame())

        if len(df_m) == 0 or len(df_z) == 0 or len(df_f) == 0:
            print("警告: 部分模型数据缺失")
            return pd.DataFrame()

        # 选择关键列
        df_m_key = df_m[['ts_code', 'name', 'industry', 'M_Score', 'manipulation_risk']].copy()
        df_z_key = df_z[['ts_code', 'Z_Score', 'risk_zone']].copy()
        df_f_key = df_f[['ts_code', 'F_Score']].copy()

        # 合并
        df = df_m_key.merge(df_z_key, on='ts_code', how='inner')
        df = df.merge(df_f_key, on='ts_code', how='inner')

        # 计算综合风险分数 (0-100, 越高风险越大)
        # M-Score 风险分 (> -1.78 为高风险)
        df['m_risk'] = np.where(df['M_Score'] > -1.78, 30,
                                np.where(df['M_Score'] > -2.22, 15, 0))

        # Z-Score 风险分 (< 1.1 为高风险)
        df['z_risk'] = np.where(df['Z_Score'] < 1.1, 30,
                                np.where(df['Z_Score'] < 2.6, 15, 0))

        # F-Score 风险分 (< 3 为高风险)
        df['f_risk'] = np.where(df['F_Score'] <= 2, 40,
                                np.where(df['F_Score'] <= 4, 20,
                                np.where(df['F_Score'] <= 5, 10, 0)))

        # 综合风险分
        df['composite_risk_score'] = df['m_risk'] + df['z_risk'] + df['f_risk']

        # 风险等级
        df['risk_level'] = pd.cut(
            df['composite_risk_score'],
            bins=[-1, 20, 40, 60, 100],
            labels=['低风险', '中等风险', '高风险', '极高风险']
        )

        # 排序
        df = df.sort_values('composite_risk_score', ascending=False)

        # 统计
        risk_dist = df['risk_level'].value_counts()
        print(f"\n分析公司总数: {len(df)}")
        print(f"\n风险等级分布:")
        for level in ['极高风险', '高风险', '中等风险', '低风险']:
            if level in risk_dist.index:
                count = risk_dist[level]
                print(f"    {level}: {count} 家 ({count/len(df)*100:.1f}%)")

        # 输出极高风险公司
        extreme_risk = df[df['risk_level'] == '极高风险']
        if len(extreme_risk) > 0:
            print(f"\n极高风险公司 (前30家):")
            for _, row in extreme_risk.head(30).iterrows():
                print(f"    {row['ts_code']} {row['name']}: 综合得分={row['composite_risk_score']:.0f}")
                print(f"        M-Score={row['M_Score']:.2f}, Z-Score={row['Z_Score']:.2f}, F-Score={int(row['F_Score'])}")

        self.results['composite_risk'] = df

        return df

    # ========== 6. 生成报告 ==========

    def generate_report(self, latest_period: str = '20231231') -> str:
        """生成完整的分析报告"""

        # 确保所有分析已完成
        earnings_quality = self.analyze_earnings_quality(latest_period)
        self.calculate_beneish_mscore(latest_period)
        self.calculate_altman_zscore(latest_period)
        self.calculate_piotroski_fscore(latest_period)
        self.detect_anomaly_patterns(latest_period)
        self.analyze_historical_cases(latest_period)
        composite = self.calculate_composite_risk_score(latest_period)

        # 生成Markdown报告
        report = []
        report.append("# 财务异常检测系统分析报告")
        report.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**分析期间**: {latest_period[:4]}年{latest_period[4:6]}月{latest_period[6:]}日")
        report.append(f"\n**数据来源**: Tushare DuckDB 数据库")

        # 目录
        report.append("\n---\n")
        report.append("## 目录\n")
        report.append("1. [执行摘要](#执行摘要)")
        report.append("2. [方法论](#方法论)")
        report.append("3. [盈余质量分析](#盈余质量分析)")
        report.append("4. [财务造假预警模型](#财务造假预警模型)")
        report.append("5. [异常模式检测](#异常模式检测)")
        report.append("6. [历史案例分析](#历史案例分析)")
        report.append("7. [高风险股票名单](#高风险股票名单)")
        report.append("8. [风险预警建议](#风险预警建议)")

        # 执行摘要
        report.append("\n---\n")
        report.append("## 执行摘要\n")

        if len(composite) > 0:
            extreme_risk = composite[composite['risk_level'] == '极高风险']
            high_risk = composite[composite['risk_level'] == '高风险']

            report.append(f"本次分析覆盖 **{len(composite)}** 家上市公司，基于最新财报数据 ({latest_period})。\n")
            report.append("\n### 关键发现\n")
            report.append(f"- **极高风险公司**: {len(extreme_risk)} 家 ({len(extreme_risk)/len(composite)*100:.1f}%)")
            report.append(f"- **高风险公司**: {len(high_risk)} 家 ({len(high_risk)/len(composite)*100:.1f}%)")

            # M-Score
            if 'beneish_mscore' in self.results:
                mscore_risk = self.results['beneish_mscore'][self.results['beneish_mscore']['manipulation_risk']]
                report.append(f"- **疑似盈余操纵 (M-Score > -1.78)**: {len(mscore_risk)} 家")

            # Z-Score
            if 'altman_zscore' in self.results:
                zscore_danger = self.results['altman_zscore'][self.results['altman_zscore']['risk_zone'] == '危险区域']
                report.append(f"- **破产风险较高 (Z-Score < 1.1)**: {len(zscore_danger)} 家")

            # F-Score
            if 'piotroski_fscore' in self.results:
                fscore_low = self.results['piotroski_fscore'][self.results['piotroski_fscore']['F_Score'] <= 2]
                report.append(f"- **财务困境 (F-Score <= 2)**: {len(fscore_low)} 家")

        # 方法论
        report.append("\n---\n")
        report.append("## 方法论\n")

        report.append("### 1. Beneish M-Score 模型\n")
        report.append("用于检测盈余操纵的经典模型，由印第安纳大学 Messod Beneish 教授提出。\n")
        report.append("\n**计算公式**:\n")
        report.append("```")
        report.append("M-Score = -4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI")
        report.append("         + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI")
        report.append("```\n")
        report.append("**指标说明**:\n")
        report.append("| 指标 | 名称 | 计算方法 |")
        report.append("|------|------|----------|")
        report.append("| DSRI | 应收账款周转天数指数 | (本期应收/本期收入) / (上期应收/上期收入) |")
        report.append("| GMI | 毛利率指数 | 上期毛利率 / 本期毛利率 |")
        report.append("| AQI | 资产质量指数 | 本期非流动资产占比 / 上期非流动资产占比 |")
        report.append("| SGI | 销售增长指数 | 本期收入 / 上期收入 |")
        report.append("| DEPI | 折旧指数 | 上期折旧率 / 本期折旧率 |")
        report.append("| SGAI | 销管费用指数 | (本期销管费/收入) / (上期销管费/收入) |")
        report.append("| TATA | 应计利润/总资产 | (净利润 - 经营现金流) / 总资产 |")
        report.append("| LVGI | 杠杆指数 | 本期负债率 / 上期负债率 |\n")
        report.append("**判断标准**: M-Score > -1.78 表示可能存在盈余操纵\n")

        report.append("### 2. Altman Z-Score 模型\n")
        report.append("用于预测企业破产风险的经典模型，由纽约大学 Edward Altman 教授提出。\n")
        report.append("\n**计算公式** (非制造业改进版):\n")
        report.append("```")
        report.append("Z-Score = 6.56×X1 + 3.26×X2 + 6.72×X3 + 1.05×X4")
        report.append("```\n")
        report.append("**指标说明**:\n")
        report.append("| 指标 | 计算方法 |")
        report.append("|------|----------|")
        report.append("| X1 | 营运资本 / 总资产 |")
        report.append("| X2 | 留存收益 / 总资产 |")
        report.append("| X3 | EBIT / 总资产 |")
        report.append("| X4 | 股东权益 / 总负债 |\n")
        report.append("**判断标准**:\n")
        report.append("- Z > 2.6: 安全区域")
        report.append("- 1.1 < Z < 2.6: 灰色区域")
        report.append("- Z < 1.1: 危险区域\n")

        report.append("### 3. Piotroski F-Score 模型\n")
        report.append("用于评估企业财务健康状况的综合评分模型，由斯坦福大学 Joseph Piotroski 教授提出。\n")
        report.append("\n**评分项目** (满分9分):\n")
        report.append("\n**盈利能力 (4分)**:\n")
        report.append("1. ROA > 0 (+1分)")
        report.append("2. 经营现金流 > 0 (+1分)")
        report.append("3. ROA 同比增长 (+1分)")
        report.append("4. 现金流 > 净利润 (+1分)\n")
        report.append("**杠杆与流动性 (3分)**:\n")
        report.append("5. 长期负债比下降 (+1分)")
        report.append("6. 流动比率上升 (+1分)")
        report.append("7. 无新股稀释 (+1分)\n")
        report.append("**运营效率 (2分)**:\n")
        report.append("8. 毛利率上升 (+1分)")
        report.append("9. 资产周转率上升 (+1分)\n")
        report.append("**判断标准**:\n")
        report.append("- F-Score >= 8: 财务健康")
        report.append("- F-Score <= 2: 财务困境\n")

        report.append("### 4. 自定义红旗指标\n")
        report.append("基于财务造假案例研究总结的额外预警信号:\n")
        report.append("- 应计利润占净利润比例 > 100%")
        report.append("- 非经常性损益占净利润 > 50%")
        report.append("- 收入增长与销售现金流背离 > 30%")
        report.append("- 存货增长远超收入增长 > 30%")
        report.append("- 应收账款增长远超收入增长 > 30%")
        report.append("- 毛利率异常波动 (Z-score > 2)")
        report.append("- 费用率异常变化 > 10%\n")

        # 盈余质量分析
        report.append("\n---\n")
        report.append("## 盈余质量分析\n")

        if 'high_accrual_companies' in earnings_quality:
            df = earnings_quality['high_accrual_companies']
            report.append("### 高应计比率公司\n")
            report.append("应计利润占净利润比例超过100%的公司，盈余质量存疑。\n")
            if len(df) > 0:
                report.append(f"\n共发现 **{len(df)}** 家公司\n")
                report.append("\n| 代码 | 名称 | 行业 | 应计比率(%) | 现金收入比(%) |")
                report.append("|------|------|------|-------------|---------------|")
                for _, row in df.head(20).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['accrual_ratio']:.1f} | {row['ocf_to_income_ratio']:.1f} |")

        if 'high_nonrecurring_companies' in earnings_quality:
            df = earnings_quality['high_nonrecurring_companies']
            report.append("\n### 非经常性损益占比高的公司\n")
            report.append("非经常性损益占净利润超过50%，利润可持续性存疑。\n")
            if len(df) > 0:
                report.append(f"\n共发现 **{len(df)}** 家公司\n")
                report.append("\n| 代码 | 名称 | 行业 | 非经常性损益占比(%) |")
                report.append("|------|------|------|---------------------|")
                for _, row in df.head(20).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['nonrecurring_ratio']:.1f} |")

        # 财务造假预警模型
        report.append("\n---\n")
        report.append("## 财务造假预警模型\n")

        # M-Score结果
        if 'beneish_mscore' in self.results:
            df = self.results['beneish_mscore']
            high_risk = df[df['manipulation_risk']].sort_values('M_Score', ascending=False)

            report.append("### Beneish M-Score 模型结果\n")
            report.append(f"- 分析公司数: {len(df)}")
            report.append(f"- 疑似盈余操纵: {len(high_risk)} 家 ({len(high_risk)/len(df)*100:.1f}%)\n")

            if len(high_risk) > 0:
                report.append("\n**高风险公司 (M-Score > -1.78)**:\n")
                report.append("\n| 代码 | 名称 | 行业 | M-Score | DSRI | GMI | TATA |")
                report.append("|------|------|------|---------|------|-----|------|")
                for _, row in high_risk.head(30).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['M_Score']:.2f} | {row['DSRI']:.2f} | {row['GMI']:.2f} | {row['TATA']:.3f} |")

        # Z-Score结果
        if 'altman_zscore' in self.results:
            df = self.results['altman_zscore']
            danger = df[df['risk_zone'] == '危险区域'].sort_values('Z_Score')

            report.append("\n### Altman Z-Score 模型结果\n")
            report.append(f"- 分析公司数: {len(df)}")
            report.append(f"- 危险区域: {len(danger)} 家")
            report.append(f"- 灰色区域: {len(df[df['risk_zone'] == '灰色区域'])} 家")
            report.append(f"- 安全区域: {len(df[df['risk_zone'] == '安全区域'])} 家\n")

            if len(danger) > 0:
                report.append("\n**危险区域公司 (Z-Score < 1.1)**:\n")
                report.append("\n| 代码 | 名称 | 行业 | Z-Score | X1 | X2 | X3 | X4 |")
                report.append("|------|------|------|---------|-----|-----|-----|-----|")
                for _, row in danger.head(30).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['Z_Score']:.2f} | {row['X1']:.2f} | {row['X2']:.2f} | {row['X3']:.2f} | {row['X4']:.2f} |")

        # F-Score结果
        if 'piotroski_fscore' in self.results:
            df = self.results['piotroski_fscore']
            low_score = df[df['F_Score'] <= 2].sort_values('F_Score')

            report.append("\n### Piotroski F-Score 模型结果\n")
            report.append(f"- 分析公司数: {len(df)}")
            report.append(f"- 财务困境 (<=2分): {len(low_score)} 家")
            report.append(f"- 财务健康 (>=8分): {len(df[df['F_Score'] >= 8])} 家\n")

            # 分数分布
            report.append("\n**评分分布**:\n")
            report.append("| 分数 | 公司数 | 占比 |")
            report.append("|------|--------|------|")
            for score in range(10):
                count = len(df[df['F_Score'] == score])
                if count > 0:
                    report.append(f"| {score}分 | {count} | {count/len(df)*100:.1f}% |")

            if len(low_score) > 0:
                report.append("\n**财务困境公司 (F-Score <= 2)**:\n")
                report.append("\n| 代码 | 名称 | 行业 | F-Score | ROA+ | OCF+ | ROA增 | 现金>利润 |")
                report.append("|------|------|------|---------|------|------|-------|-----------|")
                for _, row in low_score.head(30).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {int(row['F_Score'])} | {int(row['F1_ROA_positive'])} | {int(row['F2_OCF_positive'])} | {int(row['F3_ROA_increase'])} | {int(row['F4_accrual_negative'])} |")

        # 异常模式检测
        report.append("\n---\n")
        report.append("## 异常模式检测\n")

        if 'anomaly_patterns' in self.results:
            patterns = self.results['anomaly_patterns']

            if 'inventory_anomaly' in patterns and len(patterns['inventory_anomaly']) > 0:
                df = patterns['inventory_anomaly']
                report.append("### 存货异常增长\n")
                report.append(f"存货增速显著超过收入增速的公司 (共 {len(df)} 家):\n")
                report.append("\n| 代码 | 名称 | 行业 | 存货增长(%) | 收入增长(%) | 背离(%) |")
                report.append("|------|------|------|-------------|-------------|---------|")
                for _, row in df.head(20).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row.get('inv_growth', 0):.1f} | {row.get('rev_growth', 0):.1f} | {row['inv_rev_divergence']:.1f} |")

            if 'ar_anomaly' in patterns and len(patterns['ar_anomaly']) > 0:
                df = patterns['ar_anomaly']
                report.append("\n### 应收账款异常\n")
                report.append(f"应收增速显著超过收入增速的公司 (共 {len(df)} 家):\n")
                report.append("\n| 代码 | 名称 | 行业 | 应收增长(%) | 收入增长(%) | 背离(%) |")
                report.append("|------|------|------|-------------|-------------|---------|")
                for _, row in df.head(20).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row.get('ar_growth', 0):.1f} | {row.get('rev_growth', 0):.1f} | {row['ar_rev_divergence']:.1f} |")

            if 'margin_anomaly' in patterns and len(patterns['margin_anomaly']) > 0:
                df = patterns['margin_anomaly']
                report.append("\n### 毛利率异常波动\n")
                report.append(f"毛利率偏离历史均值超过2个标准差的公司 (共 {len(df)} 家):\n")
                report.append("\n| 代码 | 名称 | 行业 | 当前毛利率(%) | 历史均值(%) | Z-score |")
                report.append("|------|------|------|---------------|-------------|---------|")
                for _, row in df.head(20).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row.get('current_margin', 0):.1f} | {row.get('avg_margin', 0):.1f} | {row['margin_zscore']:.2f} |")

        # 历史案例分析
        report.append("\n---\n")
        report.append("## 历史案例分析\n")

        if 'historical_cases' in self.results:
            cases = self.results['historical_cases']

            if 'st_companies' in cases:
                df = cases['st_companies']
                report.append("### 当前ST公司\n")
                star_st = df[df['st_type'].str.contains('\\*ST')]
                st = df[~df['st_type'].str.contains('\\*ST')]
                report.append(f"- *ST (退市风险警示): {len(star_st)} 家")
                report.append(f"- ST (特别处理): {len(st)} 家\n")

                report.append("\n**行业分布**:\n")
                report.append("| 行业 | 数量 |")
                report.append("|------|------|")
                industry_dist = df['industry'].value_counts().head(10)
                for ind, cnt in industry_dist.items():
                    report.append(f"| {ind} | {cnt} |")

            if 'st_financials' in cases and len(cases['st_financials']) > 0:
                df = cases['st_financials']
                report.append("\n### ST公司财务特征\n")
                report.append(f"基于 {latest_period} 财报数据:\n")
                report.append(f"- ROE中位数: {df['roe'].median():.2f}%")
                report.append(f"- ROA中位数: {df['roa'].median():.2f}%")
                report.append(f"- 流动比率中位数: {df['current_ratio'].median():.2f}")
                report.append(f"- 亏损公司占比: {(df['net_income'] < 0).mean()*100:.1f}%")
                report.append(f"- 经营现金流为负占比: {(df['ocf'] < 0).mean()*100:.1f}%\n")

        # 高风险股票名单
        report.append("\n---\n")
        report.append("## 高风险股票名单\n")

        if 'composite_risk' in self.results:
            df = self.results['composite_risk']

            extreme_risk = df[df['risk_level'] == '极高风险'].sort_values('composite_risk_score', ascending=False)
            high_risk = df[df['risk_level'] == '高风险'].sort_values('composite_risk_score', ascending=False)

            report.append("### 综合风险评分说明\n")
            report.append("综合风险评分整合了 M-Score、Z-Score 和 F-Score 三个模型的结果:\n")
            report.append("- M-Score 贡献: 0-30分 (盈余操纵风险)")
            report.append("- Z-Score 贡献: 0-30分 (破产风险)")
            report.append("- F-Score 贡献: 0-40分 (财务健康风险)\n")
            report.append("**风险等级划分**:\n")
            report.append("- 极高风险: 60-100分")
            report.append("- 高风险: 40-60分")
            report.append("- 中等风险: 20-40分")
            report.append("- 低风险: 0-20分\n")

            report.append(f"\n### 风险等级分布\n")
            report.append(f"- 极高风险: {len(extreme_risk)} 家 ({len(extreme_risk)/len(df)*100:.1f}%)")
            report.append(f"- 高风险: {len(high_risk)} 家 ({len(high_risk)/len(df)*100:.1f}%)")
            report.append(f"- 中等风险: {len(df[df['risk_level'] == '中等风险'])} 家")
            report.append(f"- 低风险: {len(df[df['risk_level'] == '低风险'])} 家\n")

            if len(extreme_risk) > 0:
                report.append("\n### 极高风险公司完整名单\n")
                report.append("\n| 代码 | 名称 | 行业 | 综合评分 | M-Score | Z-Score | F-Score |")
                report.append("|------|------|------|----------|---------|---------|---------|")
                for _, row in extreme_risk.iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['composite_risk_score']:.0f} | {row['M_Score']:.2f} | {row['Z_Score']:.2f} | {int(row['F_Score'])} |")

            if len(high_risk) > 0:
                report.append("\n### 高风险公司名单 (前50家)\n")
                report.append("\n| 代码 | 名称 | 行业 | 综合评分 | M-Score | Z-Score | F-Score |")
                report.append("|------|------|------|----------|---------|---------|---------|")
                for _, row in high_risk.head(50).iterrows():
                    report.append(f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['composite_risk_score']:.0f} | {row['M_Score']:.2f} | {row['Z_Score']:.2f} | {int(row['F_Score'])} |")

        # 风险预警建议
        report.append("\n---\n")
        report.append("## 风险预警建议\n")

        report.append("### 投资者建议\n")
        report.append("1. **回避极高风险股票**: 综合评分超过60分的公司存在多重财务风险，建议回避投资。\n")
        report.append("2. **审慎对待高风险股票**: 对综合评分40-60分的公司，需要深入调研其具体风险点。\n")
        report.append("3. **关注核心指标变化**: 定期跟踪应收账款周转、存货周转、毛利率等指标的异常变动。\n")
        report.append("4. **现金流为王**: 优先选择经营现金流持续为正、应计利润较低的公司。\n")
        report.append("5. **分散投资**: 避免过度集中于高风险行业或个股。\n")

        report.append("\n### 监管建议\n")
        report.append("1. **重点关注**: 对M-Score显著高于-1.78的公司进行重点审查。\n")
        report.append("2. **持续监控**: 建立动态预警机制，跟踪高风险公司的财务变化。\n")
        report.append("3. **信息披露**: 加强对非经常性损益、关联交易等事项的披露要求。\n")
        report.append("4. **审计关注**: 对存货、应收账款异常增长的公司加强审计程序。\n")

        report.append("\n### 后续分析建议\n")
        report.append("1. 结合行业特点进行细分分析")
        report.append("2. 追踪季度财报的变化趋势")
        report.append("3. 结合非财务信息进行交叉验证")
        report.append("4. 关注审计意见和会计政策变更")
        report.append("5. 建立定期更新机制\n")

        report.append("\n---\n")
        report.append("## 免责声明\n")
        report.append("本报告仅供参考，不构成投资建议。财务分析模型存在局限性，")
        report.append("实际投资决策需结合更多信息进行综合判断。模型识别的风险公司")
        report.append("不代表一定存在财务造假，同样，未被识别的公司也不代表完全没有风险。")
        report.append("请投资者审慎决策，风险自担。\n")

        # 写入文件
        report_path = REPORT_DIR / "financial_anomaly_detection.md"
        report_content = "\n".join(report)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n报告已保存至: {report_path}")

        return report_content


def main():
    """主函数"""
    print("\n" + "="*70)
    print("         财务异常检测系统 - Financial Anomaly Detection System")
    print("="*70)

    # 初始化检测器
    detector = FinancialAnomalyDetector(DB_PATH)

    try:
        # 使用最新可用的年报数据
        # 先检查最新数据
        latest_query = "SELECT MAX(end_date) as latest FROM income WHERE report_type = '1' AND end_date LIKE '%1231'"
        latest = detector._execute_query(latest_query)

        if len(latest) > 0 and latest['latest'].iloc[0]:
            latest_period = latest['latest'].iloc[0]
        else:
            latest_period = '20231231'

        print(f"\n使用财报期间: {latest_period}")

        # 生成完整报告
        detector.generate_report(latest_period)

    finally:
        detector.close()

    print("\n" + "="*70)
    print("                          分析完成")
    print("="*70)


if __name__ == "__main__":
    main()
