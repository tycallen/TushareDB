"""
板块资金流向策略示例

结合资金流向和板块传导关系，捕捉资金潜伏信号

策略逻辑：
1. 计算板块资金净流入（聚合个股数据）
2. 检测资金潜伏信号（量增价平）
3. 结合传导关系预测跟随板块
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db import DataReader


class SectorMoneyFlowAnalyzer:
    """板块资金流向分析器"""

    def __init__(self, db_path: str):
        self.reader = DataReader(db_path=db_path)

    def calculate_sector_money_flow(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        计算板块资金净流入

        通过聚合成分股的资金流向数据到板块层级

        Returns:
            DataFrame with columns: trade_date, sector_code, sector_name,
                                   net_amount, buy_lg_amount, vol, amount
        """

        # 获取板块成分股映射
        if level == 'L1':
            code_col = 'l1_code'
            name_col = 'l1_name'
        elif level == 'L2':
            code_col = 'l2_code'
            name_col = 'l2_name'
        else:  # L3
            code_col = 'l3_code'
            name_col = 'l3_name'

        # SQL查询：聚合个股资金流向到板块
        query = f"""
        WITH sector_stocks AS (
            SELECT DISTINCT
                ts_code,
                {code_col} as sector_code,
                {name_col} as sector_name
            FROM index_member_all
            WHERE is_new = 'Y'
            AND {code_col} IS NOT NULL
            AND in_date <= '{end_date}'
            AND (out_date IS NULL OR out_date >= '{start_date}')
        )
        SELECT
            m.trade_date,
            s.sector_code,
            s.sector_name,
            SUM(m.net_mf_amount) as net_amount,
            SUM(m.buy_lg_amount + m.buy_elg_amount) as buy_lg_amount,
            SUM(m.sell_lg_amount + m.sell_elg_amount) as sell_lg_amount,
            SUM(m.buy_sm_amount + m.buy_md_amount) as buy_sm_amount,
            COUNT(DISTINCT m.ts_code) as stock_count
        FROM moneyflow m
        INNER JOIN sector_stocks s ON m.ts_code = s.ts_code
        WHERE m.trade_date >= '{start_date}'
        AND m.trade_date <= '{end_date}'
        GROUP BY m.trade_date, s.sector_code, s.sector_name
        ORDER BY m.trade_date, s.sector_code
        """

        df = self.reader.db.con.execute(query).fetchdf()

        # 计算资金流向指标
        df['lg_net_amount'] = df['buy_lg_amount'] - df['sell_lg_amount']  # 大单净流入
        df['lg_ratio'] = df['lg_net_amount'] / (df['buy_lg_amount'] + df['sell_lg_amount'] + 1)  # 大单净流入占比

        return df

    def calculate_sector_volume_price(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        计算板块成交量和涨跌幅

        Returns:
            DataFrame with: trade_date, sector_code, vol, amount, pct_chg
        """

        if level == 'L1':
            code_col = 'l1_code'
        elif level == 'L2':
            code_col = 'l2_code'
        else:
            code_col = 'l3_code'

        query = f"""
        WITH sector_stocks AS (
            SELECT DISTINCT
                ts_code,
                {code_col} as sector_code
            FROM index_member_all
            WHERE is_new = 'Y'
            AND {code_col} IS NOT NULL
            AND in_date <= '{end_date}'
            AND (out_date IS NULL OR out_date >= '{start_date}')
        )
        SELECT
            d.trade_date,
            s.sector_code,
            SUM(d.vol) as total_vol,
            SUM(d.amount) as total_amount,
            AVG(d.pct_chg) as avg_pct_chg  -- 简化：用平均涨跌幅
        FROM daily d
        INNER JOIN sector_stocks s ON d.ts_code = s.ts_code
        WHERE d.trade_date >= '{start_date}'
        AND d.trade_date <= '{end_date}'
        GROUP BY d.trade_date, s.sector_code
        """

        return self.reader.db.con.execute(query).fetchdf()

    def detect_money_lurking(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        检测资金潜伏信号

        信号定义：
        1. 大单净流入显著增加（>历史均值的1.5倍）
        2. 但涨跌幅较小（<历史均值）
        3. 持续出现2-3天

        Returns:
            DataFrame with lurking signals
        """

        # 获取资金流向数据
        money_df = self.calculate_sector_money_flow(start_date, end_date, level)

        # 获取涨跌幅数据
        price_df = self.calculate_sector_volume_price(start_date, end_date, level)

        # 合并
        df = money_df.merge(
            price_df[['trade_date', 'sector_code', 'avg_pct_chg']],
            on=['trade_date', 'sector_code'],
            how='left'
        )

        # 计算滚动统计量
        df = df.sort_values(['sector_code', 'trade_date'])

        for sector in df['sector_code'].unique():
            mask = df['sector_code'] == sector

            # 大单净流入的历史均值和标准差
            df.loc[mask, 'lg_net_ma'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback_days, min_periods=5
            ).mean()
            df.loc[mask, 'lg_net_std'] = df.loc[mask, 'lg_net_amount'].rolling(
                lookback_days, min_periods=5
            ).std()

            # 涨跌幅的历史均值
            df.loc[mask, 'pct_chg_ma'] = df.loc[mask, 'avg_pct_chg'].rolling(
                lookback_days, min_periods=5
            ).mean()

        # 计算资金潜伏度指标
        df['lg_net_zscore'] = (df['lg_net_amount'] - df['lg_net_ma']) / (df['lg_net_std'] + 1e-6)
        df['pct_chg_zscore'] = (df['avg_pct_chg'] - df['pct_chg_ma']) / (df['pct_chg_ma'].abs() + 0.01)

        # 潜伏信号：资金流入强但涨幅弱
        df['lurking_signal'] = (
            (df['lg_net_zscore'] > 1.0) &  # 大单净流入 > 均值 + 1倍标准差
            (df['pct_chg_zscore'] < 0.5) &  # 涨幅 < 历史均值的1.5倍
            (df['lg_net_amount'] > 0)  # 大单净流入为正
        )

        # 筛选出潜伏信号
        lurking_df = df[df['lurking_signal']].copy()

        return lurking_df

    def close(self):
        self.reader.close()


def demo_money_lurking_detection():
    """演示资金潜伏检测"""

    db_path = "tushare.db"
    analyzer = SectorMoneyFlowAnalyzer(db_path)

    # 检测最近3个月的资金潜伏信号
    end_date = '20241231'
    start_date = '20241001'

    print("=" * 80)
    print("板块资金潜伏信号检测")
    print("=" * 80)
    print(f"分析时间: {start_date} ~ {end_date}")
    print()

    # 检测L1板块资金潜伏
    print("正在检测L1板块资金潜伏信号...")
    lurking_df = analyzer.detect_money_lurking(
        start_date=start_date,
        end_date=end_date,
        level='L1',
        lookback_days=20
    )

    if len(lurking_df) > 0:
        print(f"\n找到 {len(lurking_df)} 个潜伏信号\n")

        # 显示最近的信号
        recent_signals = lurking_df.sort_values('trade_date', ascending=False).head(10)[
            ['trade_date', 'sector_name', 'lg_net_amount', 'avg_pct_chg',
             'lg_net_zscore', 'pct_chg_zscore']
        ]

        print("最近的资金潜伏信号：")
        print(recent_signals.to_string(index=False))

        # 统计各板块信号出现次数
        print("\n各板块潜伏信号统计：")
        signal_count = lurking_df.groupby('sector_name').size().sort_values(ascending=False)
        print(signal_count.head(10))

    else:
        print("未发现明显的资金潜伏信号")

    analyzer.close()


def demo_money_flow_lead_lag():
    """演示资金流向与价格传导的关系"""

    db_path = "tushare.db"
    analyzer = SectorMoneyFlowAnalyzer(db_path)

    print("\n" + "=" * 80)
    print("资金流向-价格传导分析")
    print("=" * 80)

    # 分析思路：
    # 1. 板块A出现资金潜伏（T日）
    # 2. 板块A在T+N日价格上涨
    # 3. 检查是否有板块B在T+N+M日跟随上涨（传导）

    print("\n此功能需要结合传导关系分析，建议：")
    print("1. 先用detect_money_lurking()检测潜伏信号")
    print("2. 再用sector_analysis模块的lead-lag分析验证传导")
    print("3. 将两者结合构建交易策略")

    analyzer.close()


if __name__ == '__main__':
    print("板块资金流向策略示例\n")

    # 演示1：资金潜伏检测
    demo_money_lurking_detection()

    # 演示2：资金流向与传导的关系
    demo_money_flow_lead_lag()

    print("\n" + "=" * 80)
    print("策略建议")
    print("=" * 80)
    print("""
    1. 资金潜伏信号：大单净流入增加但涨幅不大
       - 可能预示后续上涨
       - 结合传导关系找跟随板块

    2. 量价配合传导：
       - 领涨板块：量价齐升
       - 跟随板块：先放量，后上涨

    3. 主力资金轮动：
       - 追踪大单资金从哪流出、流入哪
       - 验证是否符合历史传导规律

    4. 回测验证：
       - 统计潜伏信号后N日收益
       - 优化参数（阈值、持续天数等）
       - 结合传导关系提高胜率
    """)
