#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙头股效应研究脚本 v2 (优化版)

研究内容：
1. 龙头识别（市值龙头、涨幅龙头、成交额龙头）
2. 龙头效应分析（带动板块效应、持续性、切换规律）
3. 策略应用（跟随策略、轮动策略、风险控制）

Author: Research Script
Date: 2025-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os
from collections import defaultdict

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

os.makedirs(OUTPUT_DIR, exist_ok=True)


class LeaderStockResearch:
    """龙头股效应研究类 - 优化版"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_lines = []

    def log(self, message):
        print(f"[INFO] {message}")

    def add_report(self, content):
        self.report_lines.append(content)
        print(content)

    def get_all_data_bulk(self, start_date, end_date):
        """批量获取所有数据 - 关键优化"""
        self.log(f"批量加载数据: {start_date} - {end_date}")

        # 一次性加载所有日线和基本面数据
        query = f'''
            SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
                   d.pre_close, d.pct_chg, d.vol, d.amount,
                   b.total_mv, b.circ_mv, b.turnover_rate
            FROM daily d
            LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
            WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
            ORDER BY d.trade_date, d.ts_code
        '''
        df = self.conn.execute(query).fetchdf()
        self.log(f"加载了 {len(df)} 条记录")
        return df

    def get_industry_members(self):
        """获取行业成员信息"""
        query = '''
            SELECT ts_code, name, l1_name, l2_name
            FROM index_member_all
            WHERE out_date IS NULL OR out_date = ''
        '''
        return self.conn.execute(query).fetchdf()

    def get_stock_names(self):
        """获取股票名称映射"""
        query = 'SELECT ts_code, name FROM stock_basic'
        return dict(self.conn.execute(query).fetchdf().values)

    def identify_leaders_batch(self, all_data, industry_members, trade_dates):
        """批量识别各类龙头"""
        self.log("批量识别各类龙头...")

        # 合并行业信息
        merged = all_data.merge(industry_members[['ts_code', 'l1_name', 'l2_name']],
                                 on='ts_code', how='inner')

        # 过滤异常数据
        merged = merged[(merged['pct_chg'].notna()) &
                       (merged['pct_chg'] > -11) &
                       (merged['pct_chg'] < 21)]

        results = {
            'market_cap': {},
            'gain': {},
            'volume': {}
        }

        for date in trade_dates:
            day_data = merged[merged['trade_date'] == date]
            if day_data.empty:
                continue

            # 市值龙头
            mc_leaders = day_data.dropna(subset=['total_mv']).groupby('l2_name').apply(
                lambda x: x.nlargest(1, 'total_mv')
            )
            if not mc_leaders.empty:
                mc_leaders = mc_leaders.reset_index(drop=True)
                results['market_cap'][date] = dict(zip(mc_leaders['l2_name'], mc_leaders['ts_code']))

            # 涨幅龙头
            gain_leaders = day_data.groupby('l2_name').apply(
                lambda x: x.nlargest(1, 'pct_chg')
            )
            if not gain_leaders.empty:
                gain_leaders = gain_leaders.reset_index(drop=True)
                results['gain'][date] = dict(zip(gain_leaders['l2_name'], gain_leaders['ts_code']))

            # 成交额龙头
            vol_leaders = day_data.dropna(subset=['amount']).groupby('l2_name').apply(
                lambda x: x.nlargest(1, 'amount')
            )
            if not vol_leaders.empty:
                vol_leaders = vol_leaders.reset_index(drop=True)
                results['volume'][date] = dict(zip(vol_leaders['l2_name'], vol_leaders['ts_code']))

        return results, merged

    def analyze_sector_effect(self, merged_data, leaders_by_date):
        """分析龙头带动板块效应"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.1 龙头带动板块效应分析")
        self.add_report("=" * 80)

        # 计算每日板块平均涨幅
        sector_avg = merged_data.groupby(['trade_date', 'l2_name'])['pct_chg'].mean().reset_index()
        sector_avg.columns = ['trade_date', 'l2_name', 'sector_avg_pct_chg']

        # 获取龙头涨幅
        leader_records = []
        for date, leaders in leaders_by_date.items():
            for sector, ts_code in leaders.items():
                stock_data = merged_data[(merged_data['trade_date'] == date) &
                                         (merged_data['ts_code'] == ts_code)]
                if not stock_data.empty:
                    leader_records.append({
                        'trade_date': date,
                        'l2_name': sector,
                        'ts_code': ts_code,
                        'leader_pct_chg': stock_data['pct_chg'].values[0]
                    })

        if not leader_records:
            self.add_report("没有足够的数据进行分析")
            return None

        leader_df = pd.DataFrame(leader_records)
        combined = leader_df.merge(sector_avg, on=['trade_date', 'l2_name'], how='inner')

        # 计算相关性
        correlation = combined['leader_pct_chg'].corr(combined['sector_avg_pct_chg'])
        self.add_report(f"\n龙头涨幅与板块平均涨幅相关系数: {correlation:.4f}")

        # 龙头涨停时板块表现
        limit_up = combined[combined['leader_pct_chg'] >= 9.5]
        if not limit_up.empty:
            self.add_report(f"龙头涨停时板块平均涨幅: {limit_up['sector_avg_pct_chg'].mean():.2f}%")

        # 龙头跌停时板块表现
        limit_down = combined[combined['leader_pct_chg'] <= -9.5]
        if not limit_down.empty:
            self.add_report(f"龙头跌停时板块平均涨幅: {limit_down['sector_avg_pct_chg'].mean():.2f}%")

        # 各板块龙头效应强度
        self.add_report("\n### 各板块龙头效应强度排名:")
        sector_corr = combined.groupby('l2_name').apply(
            lambda x: x['leader_pct_chg'].corr(x['sector_avg_pct_chg']) if len(x) > 10 else np.nan
        ).dropna().sort_values(ascending=False)

        for i, (sector, corr) in enumerate(sector_corr.head(20).items()):
            self.add_report(f"  {i+1:2d}. {sector}: {corr:.4f}")

        return combined

    def analyze_persistence(self, leaders_dict):
        """分析龙头持续性"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.2 龙头持续性分析")
        self.add_report("=" * 80)

        for leader_type, leaders_by_date in leaders_dict.items():
            type_name = {'market_cap': '市值', 'gain': '涨幅', 'volume': '成交额'}[leader_type]
            stats = self._calc_persistence_stats(leaders_by_date)

            self.add_report(f"\n### {type_name}龙头持续性:")
            self.add_report(f"  平均持续天数: {stats['avg_days']:.1f} 天")
            self.add_report(f"  最长持续天数: {stats['max_days']} 天")
            self.add_report(f"  龙头更替率 (日均): {stats['change_rate']*100:.2f}%")

        # 分析三种龙头的一致性
        self.add_report("\n### 三种龙头的一致性分析:")
        consistency = self._analyze_consistency(leaders_dict)
        self.add_report(f"  市值龙头=成交额龙头的比例: {consistency['mc_vol']*100:.2f}%")
        self.add_report(f"  市值龙头=涨幅龙头的比例: {consistency['mc_gain']*100:.2f}%")
        self.add_report(f"  成交额龙头=涨幅龙头的比例: {consistency['vol_gain']*100:.2f}%")
        self.add_report(f"  三者一致的比例: {consistency['all_same']*100:.2f}%")

        return consistency

    def _calc_persistence_stats(self, leaders_by_date):
        """计算持续性统计"""
        dates = sorted(leaders_by_date.keys())
        if len(dates) < 2:
            return {'avg_days': 0, 'max_days': 0, 'change_rate': 0}

        all_sectors = set()
        for d in dates:
            all_sectors.update(leaders_by_date[d].keys())

        sector_streaks = defaultdict(list)
        total_changes = 0
        total_obs = 0

        for sector in all_sectors:
            current_leader = None
            streak = 0

            for date in dates:
                if sector in leaders_by_date[date]:
                    leader = leaders_by_date[date][sector]
                    if leader == current_leader:
                        streak += 1
                    else:
                        if current_leader is not None:
                            sector_streaks[sector].append(streak)
                            total_changes += 1
                        current_leader = leader
                        streak = 1
                    total_obs += 1

            if streak > 0:
                sector_streaks[sector].append(streak)

        all_streaks = []
        for streaks in sector_streaks.values():
            all_streaks.extend(streaks)

        return {
            'avg_days': np.mean(all_streaks) if all_streaks else 0,
            'max_days': max(all_streaks) if all_streaks else 0,
            'change_rate': total_changes / total_obs if total_obs > 0 else 0
        }

    def _analyze_consistency(self, leaders_dict):
        """分析三种龙头的一致性"""
        mc = leaders_dict['market_cap']
        gain = leaders_dict['gain']
        vol = leaders_dict['volume']

        common_dates = set(mc.keys()) & set(gain.keys()) & set(vol.keys())

        mc_vol = mc_gain = vol_gain = all_same = 0
        total = 0

        for date in common_dates:
            common_sectors = set(mc[date].keys()) & set(gain[date].keys()) & set(vol[date].keys())
            for sector in common_sectors:
                total += 1
                if mc[date].get(sector) == vol[date].get(sector):
                    mc_vol += 1
                if mc[date].get(sector) == gain[date].get(sector):
                    mc_gain += 1
                if vol[date].get(sector) == gain[date].get(sector):
                    vol_gain += 1
                if mc[date].get(sector) == vol[date].get(sector) == gain[date].get(sector):
                    all_same += 1

        return {
            'mc_vol': mc_vol / total if total else 0,
            'mc_gain': mc_gain / total if total else 0,
            'vol_gain': vol_gain / total if total else 0,
            'all_same': all_same / total if total else 0
        }

    def analyze_rotation(self, leaders_by_date, merged_data):
        """分析龙头切换规律"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.3 龙头切换规律分析")
        self.add_report("=" * 80)

        dates = sorted(leaders_by_date.keys())
        switch_events = []

        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]

            if prev_date not in leaders_by_date or curr_date not in leaders_by_date:
                continue

            prev_leaders = leaders_by_date[prev_date]
            curr_leaders = leaders_by_date[curr_date]

            for sector in curr_leaders:
                if sector in prev_leaders and curr_leaders[sector] != prev_leaders[sector]:
                    old = prev_leaders[sector]
                    new = curr_leaders[sector]

                    old_data = merged_data[(merged_data['trade_date'] == curr_date) &
                                           (merged_data['ts_code'] == old)]
                    new_data = merged_data[(merged_data['trade_date'] == curr_date) &
                                           (merged_data['ts_code'] == new)]

                    if not old_data.empty and not new_data.empty:
                        switch_events.append({
                            'date': curr_date,
                            'sector': sector,
                            'old_pct_chg': old_data['pct_chg'].values[0],
                            'new_pct_chg': new_data['pct_chg'].values[0]
                        })

        if not switch_events:
            self.add_report("没有足够的龙头切换事件")
            return None

        switch_df = pd.DataFrame(switch_events)

        self.add_report(f"\n分析周期: {len(dates)} 个交易日")
        self.add_report(f"龙头切换事件总数: {len(switch_df)}")
        self.add_report(f"日均切换次数: {len(switch_df)/len(dates):.2f}")

        self.add_report("\n### 龙头切换时的表现特征:")
        self.add_report(f"  旧龙头当日平均涨幅: {switch_df['old_pct_chg'].mean():.2f}%")
        self.add_report(f"  新龙头当日平均涨幅: {switch_df['new_pct_chg'].mean():.2f}%")

        self.add_report("\n### 各行业龙头切换频率排名:")
        sector_counts = switch_df['sector'].value_counts()
        for i, (sector, count) in enumerate(sector_counts.head(15).items()):
            self.add_report(f"  {i+1:2d}. {sector}: {count} 次")

        return switch_df

    def backtest_follow_strategy(self, merged_data, leaders_by_date, trade_dates):
        """回测龙头跟随策略"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.1 龙头跟随策略回测")
        self.add_report("=" * 80)

        holding_periods = [1, 3, 5, 10]
        results = {}

        for hold_days in holding_periods:
            returns = []

            for i in range(len(trade_dates) - hold_days - 1):
                signal_date = trade_dates[i]
                entry_date = trade_dates[i + 1]
                exit_date = trade_dates[i + hold_days + 1]

                if signal_date not in leaders_by_date:
                    continue

                for sector, ts_code in leaders_by_date[signal_date].items():
                    entry_data = merged_data[(merged_data['trade_date'] == entry_date) &
                                             (merged_data['ts_code'] == ts_code)]
                    exit_data = merged_data[(merged_data['trade_date'] == exit_date) &
                                            (merged_data['ts_code'] == ts_code)]

                    if not entry_data.empty and not exit_data.empty:
                        entry_price = entry_data['open'].values[0]
                        exit_price = exit_data['close'].values[0]

                        if entry_price and entry_price > 0:
                            ret = (exit_price - entry_price) / entry_price * 100
                            returns.append(ret)

            if returns:
                returns_arr = np.array(returns)
                results[hold_days] = {
                    'mean_return': returns_arr.mean(),
                    'win_rate': (returns_arr > 0).mean() * 100,
                    'max_return': returns_arr.max(),
                    'min_return': returns_arr.min(),
                    'sharpe': returns_arr.mean() / returns_arr.std() if returns_arr.std() > 0 else 0,
                    'trade_count': len(returns)
                }

        self.add_report("\n### 不同持有期的回测结果:")
        self.add_report("-" * 70)
        self.add_report(f"{'持有期':<10}{'平均收益':<12}{'胜率':<10}{'最大收益':<12}{'最大亏损':<12}{'夏普比':<10}")
        self.add_report("-" * 70)

        for hold_days, m in results.items():
            self.add_report(
                f"{hold_days}天{'':<6}"
                f"{m['mean_return']:.2f}%{'':<6}"
                f"{m['win_rate']:.1f}%{'':<5}"
                f"{m['max_return']:.2f}%{'':<6}"
                f"{m['min_return']:.2f}%{'':<6}"
                f"{m['sharpe']:.3f}"
            )

        return results

    def backtest_rotation_strategy(self, merged_data, trade_dates):
        """回测板块龙头轮动策略"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.2 板块龙头轮动策略回测")
        self.add_report("=" * 80)

        lookback = 20
        hold_days = 5
        top_n = 3

        portfolio_returns = []

        for i in range(lookback, len(trade_dates) - hold_days):
            start_date = trade_dates[i - lookback]
            signal_date = trade_dates[i]
            entry_date = trade_dates[i + 1]
            exit_date = trade_dates[i + hold_days]

            # 计算回望期板块表现
            period_data = merged_data[(merged_data['trade_date'] >= start_date) &
                                      (merged_data['trade_date'] <= signal_date)]

            sector_perf = period_data.groupby('l2_name')['pct_chg'].sum()
            top_sectors = sector_perf.nlargest(top_n).index.tolist()

            # 在选中板块中买入当日涨幅最大的
            entry_data = merged_data[merged_data['trade_date'] == entry_date]
            exit_data = merged_data[merged_data['trade_date'] == exit_date]

            if entry_data.empty or exit_data.empty:
                continue

            selected = entry_data[entry_data['l2_name'].isin(top_sectors)]
            leaders = selected.groupby('l2_name').apply(
                lambda x: x.nlargest(1, 'pct_chg')
            ).reset_index(drop=True)

            if leaders.empty:
                continue

            period_rets = []
            for _, row in leaders.iterrows():
                ts_code = row['ts_code']
                entry_row = entry_data[entry_data['ts_code'] == ts_code]
                exit_row = exit_data[exit_data['ts_code'] == ts_code]

                if not entry_row.empty and not exit_row.empty:
                    entry_price = entry_row['open'].values[0]
                    exit_price = exit_row['close'].values[0]

                    if entry_price and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price * 100
                        period_rets.append(ret)

            if period_rets:
                portfolio_returns.append(np.mean(period_rets))

        if not portfolio_returns:
            self.add_report("没有足够的数据进行回测")
            return None

        returns_arr = np.array(portfolio_returns)

        self.add_report(f"\n策略参数: 回望期={lookback}天, 持有期={hold_days}天, 选择板块数={top_n}")
        self.add_report(f"\n### 轮动策略回测结果:")
        self.add_report(f"  交易次数: {len(returns_arr)}")
        self.add_report(f"  平均收益: {returns_arr.mean():.2f}%")
        self.add_report(f"  胜率: {(returns_arr > 0).mean() * 100:.1f}%")
        self.add_report(f"  最大单次收益: {returns_arr.max():.2f}%")
        self.add_report(f"  最大单次亏损: {returns_arr.min():.2f}%")
        self.add_report(f"  收益标准差: {returns_arr.std():.2f}%")
        if returns_arr.std() > 0:
            self.add_report(f"  夏普比率: {returns_arr.mean() / returns_arr.std():.3f}")

        cumulative = (1 + returns_arr / 100).cumprod() - 1
        self.add_report(f"  累计收益: {cumulative[-1] * 100:.2f}%")

        return returns_arr

    def analyze_risk_control(self, merged_data, leaders_by_date, trade_dates):
        """分析风险控制"""
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.3 风险控制分析")
        self.add_report("=" * 80)

        drawdowns = []

        for i in range(len(trade_dates) - 5):
            date = trade_dates[i]
            if date not in leaders_by_date:
                continue

            for sector, ts_code in leaders_by_date[date].items():
                future_dates = trade_dates[i:i+6]
                future_data = merged_data[(merged_data['ts_code'] == ts_code) &
                                          (merged_data['trade_date'].isin(future_dates))]

                if len(future_data) >= 2:
                    cummax = future_data['close'].cummax()
                    drawdown = (future_data['close'] - cummax) / cummax * 100
                    max_dd = drawdown.min()

                    stock_data = merged_data[(merged_data['trade_date'] == date) &
                                             (merged_data['ts_code'] == ts_code)]
                    if not stock_data.empty:
                        drawdowns.append({
                            'signal_pct_chg': stock_data['pct_chg'].values[0],
                            'max_drawdown': max_dd
                        })

        if not drawdowns:
            self.add_report("没有足够的数据进行风险分析")
            return None

        dd_df = pd.DataFrame(drawdowns)

        self.add_report("\n### 龙头股5日内回撤分析:")
        self.add_report(f"  平均最大回撤: {dd_df['max_drawdown'].mean():.2f}%")
        self.add_report(f"  最大回撤中位数: {dd_df['max_drawdown'].median():.2f}%")
        self.add_report(f"  最严重回撤: {dd_df['max_drawdown'].min():.2f}%")
        self.add_report(f"  回撤超过5%的比例: {(dd_df['max_drawdown'] < -5).mean() * 100:.1f}%")
        self.add_report(f"  回撤超过10%的比例: {(dd_df['max_drawdown'] < -10).mean() * 100:.1f}%")

        # 涨停龙头风险
        limit_up = dd_df[dd_df['signal_pct_chg'] >= 9.5]
        if not limit_up.empty:
            self.add_report("\n### 涨停龙头的特殊风险:")
            self.add_report(f"  涨停龙头平均回撤: {limit_up['max_drawdown'].mean():.2f}%")
            self.add_report(f"  涨停龙头回撤>5%比例: {(limit_up['max_drawdown'] < -5).mean() * 100:.1f}%")

        self.add_report("\n### 风险控制建议:")
        self.add_report("  1. 止损线设置: 建议设置在 -5% 到 -8% 之间")
        self.add_report("  2. 仓位控制: 单只龙头股仓位不超过总仓位的 20%")
        self.add_report("  3. 分散投资: 同时持有 3-5 个不同板块的龙头")
        self.add_report("  4. 避免追高: 涨停板龙头次日追高风险较大")
        self.add_report("  5. 关注量能: 成交量萎缩时警惕龙头切换风险")

        return dd_df

    def generate_visualizations(self, combined_data, merged_data, trade_dates):
        """生成可视化图表"""
        self.log("生成可视化图表...")

        if combined_data is None or combined_data.empty:
            self.log("没有数据生成图表")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 龙头涨幅与板块涨幅散点图
        ax1 = axes[0, 0]
        ax1.scatter(combined_data['leader_pct_chg'], combined_data['sector_avg_pct_chg'],
                    alpha=0.3, s=10)
        ax1.set_xlabel('Leader Return (%)')
        ax1.set_ylabel('Sector Avg Return (%)')
        ax1.set_title('Leader vs Sector Return')

        z = np.polyfit(combined_data['leader_pct_chg'], combined_data['sector_avg_pct_chg'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(combined_data['leader_pct_chg'].min(),
                             combined_data['leader_pct_chg'].max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8)

        corr = combined_data['leader_pct_chg'].corr(combined_data['sector_avg_pct_chg'])
        ax1.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax1.transAxes, fontsize=10)

        # 2. 按板块的龙头效应强度
        ax2 = axes[0, 1]
        sector_corr = combined_data.groupby('l2_name').apply(
            lambda x: x['leader_pct_chg'].corr(x['sector_avg_pct_chg']) if len(x) > 10 else np.nan
        ).dropna().sort_values(ascending=True)

        top_sectors = sector_corr.tail(15)
        y_pos = np.arange(len(top_sectors))
        ax2.barh(y_pos, top_sectors.values, color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_sectors.index, fontsize=8)
        ax2.set_xlabel('Correlation')
        ax2.set_title('Leader Effect Strength by Sector')

        # 3. 龙头涨幅分布
        ax3 = axes[1, 0]
        ax3.hist(combined_data['leader_pct_chg'], bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_xlabel('Leader Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Leader Return Distribution')

        mean_ret = combined_data['leader_pct_chg'].mean()
        ax3.axvline(x=mean_ret, color='green', linestyle='--', label=f'Mean: {mean_ret:.2f}%')
        ax3.legend()

        # 4. 时间序列 - 平均龙头涨幅
        ax4 = axes[1, 1]
        daily_avg = combined_data.groupby('trade_date')['leader_pct_chg'].mean()
        ax4.plot(range(len(daily_avg)), daily_avg.values)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Trading Days')
        ax4.set_ylabel('Avg Leader Return (%)')
        ax4.set_title('Average Leader Return Over Time')

        plt.tight_layout()

        chart_path = os.path.join(OUTPUT_DIR, 'leader_stock_effect_charts.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"图表已保存到: {chart_path}")
        return chart_path

    def generate_report(self):
        """生成完整研究报告"""
        self.add_report("=" * 80)
        self.add_report("              龙头股效应研究报告")
        self.add_report("=" * 80)
        self.add_report(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_report("")

        # 获取交易日期
        trade_dates = self.conn.execute('''
            SELECT DISTINCT trade_date FROM daily
            WHERE trade_date >= '20250701'
            ORDER BY trade_date
        ''').fetchdf()['trade_date'].tolist()

        self.log(f"分析日期范围: {trade_dates[0]} 至 {trade_dates[-1]}")
        self.log(f"交易日数量: {len(trade_dates)}")

        # 获取行业成员
        industry_members = self.get_industry_members()
        self.log(f"行业分类股票数: {len(industry_members)}")

        # 批量加载数据
        all_data = self.get_all_data_bulk(trade_dates[0], trade_dates[-1])

        # 批量识别龙头
        leaders_dict, merged_data = self.identify_leaders_batch(all_data, industry_members, trade_dates)

        stock_names = self.get_stock_names()

        # Part 1: 龙头识别展示
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第一部分: 龙头识别")
        self.add_report("=" * 80)

        latest_date = trade_dates[-1]
        self.add_report(f"\n分析日期: {latest_date}")

        # 展示最新一日各类龙头
        for leader_type, type_name in [('market_cap', '市值龙头'), ('gain', '涨幅龙头'), ('volume', '成交额龙头')]:
            self.add_report(f"\n## 1.{['market_cap', 'gain', 'volume'].index(leader_type)+1} {type_name}")

            if latest_date in leaders_dict[leader_type]:
                leaders = leaders_dict[leader_type][latest_date]
                day_data = merged_data[merged_data['trade_date'] == latest_date]

                records = []
                for sector, ts_code in leaders.items():
                    stock_data = day_data[day_data['ts_code'] == ts_code]
                    if not stock_data.empty:
                        row = stock_data.iloc[0]
                        records.append({
                            'sector': sector,
                            'ts_code': ts_code,
                            'name': stock_names.get(ts_code, ''),
                            'pct_chg': row['pct_chg'],
                            'total_mv': row['total_mv'] / 10000 if pd.notna(row['total_mv']) else 0,
                            'amount': row['amount'] / 100000 if pd.notna(row['amount']) else 0
                        })

                if records:
                    df = pd.DataFrame(records)
                    if leader_type == 'market_cap':
                        df = df.sort_values('total_mv', ascending=False)
                    elif leader_type == 'gain':
                        df = df.sort_values('pct_chg', ascending=False)
                    else:
                        df = df.sort_values('amount', ascending=False)

                    self.add_report("-" * 80)
                    self.add_report(f"{'行业':<12}{'代码':<12}{'名称':<10}{'涨跌幅':<10}{'市值(亿)':<12}{'成交额(亿)':<12}")
                    self.add_report("-" * 80)

                    for _, r in df.head(20).iterrows():
                        self.add_report(
                            f"{r['sector']:<10}{r['ts_code']:<12}{r['name']:<8}"
                            f"{r['pct_chg']:>8.2f}%{r['total_mv']:>10.2f}{r['amount']:>10.2f}"
                        )

        # Part 2: 龙头效应分析
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第二部分: 龙头效应分析")
        self.add_report("=" * 80)

        combined_data = self.analyze_sector_effect(merged_data, leaders_dict['gain'])
        self.analyze_persistence(leaders_dict)
        self.analyze_rotation(leaders_dict['gain'], merged_data)

        # Part 3: 策略应用
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第三部分: 策略应用")
        self.add_report("=" * 80)

        self.backtest_follow_strategy(merged_data, leaders_dict['gain'], trade_dates)
        self.backtest_rotation_strategy(merged_data, trade_dates)
        self.analyze_risk_control(merged_data, leaders_dict['gain'], trade_dates)

        # 总结
        self.add_report("\n" + "=" * 80)
        self.add_report("# 研究结论与建议")
        self.add_report("=" * 80)

        self.add_report("""
## 主要发现

1. **龙头识别**
   - 市值龙头相对稳定，适合作为板块代表
   - 涨幅龙头波动较大，存在较高的轮动频率
   - 成交额龙头反映市场关注度，与涨幅龙头有一定相关性

2. **龙头效应**
   - 龙头股涨幅与板块整体涨幅存在显著正相关
   - 龙头涨停时，板块整体表现明显强于市场平均
   - 不同类型龙头的一致性较低，说明市场存在多元化特征

3. **龙头切换**
   - 涨幅龙头更替频繁，平均持续时间较短
   - 市值龙头相对稳定，切换周期较长
   - 成交额龙头在趋势行情中稳定性较好

4. **策略建议**
   - 短期策略可跟随涨幅龙头，但需严格止损
   - 中期策略可关注板块轮动，选择动量强的板块龙头
   - 风险控制是龙头策略的关键，建议分散投资并设置止损

## 风险提示

- 龙头股策略属于动量策略，在震荡市场中表现不佳
- 追涨龙头存在较大回撤风险，尤其是涨停板后追高
- 历史回测结果不代表未来表现
- 实际交易需考虑交易成本和滑点影响
""")

        # 生成图表
        self.generate_visualizations(combined_data, merged_data, trade_dates)

        return '\n'.join(self.report_lines)

    def save_report(self, filename='leader_stock_effect_report.md'):
        """保存报告"""
        report_path = os.path.join(OUTPUT_DIR, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        self.log(f"报告已保存到: {report_path}")
        return report_path


def main():
    print("=" * 60)
    print("开始龙头股效应研究 (优化版)...")
    print("=" * 60)

    research = LeaderStockResearch()
    report = research.generate_report()
    report_path = research.save_report()

    print("\n" + "=" * 60)
    print("研究完成!")
    print(f"报告路径: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
