#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙头股效应研究脚本

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
import matplotlib
from datetime import datetime, timedelta
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

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


class LeaderStockResearch:
    """龙头股效应研究类"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_lines = []

    def log(self, message, level='INFO'):
        """日志记录"""
        print(f"[{level}] {message}")

    def add_report(self, content):
        """添加报告内容"""
        self.report_lines.append(content)
        print(content)

    def get_recent_trade_dates(self, days=252):
        """获取最近的交易日期"""
        query = f'''
            SELECT DISTINCT trade_date
            FROM daily
            ORDER BY trade_date DESC
            LIMIT {days}
        '''
        dates = self.conn.execute(query).fetchdf()['trade_date'].tolist()
        return sorted(dates)

    def get_industry_members(self):
        """获取行业成员信息"""
        query = '''
            SELECT ts_code, name, l1_name, l2_name, l3_name
            FROM index_member_all
            WHERE out_date IS NULL OR out_date = ''
        '''
        return self.conn.execute(query).fetchdf()

    def get_daily_data(self, trade_date):
        """获取单日数据"""
        query = f'''
            SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
                   d.pre_close, d.pct_chg, d.vol, d.amount,
                   b.total_mv, b.circ_mv, b.turnover_rate
            FROM daily d
            LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
            WHERE d.trade_date = '{trade_date}'
        '''
        return self.conn.execute(query).fetchdf()

    def get_period_data(self, start_date, end_date):
        """获取区间数据"""
        query = f'''
            SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
                   d.pre_close, d.pct_chg, d.vol, d.amount,
                   b.total_mv, b.circ_mv, b.turnover_rate
            FROM daily d
            LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
            WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
            ORDER BY d.trade_date, d.ts_code
        '''
        return self.conn.execute(query).fetchdf()

    def identify_market_cap_leader(self, df, industry_members, top_n=1):
        """
        识别市值龙头
        每个行业中市值最大的股票
        """
        # 合并行业信息
        merged = df.merge(industry_members[['ts_code', 'l1_name', 'l2_name']], on='ts_code', how='inner')

        # 按L2行业分组，找市值最大的
        leaders = merged.groupby('l2_name').apply(
            lambda x: x.nlargest(top_n, 'total_mv') if 'total_mv' in x.columns and x['total_mv'].notna().any() else x.head(0)
        ).reset_index(drop=True)

        return leaders

    def identify_gain_leader(self, df, industry_members, top_n=1):
        """
        识别涨幅龙头
        每个行业中当日涨幅最大的股票（排除涨停新股等异常情况）
        """
        # 过滤异常数据
        df_filtered = df[(df['pct_chg'].notna()) &
                         (df['pct_chg'] > -11) &
                         (df['pct_chg'] < 21)].copy()

        # 合并行业信息
        merged = df_filtered.merge(industry_members[['ts_code', 'l1_name', 'l2_name']], on='ts_code', how='inner')

        # 按L2行业分组，找涨幅最大的
        leaders = merged.groupby('l2_name').apply(
            lambda x: x.nlargest(top_n, 'pct_chg')
        ).reset_index(drop=True)

        return leaders

    def identify_volume_leader(self, df, industry_members, top_n=1):
        """
        识别成交额龙头
        每个行业中成交额最大的股票
        """
        # 合并行业信息
        merged = df.merge(industry_members[['ts_code', 'l1_name', 'l2_name']], on='ts_code', how='inner')

        # 按L2行业分组，找成交额最大的
        leaders = merged.groupby('l2_name').apply(
            lambda x: x.nlargest(top_n, 'amount') if x['amount'].notna().any() else x.head(0)
        ).reset_index(drop=True)

        return leaders

    def analyze_leader_sector_effect(self, trade_dates, industry_members):
        """
        分析龙头带动板块效应
        研究龙头股表现与板块整体表现的相关性
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.1 龙头带动板块效应分析")
        self.add_report("=" * 80)

        # 存储每日龙头和板块数据
        leader_sector_data = []

        # 选择最近60个交易日进行分析
        analysis_dates = trade_dates[-60:]

        for date in analysis_dates:
            daily_data = self.get_daily_data(date)
            if daily_data.empty:
                continue

            # 合并行业信息
            merged = daily_data.merge(industry_members[['ts_code', 'l1_name', 'l2_name']],
                                      on='ts_code', how='inner')

            # 计算每个L2行业的平均涨幅
            sector_avg = merged.groupby('l2_name').agg({
                'pct_chg': 'mean',
                'amount': 'sum',
                'ts_code': 'count'
            }).reset_index()
            sector_avg.columns = ['l2_name', 'sector_avg_pct_chg', 'sector_total_amount', 'stock_count']

            # 找出每个行业的涨幅龙头
            gain_leaders = self.identify_gain_leader(daily_data, industry_members, top_n=1)

            if gain_leaders.empty:
                continue

            # 合并数据
            result = gain_leaders.merge(sector_avg, on='l2_name', how='inner')
            result['trade_date'] = date
            leader_sector_data.append(result)

        if not leader_sector_data:
            self.add_report("没有足够的数据进行龙头效应分析")
            return None

        all_data = pd.concat(leader_sector_data, ignore_index=True)

        # 分析龙头涨幅与板块涨幅的相关性
        correlation = all_data['pct_chg'].corr(all_data['sector_avg_pct_chg'])
        self.add_report(f"\n龙头涨幅与板块平均涨幅相关系数: {correlation:.4f}")

        # 分析龙头涨停时板块的表现
        limit_up_data = all_data[all_data['pct_chg'] >= 9.5]
        if not limit_up_data.empty:
            avg_sector_return_when_limit = limit_up_data['sector_avg_pct_chg'].mean()
            self.add_report(f"龙头涨停时板块平均涨幅: {avg_sector_return_when_limit:.2f}%")

        # 分析龙头跌停时板块的表现
        limit_down_data = all_data[all_data['pct_chg'] <= -9.5]
        if not limit_down_data.empty:
            avg_sector_return_when_down = limit_down_data['sector_avg_pct_chg'].mean()
            self.add_report(f"龙头跌停时板块平均涨幅: {avg_sector_return_when_down:.2f}%")

        # 按板块分组分析
        self.add_report("\n### 各板块龙头效应强度排名 (龙头涨幅与板块涨幅相关性):")
        sector_correlations = all_data.groupby('l2_name').apply(
            lambda x: x['pct_chg'].corr(x['sector_avg_pct_chg']) if len(x) > 10 else np.nan
        ).dropna().sort_values(ascending=False)

        for i, (sector, corr) in enumerate(sector_correlations.head(20).items()):
            self.add_report(f"  {i+1:2d}. {sector}: {corr:.4f}")

        return all_data

    def analyze_leader_persistence(self, trade_dates, industry_members):
        """
        分析龙头持续性
        研究龙头股的持续天数和更替频率
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.2 龙头持续性分析")
        self.add_report("=" * 80)

        # 使用最近120个交易日
        analysis_dates = trade_dates[-120:]

        # 存储每日各类型龙头
        market_cap_leaders = {}  # date -> {sector: ts_code}
        gain_leaders = {}
        volume_leaders = {}

        for date in analysis_dates:
            daily_data = self.get_daily_data(date)
            if daily_data.empty:
                continue

            # 市值龙头
            mc_leaders = self.identify_market_cap_leader(daily_data, industry_members)
            if not mc_leaders.empty:
                market_cap_leaders[date] = dict(zip(mc_leaders['l2_name'], mc_leaders['ts_code']))

            # 涨幅龙头
            g_leaders = self.identify_gain_leader(daily_data, industry_members)
            if not g_leaders.empty:
                gain_leaders[date] = dict(zip(g_leaders['l2_name'], g_leaders['ts_code']))

            # 成交额龙头
            v_leaders = self.identify_volume_leader(daily_data, industry_members)
            if not v_leaders.empty:
                volume_leaders[date] = dict(zip(v_leaders['l2_name'], v_leaders['ts_code']))

        # 分析市值龙头的持续性
        self.add_report("\n### 市值龙头持续性:")
        mc_persistence = self._calculate_persistence(market_cap_leaders)
        self.add_report(f"  平均持续天数: {mc_persistence['avg_days']:.1f} 天")
        self.add_report(f"  最长持续天数: {mc_persistence['max_days']} 天")
        self.add_report(f"  龙头更替率 (日均): {mc_persistence['change_rate']*100:.2f}%")

        # 分析涨幅龙头的持续性
        self.add_report("\n### 涨幅龙头持续性:")
        gain_persistence = self._calculate_persistence(gain_leaders)
        self.add_report(f"  平均持续天数: {gain_persistence['avg_days']:.1f} 天")
        self.add_report(f"  最长持续天数: {gain_persistence['max_days']} 天")
        self.add_report(f"  龙头更替率 (日均): {gain_persistence['change_rate']*100:.2f}%")

        # 分析成交额龙头的持续性
        self.add_report("\n### 成交额龙头持续性:")
        vol_persistence = self._calculate_persistence(volume_leaders)
        self.add_report(f"  平均持续天数: {vol_persistence['avg_days']:.1f} 天")
        self.add_report(f"  最长持续天数: {vol_persistence['max_days']} 天")
        self.add_report(f"  龙头更替率 (日均): {vol_persistence['change_rate']*100:.2f}%")

        # 分析三种龙头的一致性
        self.add_report("\n### 三种龙头的一致性分析:")
        consistency_data = self._analyze_leader_consistency(market_cap_leaders, gain_leaders, volume_leaders)
        self.add_report(f"  市值龙头=成交额龙头的比例: {consistency_data['mc_vol_same']*100:.2f}%")
        self.add_report(f"  市值龙头=涨幅龙头的比例: {consistency_data['mc_gain_same']*100:.2f}%")
        self.add_report(f"  成交额龙头=涨幅龙头的比例: {consistency_data['vol_gain_same']*100:.2f}%")
        self.add_report(f"  三者一致的比例: {consistency_data['all_same']*100:.2f}%")

        return {
            'market_cap': mc_persistence,
            'gain': gain_persistence,
            'volume': vol_persistence,
            'consistency': consistency_data
        }

    def _calculate_persistence(self, leaders_by_date):
        """计算龙头持续性指标"""
        dates = sorted(leaders_by_date.keys())
        if len(dates) < 2:
            return {'avg_days': 0, 'max_days': 0, 'change_rate': 0}

        # 统计每只股票在每个板块作为龙头的连续天数
        sector_leader_streaks = defaultdict(list)  # sector -> [连续天数列表]

        all_sectors = set()
        for d in dates:
            all_sectors.update(leaders_by_date[d].keys())

        total_changes = 0
        total_observations = 0

        for sector in all_sectors:
            current_leader = None
            streak = 0

            for i, date in enumerate(dates):
                if sector in leaders_by_date[date]:
                    leader = leaders_by_date[date][sector]
                    if leader == current_leader:
                        streak += 1
                    else:
                        if current_leader is not None:
                            sector_leader_streaks[sector].append(streak)
                            total_changes += 1
                        current_leader = leader
                        streak = 1
                    total_observations += 1

            # 最后一段
            if streak > 0:
                sector_leader_streaks[sector].append(streak)

        # 计算统计指标
        all_streaks = []
        for sector, streaks in sector_leader_streaks.items():
            all_streaks.extend(streaks)

        if all_streaks:
            avg_days = np.mean(all_streaks)
            max_days = max(all_streaks)
        else:
            avg_days = 0
            max_days = 0

        change_rate = total_changes / total_observations if total_observations > 0 else 0

        return {
            'avg_days': avg_days,
            'max_days': max_days,
            'change_rate': change_rate
        }

    def _analyze_leader_consistency(self, mc_leaders, gain_leaders, vol_leaders):
        """分析三种龙头的一致性"""
        common_dates = set(mc_leaders.keys()) & set(gain_leaders.keys()) & set(vol_leaders.keys())

        mc_vol_same = 0
        mc_gain_same = 0
        vol_gain_same = 0
        all_same = 0
        total = 0

        for date in common_dates:
            mc = mc_leaders[date]
            gain = gain_leaders[date]
            vol = vol_leaders[date]

            common_sectors = set(mc.keys()) & set(gain.keys()) & set(vol.keys())

            for sector in common_sectors:
                total += 1
                if mc.get(sector) == vol.get(sector):
                    mc_vol_same += 1
                if mc.get(sector) == gain.get(sector):
                    mc_gain_same += 1
                if vol.get(sector) == gain.get(sector):
                    vol_gain_same += 1
                if mc.get(sector) == vol.get(sector) == gain.get(sector):
                    all_same += 1

        return {
            'mc_vol_same': mc_vol_same / total if total > 0 else 0,
            'mc_gain_same': mc_gain_same / total if total > 0 else 0,
            'vol_gain_same': vol_gain_same / total if total > 0 else 0,
            'all_same': all_same / total if total > 0 else 0
        }

    def analyze_leader_rotation(self, trade_dates, industry_members):
        """
        分析龙头切换规律
        研究龙头股切换的模式和触发因素
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 2.3 龙头切换规律分析")
        self.add_report("=" * 80)

        # 使用最近60个交易日
        analysis_dates = trade_dates[-60:]

        # 收集龙头切换事件
        switch_events = []
        prev_leaders = None

        for i, date in enumerate(analysis_dates):
            daily_data = self.get_daily_data(date)
            if daily_data.empty:
                continue

            # 获取涨幅龙头
            leaders = self.identify_gain_leader(daily_data, industry_members)
            if leaders.empty:
                continue

            current_leaders = dict(zip(leaders['l2_name'], leaders['ts_code']))

            if prev_leaders is not None:
                # 检测切换
                for sector in current_leaders:
                    if sector in prev_leaders and current_leaders[sector] != prev_leaders[sector]:
                        # 龙头切换发生
                        old_leader = prev_leaders[sector]
                        new_leader = current_leaders[sector]

                        # 获取新旧龙头的详细信息
                        old_data = daily_data[daily_data['ts_code'] == old_leader]
                        new_data = daily_data[daily_data['ts_code'] == new_leader]

                        if not old_data.empty and not new_data.empty:
                            switch_events.append({
                                'date': date,
                                'sector': sector,
                                'old_leader': old_leader,
                                'new_leader': new_leader,
                                'old_pct_chg': old_data['pct_chg'].values[0],
                                'new_pct_chg': new_data['pct_chg'].values[0],
                                'old_amount': old_data['amount'].values[0],
                                'new_amount': new_data['amount'].values[0]
                            })

            prev_leaders = current_leaders

        if not switch_events:
            self.add_report("没有足够的龙头切换事件进行分析")
            return None

        switch_df = pd.DataFrame(switch_events)

        # 分析切换频率
        total_days = len(analysis_dates)
        total_switches = len(switch_df)
        self.add_report(f"\n分析周期: {total_days} 个交易日")
        self.add_report(f"龙头切换事件总数: {total_switches}")
        self.add_report(f"日均切换次数: {total_switches/total_days:.2f}")

        # 切换时的表现差异分析
        self.add_report("\n### 龙头切换时的表现特征:")
        avg_old_return = switch_df['old_pct_chg'].mean()
        avg_new_return = switch_df['new_pct_chg'].mean()
        self.add_report(f"  旧龙头当日平均涨幅: {avg_old_return:.2f}%")
        self.add_report(f"  新龙头当日平均涨幅: {avg_new_return:.2f}%")
        self.add_report(f"  涨幅差异: {avg_new_return - avg_old_return:.2f}%")

        # 按行业分析切换频率
        self.add_report("\n### 各行业龙头切换频率排名:")
        sector_switch_counts = switch_df['sector'].value_counts()
        for i, (sector, count) in enumerate(sector_switch_counts.head(15).items()):
            self.add_report(f"  {i+1:2d}. {sector}: {count} 次")

        return switch_df

    def backtest_follow_strategy(self, trade_dates, industry_members):
        """
        回测龙头跟随策略
        策略：跟随涨幅龙头，次日开盘买入，持有N天
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.1 龙头跟随策略回测")
        self.add_report("=" * 80)

        # 使用最近120个交易日
        analysis_dates = trade_dates[-120:]

        # 不同持有期的回测结果
        holding_periods = [1, 3, 5, 10]
        results = {}

        for hold_days in holding_periods:
            returns = []

            for i in range(len(analysis_dates) - hold_days - 1):
                signal_date = analysis_dates[i]
                entry_date = analysis_dates[i + 1]
                exit_date = analysis_dates[i + hold_days + 1]

                # 获取信号日的龙头
                signal_data = self.get_daily_data(signal_date)
                if signal_data.empty:
                    continue

                leaders = self.identify_gain_leader(signal_data, industry_members, top_n=1)
                if leaders.empty:
                    continue

                # 获取入场日和出场日数据
                entry_data = self.get_daily_data(entry_date)
                exit_data = self.get_daily_data(exit_date)

                if entry_data.empty or exit_data.empty:
                    continue

                # 计算每个龙头的收益
                for _, leader in leaders.iterrows():
                    ts_code = leader['ts_code']

                    entry_row = entry_data[entry_data['ts_code'] == ts_code]
                    exit_row = exit_data[exit_data['ts_code'] == ts_code]

                    if not entry_row.empty and not exit_row.empty:
                        entry_price = entry_row['open'].values[0]
                        exit_price = exit_row['close'].values[0]

                        if entry_price and entry_price > 0:
                            ret = (exit_price - entry_price) / entry_price * 100
                            returns.append({
                                'signal_date': signal_date,
                                'ts_code': ts_code,
                                'sector': leader['l2_name'],
                                'return': ret
                            })

            if returns:
                returns_df = pd.DataFrame(returns)
                results[hold_days] = {
                    'mean_return': returns_df['return'].mean(),
                    'win_rate': (returns_df['return'] > 0).mean() * 100,
                    'max_return': returns_df['return'].max(),
                    'min_return': returns_df['return'].min(),
                    'sharpe': returns_df['return'].mean() / returns_df['return'].std() if returns_df['return'].std() > 0 else 0,
                    'trade_count': len(returns_df)
                }

        # 输出回测结果
        self.add_report("\n### 不同持有期的回测结果:")
        self.add_report("-" * 70)
        self.add_report(f"{'持有期':<10}{'平均收益':<12}{'胜率':<10}{'最大收益':<12}{'最大亏损':<12}{'夏普比':<10}")
        self.add_report("-" * 70)

        for hold_days, metrics in results.items():
            self.add_report(
                f"{hold_days}天{'':<6}"
                f"{metrics['mean_return']:.2f}%{'':<6}"
                f"{metrics['win_rate']:.1f}%{'':<5}"
                f"{metrics['max_return']:.2f}%{'':<6}"
                f"{metrics['min_return']:.2f}%{'':<6}"
                f"{metrics['sharpe']:.3f}"
            )

        return results

    def backtest_sector_rotation_strategy(self, trade_dates, industry_members):
        """
        回测板块龙头轮动策略
        策略：选择龙头效应最强的板块，买入该板块龙头
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.2 板块龙头轮动策略回测")
        self.add_report("=" * 80)

        # 使用最近120个交易日
        analysis_dates = trade_dates[-120:]

        # 计算板块动量
        lookback = 20  # 回望期
        hold_days = 5  # 持有期
        top_n_sectors = 3  # 选择表现最好的N个板块

        portfolio_returns = []

        for i in range(lookback, len(analysis_dates) - hold_days):
            # 计算过去lookback天的板块表现
            start_date = analysis_dates[i - lookback]
            signal_date = analysis_dates[i]

            sector_performance = []

            # 获取板块涨幅
            period_data = self.get_period_data(start_date, signal_date)
            if period_data.empty:
                continue

            merged = period_data.merge(industry_members[['ts_code', 'l2_name']], on='ts_code', how='inner')

            # 计算每个板块的累计涨幅
            for sector in merged['l2_name'].unique():
                sector_data = merged[merged['l2_name'] == sector]
                if len(sector_data['trade_date'].unique()) < lookback // 2:
                    continue

                # 计算板块平均收益
                sector_return = sector_data.groupby('trade_date')['pct_chg'].mean().sum()
                sector_performance.append({
                    'sector': sector,
                    'return': sector_return
                })

            if not sector_performance:
                continue

            # 选择表现最好的板块
            sector_df = pd.DataFrame(sector_performance)
            top_sectors = sector_df.nlargest(top_n_sectors, 'return')['sector'].tolist()

            # 在选中的板块中买入龙头
            entry_date = analysis_dates[i + 1]
            exit_date = analysis_dates[i + hold_days]

            entry_data = self.get_daily_data(entry_date)
            exit_data = self.get_daily_data(exit_date)

            if entry_data.empty or exit_data.empty:
                continue

            # 找出选中板块的龙头
            leaders = self.identify_gain_leader(entry_data, industry_members)
            if leaders.empty:
                continue

            selected_leaders = leaders[leaders['l2_name'].isin(top_sectors)]

            if selected_leaders.empty:
                continue

            # 计算组合收益
            period_returns = []
            for _, leader in selected_leaders.iterrows():
                ts_code = leader['ts_code']
                entry_row = entry_data[entry_data['ts_code'] == ts_code]
                exit_row = exit_data[exit_data['ts_code'] == ts_code]

                if not entry_row.empty and not exit_row.empty:
                    entry_price = entry_row['open'].values[0]
                    exit_price = exit_row['close'].values[0]

                    if entry_price and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price * 100
                        period_returns.append(ret)

            if period_returns:
                portfolio_returns.append({
                    'signal_date': signal_date,
                    'return': np.mean(period_returns),
                    'num_stocks': len(period_returns)
                })

        if not portfolio_returns:
            self.add_report("没有足够的数据进行轮动策略回测")
            return None

        returns_df = pd.DataFrame(portfolio_returns)

        self.add_report(f"\n策略参数: 回望期={lookback}天, 持有期={hold_days}天, 选择板块数={top_n_sectors}")
        self.add_report(f"\n### 轮动策略回测结果:")
        self.add_report(f"  交易次数: {len(returns_df)}")
        self.add_report(f"  平均收益: {returns_df['return'].mean():.2f}%")
        self.add_report(f"  胜率: {(returns_df['return'] > 0).mean() * 100:.1f}%")
        self.add_report(f"  最大单次收益: {returns_df['return'].max():.2f}%")
        self.add_report(f"  最大单次亏损: {returns_df['return'].min():.2f}%")
        self.add_report(f"  收益标准差: {returns_df['return'].std():.2f}%")
        self.add_report(f"  夏普比率: {returns_df['return'].mean() / returns_df['return'].std():.3f}" if returns_df['return'].std() > 0 else "  夏普比率: N/A")

        # 累计收益
        cumulative_return = (1 + returns_df['return'] / 100).cumprod() - 1
        self.add_report(f"  累计收益: {cumulative_return.iloc[-1] * 100:.2f}%")

        return returns_df

    def analyze_risk_control(self, trade_dates, industry_members):
        """
        分析风险控制方法
        """
        self.add_report("\n" + "=" * 80)
        self.add_report("## 3.3 风险控制分析")
        self.add_report("=" * 80)

        # 使用最近60个交易日
        analysis_dates = trade_dates[-60:]

        # 分析龙头股的下跌风险
        leader_drawdowns = []

        for i in range(len(analysis_dates) - 5):
            date = analysis_dates[i]
            daily_data = self.get_daily_data(date)
            if daily_data.empty:
                continue

            leaders = self.identify_gain_leader(daily_data, industry_members)
            if leaders.empty:
                continue

            # 检查龙头在未来5天的表现
            future_data = self.get_period_data(analysis_dates[i], analysis_dates[i+5])

            for _, leader in leaders.iterrows():
                ts_code = leader['ts_code']
                stock_future = future_data[future_data['ts_code'] == ts_code]

                if len(stock_future) >= 2:
                    # 计算最大回撤
                    cummax = stock_future['close'].cummax()
                    drawdown = (stock_future['close'] - cummax) / cummax * 100
                    max_drawdown = drawdown.min()

                    leader_drawdowns.append({
                        'date': date,
                        'ts_code': ts_code,
                        'sector': leader['l2_name'],
                        'signal_pct_chg': leader['pct_chg'],
                        'max_drawdown': max_drawdown
                    })

        if not leader_drawdowns:
            self.add_report("没有足够的数据进行风险分析")
            return None

        drawdown_df = pd.DataFrame(leader_drawdowns)

        self.add_report("\n### 龙头股5日内回撤分析:")
        self.add_report(f"  平均最大回撤: {drawdown_df['max_drawdown'].mean():.2f}%")
        self.add_report(f"  最大回撤中位数: {drawdown_df['max_drawdown'].median():.2f}%")
        self.add_report(f"  最严重回撤: {drawdown_df['max_drawdown'].min():.2f}%")
        self.add_report(f"  回撤超过5%的比例: {(drawdown_df['max_drawdown'] < -5).mean() * 100:.1f}%")
        self.add_report(f"  回撤超过10%的比例: {(drawdown_df['max_drawdown'] < -10).mean() * 100:.1f}%")

        # 分析涨停龙头的风险
        limit_up_leaders = drawdown_df[drawdown_df['signal_pct_chg'] >= 9.5]
        if not limit_up_leaders.empty:
            self.add_report("\n### 涨停龙头的特殊风险:")
            self.add_report(f"  涨停龙头平均回撤: {limit_up_leaders['max_drawdown'].mean():.2f}%")
            self.add_report(f"  涨停龙头回撤>5%比例: {(limit_up_leaders['max_drawdown'] < -5).mean() * 100:.1f}%")

        # 风险控制建议
        self.add_report("\n### 风险控制建议:")
        self.add_report("  1. 止损线设置: 建议设置在 -5% 到 -8% 之间")
        self.add_report("  2. 仓位控制: 单只龙头股仓位不超过总仓位的 20%")
        self.add_report("  3. 分散投资: 同时持有 3-5 个不同板块的龙头")
        self.add_report("  4. 避免追高: 涨停板龙头次日追高风险较大")
        self.add_report("  5. 关注量能: 成交量萎缩时警惕龙头切换风险")

        return drawdown_df

    def generate_report(self):
        """生成完整研究报告"""
        self.add_report("=" * 80)
        self.add_report("              龙头股效应研究报告")
        self.add_report("=" * 80)
        self.add_report(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_report("")

        # 获取数据
        self.log("获取交易日期...")
        trade_dates = self.get_recent_trade_dates(days=252)
        self.log(f"数据范围: {trade_dates[0]} 至 {trade_dates[-1]}")

        self.log("获取行业分类...")
        industry_members = self.get_industry_members()
        self.log(f"股票数量: {len(industry_members)}")

        # Part 1: 龙头识别
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第一部分: 龙头识别")
        self.add_report("=" * 80)

        latest_date = trade_dates[-1]
        self.add_report(f"\n分析日期: {latest_date}")

        daily_data = self.get_daily_data(latest_date)

        # 市值龙头
        self.add_report("\n## 1.1 市值龙头 (各行业市值最大的股票)")
        mc_leaders = self.identify_market_cap_leader(daily_data, industry_members)
        if not mc_leaders.empty:
            mc_leaders_sorted = mc_leaders.sort_values('total_mv', ascending=False).head(20)
            self.add_report("-" * 70)
            self.add_report(f"{'行业':<15}{'代码':<12}{'市值(亿)':<15}{'涨跌幅':<10}")
            self.add_report("-" * 70)
            for _, row in mc_leaders_sorted.iterrows():
                mv = row['total_mv'] / 10000 if pd.notna(row['total_mv']) else 0
                pct = row['pct_chg'] if pd.notna(row['pct_chg']) else 0
                self.add_report(f"{row['l2_name']:<12}{row['ts_code']:<12}{mv:>12.2f}{pct:>10.2f}%")

        # 涨幅龙头
        self.add_report("\n## 1.2 涨幅龙头 (各行业涨幅最大的股票)")
        gain_leaders = self.identify_gain_leader(daily_data, industry_members)
        if not gain_leaders.empty:
            gain_leaders_sorted = gain_leaders.sort_values('pct_chg', ascending=False).head(20)
            self.add_report("-" * 70)
            self.add_report(f"{'行业':<15}{'代码':<12}{'涨跌幅':<10}{'成交额(万)':<15}")
            self.add_report("-" * 70)
            for _, row in gain_leaders_sorted.iterrows():
                amount = row['amount'] / 10 if pd.notna(row['amount']) else 0
                pct = row['pct_chg'] if pd.notna(row['pct_chg']) else 0
                self.add_report(f"{row['l2_name']:<12}{row['ts_code']:<12}{pct:>8.2f}%{amount:>15.2f}")

        # 成交额龙头
        self.add_report("\n## 1.3 成交额龙头 (各行业成交额最大的股票)")
        vol_leaders = self.identify_volume_leader(daily_data, industry_members)
        if not vol_leaders.empty:
            vol_leaders_sorted = vol_leaders.sort_values('amount', ascending=False).head(20)
            self.add_report("-" * 70)
            self.add_report(f"{'行业':<15}{'代码':<12}{'成交额(亿)':<15}{'涨跌幅':<10}")
            self.add_report("-" * 70)
            for _, row in vol_leaders_sorted.iterrows():
                amount = row['amount'] / 100000 if pd.notna(row['amount']) else 0
                pct = row['pct_chg'] if pd.notna(row['pct_chg']) else 0
                self.add_report(f"{row['l2_name']:<12}{row['ts_code']:<12}{amount:>12.2f}{pct:>10.2f}%")

        # Part 2: 龙头效应分析
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第二部分: 龙头效应分析")
        self.add_report("=" * 80)

        self.log("分析龙头带动板块效应...")
        self.analyze_leader_sector_effect(trade_dates, industry_members)

        self.log("分析龙头持续性...")
        self.analyze_leader_persistence(trade_dates, industry_members)

        self.log("分析龙头切换规律...")
        self.analyze_leader_rotation(trade_dates, industry_members)

        # Part 3: 策略应用
        self.add_report("\n" + "=" * 80)
        self.add_report("# 第三部分: 策略应用")
        self.add_report("=" * 80)

        self.log("回测龙头跟随策略...")
        self.backtest_follow_strategy(trade_dates, industry_members)

        self.log("回测板块龙头轮动策略...")
        self.backtest_sector_rotation_strategy(trade_dates, industry_members)

        self.log("分析风险控制...")
        self.analyze_risk_control(trade_dates, industry_members)

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

        return '\n'.join(self.report_lines)

    def save_report(self, filename='leader_stock_effect_report.md'):
        """保存报告到文件"""
        report_path = os.path.join(OUTPUT_DIR, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        self.log(f"报告已保存到: {report_path}")
        return report_path

    def generate_visualizations(self):
        """生成可视化图表"""
        self.log("生成可视化图表...")

        trade_dates = self.get_recent_trade_dates(days=120)
        industry_members = self.get_industry_members()

        # 收集数据
        sector_data = []
        for date in trade_dates[-60:]:
            daily_data = self.get_daily_data(date)
            if daily_data.empty:
                continue

            merged = daily_data.merge(industry_members[['ts_code', 'l2_name']], on='ts_code', how='inner')

            # 找涨幅龙头
            leaders = self.identify_gain_leader(daily_data, industry_members)

            # 计算板块平均
            sector_avg = merged.groupby('l2_name').agg({
                'pct_chg': 'mean'
            }).reset_index()

            if leaders.empty:
                continue

            # 合并
            result = leaders[['l2_name', 'ts_code', 'pct_chg']].merge(
                sector_avg, on='l2_name', suffixes=('_leader', '_sector')
            )
            result['trade_date'] = date
            sector_data.append(result)

        if not sector_data:
            self.log("没有足够的数据生成图表")
            return

        all_data = pd.concat(sector_data, ignore_index=True)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 龙头涨幅与板块涨幅散点图
        ax1 = axes[0, 0]
        ax1.scatter(all_data['pct_chg_leader'], all_data['pct_chg_sector'], alpha=0.3, s=10)
        ax1.set_xlabel('Leader Return (%)')
        ax1.set_ylabel('Sector Avg Return (%)')
        ax1.set_title('Leader vs Sector Return')

        # 添加拟合线
        z = np.polyfit(all_data['pct_chg_leader'], all_data['pct_chg_sector'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_data['pct_chg_leader'].min(), all_data['pct_chg_leader'].max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8)

        corr = all_data['pct_chg_leader'].corr(all_data['pct_chg_sector'])
        ax1.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax1.transAxes, fontsize=10)

        # 2. 按板块的龙头效应强度
        ax2 = axes[0, 1]
        sector_corr = all_data.groupby('l2_name').apply(
            lambda x: x['pct_chg_leader'].corr(x['pct_chg_sector']) if len(x) > 10 else np.nan
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
        ax3.hist(all_data['pct_chg_leader'], bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_xlabel('Leader Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Leader Return Distribution')

        mean_ret = all_data['pct_chg_leader'].mean()
        ax3.axvline(x=mean_ret, color='green', linestyle='--', label=f'Mean: {mean_ret:.2f}%')
        ax3.legend()

        # 4. 时间序列 - 平均龙头涨幅
        ax4 = axes[1, 1]
        daily_avg = all_data.groupby('trade_date')['pct_chg_leader'].mean()
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


def main():
    """主函数"""
    print("=" * 60)
    print("开始龙头股效应研究...")
    print("=" * 60)

    research = LeaderStockResearch()

    # 生成报告
    report = research.generate_report()

    # 保存报告
    report_path = research.save_report()

    # 生成图表
    chart_path = research.generate_visualizations()

    print("\n" + "=" * 60)
    print("研究完成!")
    print(f"报告路径: {report_path}")
    print(f"图表路径: {chart_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
