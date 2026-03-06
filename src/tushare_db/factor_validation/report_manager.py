#!/usr/bin/env python
"""
报告管理系统 - 因子质检结果的历史记录和对比分析

功能：
1. 保存每次运行的详细结果到本地数据库
2. 支持按因子、股票、日期等多维度查询
3. 历史对比分析（同一因子不同时间、不同因子同一时间）
4. 统计报表和趋势分析
5. 导出功能（CSV、Excel、Markdown）
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import os
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from src.tushare_db.factor_validation import FactorReport, FactorRegistry


@dataclass
class ValidationRecord:
    """单次验证记录"""
    id: Optional[int] = None
    timestamp: str = ""
    factor_name: str = ""
    factor_description: str = ""
    ts_code: str = ""
    stock_name: str = ""
    start_date: str = ""
    end_date: str = ""
    n_days: int = 0
    n_simulations: int = 0
    simulation_days: int = 0
    alpha_threshold: float = 1.5
    p_actual: float = 0.0
    p_random: float = 0.0
    alpha_ratio: float = 0.0
    n_signals_actual: int = 0
    n_signals_random: int = 0
    p_value: float = 0.0
    is_significant: bool = False
    recommendation: str = ""
    mu: float = 0.0
    sigma: float = 0.0
    notes: str = ""


class ReportManager:
    """报告管理器"""

    def __init__(self, db_path: str = "~/.factor_validation/reports.db"):
        """
        初始化报告管理器

        Args:
            db_path: SQLite数据库路径，默认在用户目录下
        """
        self.db_path = os.path.expanduser(db_path)
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"✓ 创建报告目录: {db_dir}")

    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    factor_description TEXT,
                    ts_code TEXT NOT NULL,
                    stock_name TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    n_days INTEGER,
                    n_simulations INTEGER,
                    simulation_days INTEGER,
                    alpha_threshold REAL,
                    p_actual REAL,
                    p_random REAL,
                    alpha_ratio REAL,
                    n_signals_actual INTEGER,
                    n_signals_random INTEGER,
                    p_value REAL,
                    is_significant INTEGER,
                    recommendation TEXT,
                    mu REAL,
                    sigma REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引以加速查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_factor_ts_date
                ON validation_records(factor_name, ts_code, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON validation_records(timestamp DESC)
            """)

            conn.commit()

    def save_report(
        self,
        report: FactorReport,
        ts_code: str,
        stock_name: str = "",
        start_date: str = "",
        end_date: str = "",
        notes: str = ""
    ) -> int:
        """
        保存因子报告到数据库

        Args:
            report: FactorReport 对象
            ts_code: 股票代码
            stock_name: 股票名称
            start_date: 数据开始日期
            end_date: 数据结束日期
            notes: 备注

        Returns:
            记录ID
        """
        # 从报告中提取第一条结果（单股票情况）
        if report.results:
            result = report.results[0]
            record = ValidationRecord(
                timestamp=report.timestamp.isoformat(),
                factor_name=report.factor.name,
                factor_description=report.factor.description,
                ts_code=ts_code,
                stock_name=stock_name,
                start_date=start_date,
                end_date=end_date,
                n_days=result.n_signals_actual,  # 近似
                n_simulations=report.parameters.get('n_simulations', 0),
                simulation_days=report.parameters.get('simulation_days', 0),
                alpha_threshold=report.parameters.get('alpha_threshold', 1.5),
                p_actual=result.p_actual,
                p_random=result.p_random,
                alpha_ratio=result.alpha_ratio,
                n_signals_actual=result.n_signals_actual,
                n_signals_random=result.n_signals_random,
                p_value=result.p_value,
                is_significant=result.is_significant,
                recommendation=result.recommendation,
                mu=0.0,  # 可以从其他地方获取
                sigma=0.0,
                notes=notes
            )
        else:
            # 使用汇总统计
            summary = report.summary
            record = ValidationRecord(
                timestamp=report.timestamp.isoformat(),
                factor_name=report.factor.name,
                factor_description=report.factor.description,
                ts_code=ts_code,
                stock_name=stock_name,
                start_date=start_date,
                end_date=end_date,
                n_simulations=report.parameters.get('n_simulations', 0),
                simulation_days=report.parameters.get('simulation_days', 0),
                alpha_threshold=report.parameters.get('alpha_threshold', 1.5),
                p_actual=summary.get('p_actual_median', 0),
                p_random=summary.get('p_random_median', 0),
                alpha_ratio=summary.get('alpha_ratio_median', 0),
                recommendation='KEEP' if summary.get('alpha_ratio_median', 0) >= 1.5 else 'DISCARD',
                notes=notes
            )

        return self._insert_record(record)

    def _insert_record(self, record: ValidationRecord) -> int:
        """插入记录到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO validation_records (
                    timestamp, factor_name, factor_description, ts_code, stock_name,
                    start_date, end_date, n_days, n_simulations, simulation_days,
                    alpha_threshold, p_actual, p_random, alpha_ratio,
                    n_signals_actual, n_signals_random, p_value, is_significant,
                    recommendation, mu, sigma, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.factor_name, record.factor_description,
                record.ts_code, record.stock_name, record.start_date, record.end_date,
                record.n_days, record.n_simulations, record.simulation_days,
                record.alpha_threshold, record.p_actual, record.p_random,
                record.alpha_ratio, record.n_signals_actual, record.n_signals_random,
                record.p_value, int(record.is_significant), record.recommendation,
                record.mu, record.sigma, record.notes
            ))
            conn.commit()
            return cursor.lastrowid

    def query_records(
        self,
        factor_name: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        recommendation: Optional[str] = None,
        is_significant: Optional[bool] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        查询历史记录

        Args:
            factor_name: 因子名称筛选
            ts_code: 股票代码筛选
            start_date: 开始日期筛选 (YYYY-MM-DD)
            end_date: 结束日期筛选 (YYYY-MM-DD)
            recommendation: 建议筛选 (KEEP/OPTIMIZE/DISCARD)
            is_significant: 是否显著筛选
            limit: 返回记录数限制

        Returns:
            DataFrame 包含查询结果
        """
        conditions = []
        params = []

        if factor_name:
            conditions.append("factor_name = ?")
            params.append(factor_name)

        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if recommendation:
            conditions.append("recommendation = ?")
            params.append(recommendation)

        if is_significant is not None:
            conditions.append("is_significant = ?")
            params.append(int(is_significant))

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        query = f"""
            SELECT * FROM validation_records
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # 转换布尔值
        if 'is_significant' in df.columns:
            df['is_significant'] = df['is_significant'].astype(bool)

        return df

    def get_factor_history(self, factor_name: str, ts_code: Optional[str] = None) -> pd.DataFrame:
        """
        获取特定因子的历史记录

        Args:
            factor_name: 因子名称
            ts_code: 可选的股票代码筛选

        Returns:
            DataFrame 按时间排序的历史记录
        """
        return self.query_records(factor_name=factor_name, ts_code=ts_code, limit=1000)

    def get_stock_history(self, ts_code: str) -> pd.DataFrame:
        """
        获取特定股票的所有因子历史记录

        Args:
            ts_code: 股票代码

        Returns:
            DataFrame 包含该股票的所有因子测试记录
        """
        return self.query_records(ts_code=ts_code, limit=1000)

    def compare_factor_over_time(
        self,
        factor_name: str,
        ts_code: str,
        metric: str = 'alpha_ratio'
    ) -> pd.DataFrame:
        """
        对比同一因子在不同时间的表现

        Args:
            factor_name: 因子名称
            ts_code: 股票代码
            metric: 对比指标 (alpha_ratio/p_actual/p_random)

        Returns:
            DataFrame 包含时间序列数据
        """
        df = self.get_factor_history(factor_name, ts_code)

        if df.empty:
            return df

        # 选择关键列
        columns = ['timestamp', 'factor_name', 'ts_code', 'start_date', 'end_date',
                   'p_actual', 'p_random', 'alpha_ratio', 'recommendation']
        df = df[columns].copy()

        # 计算变化
        if len(df) > 1:
            df[f'{metric}_change'] = df[metric].diff()
            df[f'{metric}_change_pct'] = df[metric].pct_change() * 100

        return df.sort_values('timestamp')

    def compare_factors_at_time(
        self,
        timestamp: str,
        ts_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        对比同一时间不同因子的表现

        Args:
            timestamp: 时间戳
            ts_code: 可选的股票代码筛选

        Returns:
            DataFrame 按 alpha_ratio 排序
        """
        df = self.query_records(ts_code=ts_code, start_date=timestamp, end_date=timestamp)

        if df.empty:
            return df

        # 按 alpha_ratio 排序
        df = df.sort_values('alpha_ratio', ascending=False)

        return df

    def get_statistics_summary(self, days: int = 30) -> Dict:
        """
        获取最近N天的统计摘要

        Args:
            days: 最近N天

        Returns:
            统计字典
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        df = self.query_records(start_date=start_date, limit=10000)

        if df.empty:
            return {"message": "No data in the specified period"}

        stats = {
            "period_days": days,
            "total_records": len(df),
            "unique_factors": df['factor_name'].nunique(),
            "unique_stocks": df['ts_code'].nunique(),
            "keep_count": (df['recommendation'] == 'KEEP').sum(),
            "discard_count": (df['recommendation'] == 'DISCARD').sum(),
            "significant_count": df['is_significant'].sum(),
            "avg_alpha_ratio": df['alpha_ratio'].mean(),
            "median_alpha_ratio": df['alpha_ratio'].median(),
            "best_factor": df.loc[df['alpha_ratio'].idxmax(), 'factor_name'] if not df.empty else None,
            "best_alpha": df['alpha_ratio'].max(),
            "worst_factor": df.loc[df['alpha_ratio'].idxmin(), 'factor_name'] if not df.empty else None,
            "worst_alpha": df['alpha_ratio'].min(),
        }

        # 按因子统计
        factor_stats = df.groupby('factor_name').agg({
            'alpha_ratio': ['mean', 'std', 'count'],
            'recommendation': lambda x: (x == 'KEEP').mean()
        }).round(3)

        stats['factor_performance'] = factor_stats.to_dict()

        return stats

    def export_to_csv(self, filepath: str, **query_params):
        """导出查询结果到CSV"""
        df = self.query_records(**query_params)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✓ 已导出 {len(df)} 条记录到: {filepath}")
        return filepath

    def export_to_excel(self, filepath: str, **query_params):
        """导出查询结果到Excel（多个工作表）"""
        df = self.query_records(**query_params)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 主数据表
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # 因子汇总表
            if not df.empty:
                factor_summary = df.groupby('factor_name').agg({
                    'alpha_ratio': ['mean', 'std', 'min', 'max', 'count'],
                    'p_actual': 'mean',
                    'p_random': 'mean'
                }).round(4)
                factor_summary.to_excel(writer, sheet_name='Factor Summary')

                # 股票汇总表
                stock_summary = df.groupby('ts_code').agg({
                    'alpha_ratio': 'mean',
                    'factor_name': 'count'
                }).round(4)
                stock_summary.to_excel(writer, sheet_name='Stock Summary')

        print(f"✓ 已导出到: {filepath}")
        return filepath

    def generate_report(self, days: int = 30) -> str:
        """生成统计报告"""
        stats = self.get_statistics_summary(days)

        if "message" in stats:
            return stats["message"]

        report = f"""
# 因子质检报告管理 - 统计摘要

## 统计期间
最近 {stats['period_days']} 天

## 总体概况
- **总记录数**: {stats['total_records']}
- **测试因子数**: {stats['unique_factors']}
- **测试股票数**: {stats['unique_stocks']}

## 质检结果分布
- **建议保留**: {stats['keep_count']} ({stats['keep_count']/stats['total_records']*100:.1f}%)
- **建议废弃**: {stats['discard_count']} ({stats['discard_count']/stats['total_records']*100:.1f}%)
- **显著因子**: {stats['significant_count']} ({stats['significant_count']/stats['total_records']*100:.1f}%)

## Alpha Ratio 统计
- **平均值**: {stats['avg_alpha_ratio']:.3f}
- **中位数**: {stats['median_alpha_ratio']:.3f}
- **最佳因子**: {stats['best_factor']} (Alpha={stats['best_alpha']:.3f})
- **最差因子**: {stats['worst_factor']} (Alpha={stats['worst_alpha']:.3f})

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    def delete_old_records(self, days: int = 90) -> int:
        """删除N天前的旧记录"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM validation_records WHERE timestamp < ?",
                (cutoff_date,)
            )
            conn.commit()
            deleted_count = cursor.rowcount

        print(f"✓ 已删除 {deleted_count} 条 {days} 天前的记录")
        return deleted_count


# 便捷函数
def get_report_manager(db_path: Optional[str] = None) -> ReportManager:
    """获取报告管理器实例（单例模式）"""
    if db_path is None:
        db_path = os.path.expanduser("~/.factor_validation/reports.db")
    return ReportManager(db_path)


if __name__ == "__main__":
    # 演示
    print("=" * 80)
    print("报告管理系统演示")
    print("=" * 80)

    # 创建管理器
    manager = ReportManager()

    print(f"\n数据库路径: {manager.db_path}")

    # 查询最近记录
    print("\n最近10条记录:")
    df = manager.query_records(limit=10)

    if not df.empty:
        print(df[['timestamp', 'factor_name', 'ts_code', 'alpha_ratio', 'recommendation']].to_string())

        # 统计摘要
        print("\n统计摘要:")
        stats = manager.get_statistics_summary(days=30)
        for key, value in stats.items():
            if key != 'factor_performance':
                print(f"  {key}: {value}")
    else:
        print("暂无记录")
        print("\n提示: 运行因子验证后会自动保存记录到数据库")
