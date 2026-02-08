#!/usr/bin/env python3
"""
数据完整性检测脚本

功能：
1. 检查所有表的记录数和日期范围
2. 检测时间序列表的缺失日期
3. 对比交易日历验证完整性
4. 生成详细报告

用法：
    python scripts/check_data_integrity.py [--verbose] [--table TABLE_NAME]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import duckdb

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tushare_db.duckdb_manager import TABLE_PRIMARY_KEYS


# 表分类配置
# start_date: Tushare 官方数据最早可用日期（用于完整性检查）
TABLE_CONFIG = {
    # 基础信息表（无时间序列）
    'static': {
        'trade_cal': {'description': '交易日历', 'date_col': 'cal_date'},
        'stock_basic': {'description': '股票列表'},
        'stock_company': {'description': '上市公司信息'},
        'index_basic': {'description': '指数列表'},
        'index_classify': {'description': '申万行业分类'},
        'ths_index': {'description': '同花顺板块列表'},
        'hs_const': {'description': '沪深港通成分'},
    },
    # 日频时间序列表（start_date 为 Tushare 官方最早日期）
    'daily': {
        'daily': {'description': '日线行情', 'date_col': 'trade_date', 'start_date': '19910404'},
        'adj_factor': {'description': '复权因子', 'date_col': 'trade_date', 'start_date': '19910403'},
        'daily_basic': {'description': '每日指标', 'date_col': 'trade_date', 'start_date': '19910404'},
        'index_daily': {'description': '指数日线', 'date_col': 'trade_date', 'start_date': '19901219'},
        'moneyflow': {'description': '个股资金流', 'date_col': 'trade_date', 'start_date': '20100104'},
        'moneyflow_dc': {'description': '资金流(DC)', 'date_col': 'trade_date', 'start_date': '20230911'},
        'moneyflow_ind_dc': {'description': '行业资金流', 'date_col': 'trade_date', 'start_date': '20230912'},
        'cyq_perf': {'description': '筹码分布', 'date_col': 'trade_date', 'start_date': '20180102'},
        'stk_factor_pro': {'description': '技术因子', 'date_col': 'trade_date', 'start_date': '20050104'},
        'dc_index': {'description': '龙虎榜指数', 'date_col': 'trade_date', 'start_date': '20241220'},
        'dc_member': {'description': '龙虎榜明细', 'date_col': 'trade_date', 'start_date': '20241220'},
        'limit_list_d': {'description': '涨跌停统计', 'date_col': 'trade_date', 'start_date': '20200102'},
        'margin_detail': {'description': '融资融券', 'date_col': 'trade_date', 'start_date': '20100331'},
        'sw_daily': {'description': '申万指数日线', 'date_col': 'trade_date', 'start_date': '20210104'},
        'ths_daily': {'description': '同花顺板块日线', 'date_col': 'trade_date', 'start_date': '20180102'},
        'kpl_concept': {'description': '开盘啦题材', 'date_col': 'trade_date', 'start_date': '20241014'},
        'kpl_concept_cons': {'description': '开盘啦成分', 'date_col': 'trade_date', 'start_date': '20251001'},
        'index_weight': {'description': '指数权重', 'date_col': 'trade_date', 'start_date': '20050930', 'monthly': True},
    },
    # 成分股映射表（周期性更新）
    'membership': {
        'index_member_all': {'description': '申万成分股', 'date_col': 'in_date'},
        'ths_member': {'description': '同花顺成分股'},
    },
    # 财务数据表（季度更新）
    'financial': {
        'fina_indicator_vip': {'description': '财务指标', 'date_col': 'end_date', 'start_date': '19901231'},
        'income': {'description': '利润表', 'date_col': 'end_date', 'start_date': '19941231'},
        'balancesheet': {'description': '资产负债表', 'date_col': 'end_date', 'start_date': '19891231'},
        'cashflow': {'description': '现金流量表', 'date_col': 'end_date', 'start_date': '19980331'},
        'dividend': {'description': '分红送股', 'date_col': 'end_date', 'start_date': '19901231'},
    },
}


class DataIntegrityChecker:
    """数据完整性检查器"""

    def __init__(self, db_path: str = 'tushare.db'):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.today = datetime.now().strftime('%Y%m%d')
        self.issues = []
        self.stats = {}

    def close(self):
        self.conn.close()

    def get_trade_dates(self, start_date: str, end_date: str) -> set:
        """获取交易日列表"""
        result = self.conn.execute("""
            SELECT cal_date FROM trade_cal
            WHERE exchange = 'SSE'
              AND is_open = 1
              AND cal_date >= ?
              AND cal_date <= ?
            ORDER BY cal_date
        """, [start_date, end_date]).fetchall()
        return {row[0] for row in result}

    def get_latest_trade_date(self) -> str:
        """获取最新交易日"""
        result = self.conn.execute("""
            SELECT MAX(cal_date) FROM trade_cal
            WHERE exchange = 'SSE' AND is_open = 1 AND cal_date <= ?
        """, [self.today]).fetchone()
        return result[0] if result else self.today

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        result = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = ?
        """, [table_name]).fetchone()
        return result[0] > 0

    def get_table_count(self, table_name: str) -> int:
        """获取表记录数"""
        if not self.table_exists(table_name):
            return -1
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]

    def get_date_range(self, table_name: str, date_col: str) -> tuple:
        """获取表的日期范围"""
        result = self.conn.execute(f"""
            SELECT MIN({date_col}), MAX({date_col}) FROM {table_name}
        """).fetchone()
        return result[0], result[1]

    def get_unique_dates(self, table_name: str, date_col: str) -> set:
        """获取表中的所有唯一日期"""
        result = self.conn.execute(f"""
            SELECT DISTINCT {date_col} FROM {table_name} ORDER BY {date_col}
        """).fetchall()
        return {row[0] for row in result}

    def check_static_table(self, table_name: str, config: dict) -> dict:
        """检查静态表"""
        result = {
            'table': table_name,
            'description': config.get('description', ''),
            'type': 'static',
            'status': 'OK',
            'issues': []
        }

        if not self.table_exists(table_name):
            result['status'] = 'MISSING'
            result['count'] = 0
            result['issues'].append('表不存在')
            return result

        count = self.get_table_count(table_name)
        result['count'] = count

        if count == 0:
            result['status'] = 'EMPTY'
            result['issues'].append('表为空')
        elif count < 10:
            result['status'] = 'WARNING'
            result['issues'].append(f'记录数过少 ({count})')

        # 特殊检查：交易日历
        if table_name == 'trade_cal':
            date_col = config.get('date_col', 'cal_date')
            min_date, max_date = self.get_date_range(table_name, date_col)
            result['date_range'] = f"{min_date} ~ {max_date}"

            # 检查是否包含未来日期
            if max_date < self.today:
                result['status'] = 'WARNING'
                result['issues'].append(f'交易日历未包含今日 ({self.today})')

        return result

    def check_daily_table(self, table_name: str, config: dict, verbose: bool = False) -> dict:
        """检查日频时间序列表"""
        result = {
            'table': table_name,
            'description': config.get('description', ''),
            'type': 'daily',
            'status': 'OK',
            'issues': []
        }

        if not self.table_exists(table_name):
            result['status'] = 'MISSING'
            result['count'] = 0
            result['issues'].append('表不存在')
            return result

        date_col = config.get('date_col', 'trade_date')
        official_start = config.get('start_date', '19900101')  # Tushare 官方最早日期
        is_monthly = config.get('monthly', False)  # 是否为月度数据

        # 基础统计
        count = self.get_table_count(table_name)
        result['count'] = count

        if count == 0:
            result['status'] = 'EMPTY'
            result['issues'].append('表为空')
            return result

        # 日期范围
        min_date, max_date = self.get_date_range(table_name, date_col)
        result['date_range'] = f"{min_date} ~ {max_date}"
        result['min_date'] = min_date
        result['max_date'] = max_date
        result['official_start'] = official_start

        # 检查数据滞后
        latest_trade_date = self.get_latest_trade_date()
        if max_date < latest_trade_date:
            days_behind = len(self.get_trade_dates(max_date, latest_trade_date)) - 1
            if days_behind > 0:
                result['days_behind'] = days_behind
                # 月度数据允许更长的滞后
                stale_threshold = 30 if is_monthly else 5
                if days_behind > stale_threshold:
                    result['status'] = 'STALE'
                    result['issues'].append(f'数据滞后 {days_behind} 个交易日')
                else:
                    result['status'] = 'WARNING'
                    result['issues'].append(f'数据滞后 {days_behind} 个交易日')

        # 检查缺失日期（月度数据跳过此检查）
        if is_monthly:
            result['unique_dates'] = len(self.get_unique_dates(table_name, date_col))
            return result

        table_dates = self.get_unique_dates(table_name, date_col)
        result['unique_dates'] = len(table_dates)

        # 获取期望的交易日：从官方最早日期或表中最早日期开始（取较晚者）
        check_start = max(min_date, official_start)
        expected_dates = self.get_trade_dates(check_start, max_date)
        missing_dates = expected_dates - table_dates

        if missing_dates:
            result['missing_count'] = len(missing_dates)
            missing_sorted = sorted(missing_dates)

            if len(missing_dates) > 10:
                result['status'] = 'ERROR'
                result['issues'].append(f'缺失 {len(missing_dates)} 个交易日')
            elif len(missing_dates) > 0:
                result['status'] = 'WARNING'
                result['issues'].append(f'缺失 {len(missing_dates)} 个交易日')

            if verbose:
                result['missing_dates'] = missing_sorted[:20]  # 最多显示20个

        return result

    def check_membership_table(self, table_name: str, config: dict) -> dict:
        """检查成分股映射表"""
        result = {
            'table': table_name,
            'description': config.get('description', ''),
            'type': 'membership',
            'status': 'OK',
            'issues': []
        }

        if not self.table_exists(table_name):
            result['status'] = 'MISSING'
            result['count'] = 0
            result['issues'].append('表不存在')
            return result

        count = self.get_table_count(table_name)
        result['count'] = count

        if count == 0:
            result['status'] = 'EMPTY'
            result['issues'].append('表为空')
            return result

        # 统计唯一值
        if table_name == 'index_member_all':
            stats = self.conn.execute("""
                SELECT
                    COUNT(DISTINCT ts_code) as stocks,
                    COUNT(DISTINCT l1_code) as l1_sectors,
                    COUNT(DISTINCT l2_code) as l2_sectors,
                    COUNT(DISTINCT l3_code) as l3_sectors
                FROM index_member_all
            """).fetchone()
            result['unique_stocks'] = stats[0]
            result['l1_sectors'] = stats[1]
            result['l2_sectors'] = stats[2]
            result['l3_sectors'] = stats[3]

        elif table_name == 'ths_member':
            stats = self.conn.execute("""
                SELECT
                    COUNT(DISTINCT ts_code) as boards,
                    COUNT(DISTINCT con_code) as stocks
                FROM ths_member
            """).fetchone()
            result['unique_boards'] = stats[0]
            result['unique_stocks'] = stats[1]

            # 检查覆盖率
            if self.table_exists('ths_index'):
                total_boards = self.get_table_count('ths_index')
                coverage = stats[0] / total_boards * 100 if total_boards > 0 else 0
                result['coverage'] = f"{coverage:.1f}%"
                if coverage < 80:
                    result['status'] = 'WARNING'
                    result['issues'].append(f'板块覆盖率仅 {coverage:.1f}%')

        return result

    def check_financial_table(self, table_name: str, config: dict) -> dict:
        """检查财务数据表"""
        result = {
            'table': table_name,
            'description': config.get('description', ''),
            'type': 'financial',
            'status': 'OK',
            'issues': []
        }

        if not self.table_exists(table_name):
            result['status'] = 'MISSING'
            result['count'] = 0
            result['issues'].append('表不存在')
            return result

        count = self.get_table_count(table_name)
        result['count'] = count

        if count == 0:
            result['status'] = 'EMPTY'
            result['issues'].append('表为空')
            return result

        date_col = config.get('date_col', 'end_date')
        min_date, max_date = self.get_date_range(table_name, date_col)
        result['date_range'] = f"{min_date} ~ {max_date}"

        # 检查最新季度
        current_year = int(self.today[:4])
        current_month = int(self.today[4:6])

        # 确定预期最新季度
        if current_month >= 11:
            expected_quarter = f"{current_year}0930"
        elif current_month >= 8:
            expected_quarter = f"{current_year}0630"
        elif current_month >= 5:
            expected_quarter = f"{current_year}0331"
        else:
            expected_quarter = f"{current_year - 1}1231"

        if max_date < expected_quarter:
            result['status'] = 'WARNING'
            result['issues'].append(f'最新数据为 {max_date}，可能需要更新')

        return result

    def run_full_check(self, verbose: bool = False, specific_table: Optional[str] = None) -> dict:
        """运行完整检查"""
        results = {
            'check_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database': self.db_path,
            'latest_trade_date': self.get_latest_trade_date(),
            'tables': {},
            'summary': {
                'total': 0,
                'ok': 0,
                'warning': 0,
                'error': 0,
                'missing': 0,
                'empty': 0,
                'stale': 0
            }
        }

        all_tables = {}
        for category, tables in TABLE_CONFIG.items():
            for table_name, config in tables.items():
                all_tables[table_name] = (category, config)

        # 如果指定了表名，只检查该表
        if specific_table:
            if specific_table in all_tables:
                all_tables = {specific_table: all_tables[specific_table]}
            else:
                print(f"警告: 表 '{specific_table}' 不在配置列表中")
                return results

        for table_name, (category, config) in all_tables.items():
            if category == 'static':
                check_result = self.check_static_table(table_name, config)
            elif category == 'daily':
                check_result = self.check_daily_table(table_name, config, verbose)
            elif category == 'membership':
                check_result = self.check_membership_table(table_name, config)
            elif category == 'financial':
                check_result = self.check_financial_table(table_name, config)
            else:
                continue

            results['tables'][table_name] = check_result
            results['summary']['total'] += 1

            status = check_result['status'].lower()
            if status in results['summary']:
                results['summary'][status] += 1

        return results

    def print_report(self, results: dict, verbose: bool = False):
        """打印检查报告"""
        print("=" * 80)
        print("数据完整性检查报告")
        print("=" * 80)
        print(f"检查时间: {results['check_time']}")
        print(f"数据库: {results['database']}")
        print(f"最新交易日: {results['latest_trade_date']}")
        print()

        # 状态图标
        status_icons = {
            'OK': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'MISSING': '🚫',
            'EMPTY': '📭',
            'STALE': '🕐'
        }

        # 按类型分组显示
        for category_name, category_label in [
            ('static', '基础信息表'),
            ('daily', '日频时间序列表'),
            ('membership', '成分股映射表'),
            ('financial', '财务数据表')
        ]:
            category_tables = [
                (name, data) for name, data in results['tables'].items()
                if data.get('type') == category_name
            ]

            if not category_tables:
                continue

            print(f"\n{'─' * 80}")
            print(f"【{category_label}】")
            print(f"{'─' * 80}")

            for table_name, data in category_tables:
                icon = status_icons.get(data['status'], '❓')
                desc = data.get('description', '')
                count = data.get('count', 0)

                # 基本信息行
                print(f"\n{icon} {table_name} ({desc})")
                print(f"   记录数: {count:,}")

                # 日期范围
                if 'date_range' in data:
                    date_info = f"   日期范围: {data['date_range']}"
                    if 'official_start' in data:
                        date_info += f"  (官方起始: {data['official_start']})"
                    print(date_info)

                # 唯一日期数
                if 'unique_dates' in data:
                    print(f"   交易日数: {data['unique_dates']}")

                # 滞后天数
                if 'days_behind' in data:
                    print(f"   滞后天数: {data['days_behind']}")

                # 缺失日期数
                if 'missing_count' in data:
                    print(f"   缺失日期: {data['missing_count']} 个")
                    if verbose and 'missing_dates' in data:
                        dates_str = ', '.join(data['missing_dates'][:10])
                        if len(data['missing_dates']) > 10:
                            dates_str += f" ... (共{len(data['missing_dates'])}个)"
                        print(f"   缺失列表: {dates_str}")

                # 成分股统计
                if 'unique_stocks' in data:
                    print(f"   唯一股票: {data['unique_stocks']:,}")
                if 'unique_boards' in data:
                    print(f"   唯一板块: {data['unique_boards']:,}")
                if 'coverage' in data:
                    print(f"   覆盖率: {data['coverage']}")

                # 问题列表
                if data.get('issues'):
                    for issue in data['issues']:
                        print(f"   ⚡ {issue}")

        # 汇总统计
        summary = results['summary']
        print(f"\n{'=' * 80}")
        print("汇总统计")
        print(f"{'=' * 80}")
        print(f"检查表总数: {summary['total']}")
        print(f"  ✅ 正常: {summary['ok']}")
        print(f"  ⚠️  警告: {summary['warning']}")
        print(f"  ❌ 错误: {summary['error']}")
        print(f"  🕐 过期: {summary['stale']}")
        print(f"  📭 空表: {summary['empty']}")
        print(f"  🚫 缺失: {summary['missing']}")

        # 需要关注的表
        problem_tables = [
            name for name, data in results['tables'].items()
            if data['status'] in ('ERROR', 'STALE', 'MISSING')
        ]

        if problem_tables:
            print(f"\n⚠️  需要关注的表: {', '.join(problem_tables)}")

        print()


def compare_with_reference(checker: DataIntegrityChecker, reference_table: str = 'daily') -> dict:
    """与参考表对比缺失情况"""
    print("\n" + "=" * 80)
    print(f"与 {reference_table} 表对比分析")
    print("=" * 80)

    if not checker.table_exists(reference_table):
        print(f"参考表 {reference_table} 不存在")
        return {}

    ref_dates = checker.get_unique_dates(reference_table, 'trade_date')
    ref_min = min(ref_dates)
    ref_max = max(ref_dates)

    print(f"参考表日期范围: {ref_min} ~ {ref_max}")
    print(f"参考表交易日数: {len(ref_dates)}")
    print()

    compare_tables = ['adj_factor', 'daily_basic', 'moneyflow', 'stk_factor_pro', 'dc_index', 'cyq_perf', 'ths_daily']

    results = {}

    for table in compare_tables:
        if not checker.table_exists(table):
            print(f"  {table}: 表不存在")
            continue

        config = TABLE_CONFIG.get('daily', {}).get(table, {})
        official_start = config.get('start_date', ref_min)  # Tushare 官方最早日期

        table_dates = checker.get_unique_dates(table, 'trade_date')
        table_min = min(table_dates) if table_dates else ref_max

        # 只比较官方数据可用的范围
        compare_start = max(ref_min, official_start, table_min)
        compare_dates = {d for d in ref_dates if d >= compare_start}

        missing = compare_dates - table_dates

        results[table] = {
            'official_start': official_start,
            'total_dates': len(table_dates),
            'missing_count': len(missing),
            'missing_dates': sorted(missing)[:10] if missing else []
        }

        status = "✅" if len(missing) == 0 else ("⚠️" if len(missing) < 10 else "❌")
        start_note = f" (官方起始: {official_start})" if official_start > ref_min else ""
        print(f"  {status} {table}: 缺失 {len(missing)} 个交易日{start_note}")

        if missing and len(missing) <= 5:
            print(f"      缺失日期: {sorted(missing)}")

    return results


def main():
    parser = argparse.ArgumentParser(description='数据完整性检测')
    parser.add_argument('--db', default='tushare.db', help='数据库路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    parser.add_argument('--table', '-t', help='只检查指定表')
    parser.add_argument('--compare', '-c', action='store_true', help='与daily表对比分析')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')

    args = parser.parse_args()

    checker = DataIntegrityChecker(args.db)

    try:
        results = checker.run_full_check(verbose=args.verbose, specific_table=args.table)

        if args.json:
            import json
            # 移除不能序列化的字段
            for table_data in results['tables'].values():
                if 'missing_dates' in table_data:
                    table_data['missing_dates'] = list(table_data['missing_dates'])
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            checker.print_report(results, verbose=args.verbose)

            if args.compare:
                compare_with_reference(checker)

    finally:
        checker.close()


if __name__ == '__main__':
    main()
