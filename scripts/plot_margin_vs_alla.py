#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两融余额N日累计变化 vs 中证全指 双Y轴日线图

使用方法:
    python scripts/plot_margin_vs_alla.py
    python scripts/plot_margin_vs_alla.py --days 60
    python scripts/plot_margin_vs_alla.py --start-date 20240101 --end-date 20241231
    python scripts/plot_margin_vs_alla.py --output my_chart.png --no-show

环境变量:
    DB_PATH: 数据库路径（可选，默认为 tushare.db）
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataReader

# 中证全指代码
ALL_A_INDEX_CODE = '000985.SH'


# 配置中文字体
_FONT_LIST = [
    'PingFang SC', 'PingFang HK', 'PingFang TC',
    'Heiti SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'Kaiti SC',
    'Arial Unicode MS', 'Microsoft YaHei', 'SimHei',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif'
]
_available_fonts = set(f.name for f in fm.fontManager.ttflist)
for _font in _FONT_LIST:
    if _font in _available_fonts or _font == 'sans-serif':
        plt.rcParams['font.sans-serif'] = [_font]
        break
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='绘制两融余额N日累计变化 vs 中证全指双Y轴日线图'
    )
    parser.add_argument(
        '--days', type=int, default=30,
        help='过去N日（默认30）'
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='起始日期 YYYYMMDD（优先级高于 --days）'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='结束日期 YYYYMMDD（默认今天）'
    )
    parser.add_argument(
        '--db-path', type=str, default=None,
        help='数据库路径（默认从 DB_PATH 环境变量或 tushare.db）'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出图片路径（默认 reports/charts/margin_vs_alla_YYYYMMDD.png）'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='不显示图表（仅保存）'
    )
    return parser.parse_args()


def query_margin_data(reader: DataReader, start_date: str, end_date: str) -> pd.DataFrame:
    """查询两融余额汇总数据，返回按日期求和后的每日总余额"""
    if not reader.table_exists('margin'):
        raise RuntimeError(
            "margin 表不存在。请先运行 DataDownloader.download_margin() 下载数据：\n"
            "  from tushare_db import DataDownloader\n"
            "  downloader = DataDownloader()\n"
            "  downloader.download_margin(start_date='20200101', end_date='20241231')\n"
            "  downloader.close()\n"
            "或直接运行: python scripts/update_daily.py"
        )

    df = reader.get_margin(start_date=start_date, end_date=end_date)
    if df.empty:
        raise RuntimeError(f"未找到 {start_date} 至 {end_date} 的两融余额数据")

    # 按日期求和（SSE + SZSE）
    df_sum = df.groupby('trade_date')['rzrqye'].sum().reset_index()
    df_sum.columns = ['trade_date', 'total_margin']
    df_sum = df_sum.sort_values('trade_date').reset_index(drop=True)
    return df_sum


def query_alla_index_data(reader: DataReader, start_date: str, end_date: str) -> pd.DataFrame:
    """查询中证全指日线数据"""
    if not reader.table_exists('index_daily'):
        raise RuntimeError("index_daily 表不存在。请先下载指数日线数据。")

    df = reader.get_index_daily(ts_code=ALL_A_INDEX_CODE, start_date=start_date, end_date=end_date)
    if df.empty:
        raise RuntimeError(
            f"未找到 {ALL_A_INDEX_CODE} 在 {start_date} 至 {end_date} 的数据。\n"
            f"请先下载：\n"
            f"  from tushare_db import DataDownloader\n"
            f"  downloader = DataDownloader()\n"
            f"  downloader.download_index_daily(ts_code='{ALL_A_INDEX_CODE}', start_date='{start_date}', end_date='{end_date}')\n"
            f"  downloader.close()"
        )

    df = df[['trade_date', 'close']].copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def calculate_n_day_change(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """计算N日累计变化量（亿元）"""
    df = df.copy()
    # 单位从元转为亿元
    df['total_margin_yi'] = df['total_margin'] / 1e8
    # N日累计变化 = 当日 - N日前
    df['margin_change'] = df['total_margin_yi'] - df['total_margin_yi'].shift(n)
    return df


def plot_chart(df: pd.DataFrame, n: int, output_path: str = None, show: bool = True):
    """绘制双Y轴日线图"""
    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 左Y轴：两融余额N日累计变化
    color_margin = '#1f77b4'
    ax1.plot(df['trade_date'], df['margin_change'], color=color_margin, linewidth=1.5, label=f'两融余额{n}日累计变化（亿元）')
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('两融余额累计变化（亿元）', color=color_margin, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_margin)

    # 右Y轴：中证全指收盘
    ax2 = ax1.twinx()
    color_index = '#ff7f0e'
    ax2.plot(df['trade_date'], df['close'], color=color_index, linewidth=1.5, label='中证全指收盘')
    ax2.set_ylabel('中证全指收盘点位', color=color_index, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_index)

    # 参考横线（左Y轴）
    ax1.axhline(y=2000, color='red', linestyle='--', alpha=0.6, label='红线: 2000亿 / 5500')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.4, label='0轴: 0 / 4500')
    ax1.axhline(y=-1000, color='green', linestyle='--', alpha=0.6, label='绿线: -1000亿 / 4000')

    # 标题
    start_str = df['trade_date'].min().strftime('%Y-%m-%d')
    end_str = df['trade_date'].max().strftime('%Y-%m-%d')
    plt.title(f'两融余额{n}日累计变化 vs 中证全指\n({start_str} ~ {end_str})', fontsize=14)

    # 日期格式（根据数据跨度自动调整）
    days_span = (df['trade_date'].max() - df['trade_date'].min()).days
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if days_span > 730:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # >2年：每半年
    elif days_span > 365:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 1-2年：每季度
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # <1年：每月
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 每月虚线格（按月分界显示垂直虚线）
    ax1.xaxis.grid(True, which='major', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)  # 网格线置于数据下方

    # 合并图例（放在左上角）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()

    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {output_path}")

    if show:
        plt.show()


def _get_nth_trading_day_before(reader: DataReader, date_str: str, n: int) -> str:
    """使用交易日历返回 date_str 往前第 n 个交易日的日期"""
    df = reader.query(
        "SELECT cal_date FROM trade_cal WHERE is_open = 1 AND cal_date <= ? ORDER BY cal_date DESC LIMIT ?",
        [date_str, n + 1]
    )
    if len(df) < n + 1:
        return None
    return str(df.iloc[-1]['cal_date'])


def main():
    args = parse_args()

    # N日参数始终有效（指交易日）
    n = args.days

    # 结束日期
    if args.end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    else:
        end_date = args.end_date

    # 绘图显示起始日期
    if args.start_date:
        plot_start = args.start_date
    else:
        # 默认：从 end_date 往前约 2*n 个自然日，之后过滤到恰好 n 个交易日
        plot_start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=n * 2 + 10)
        plot_start = plot_start_dt.strftime('%Y%m%d')

    # 数据库路径
    db_path = args.db_path or os.getenv('DB_PATH', 'tushare.db')

    # 读取数据（先建立 reader 连接以查询交易日历）
    reader = DataReader(db_path=db_path)
    try:
        # 用交易日历精确计算：query_start = plot_start 往前第 n 个交易日
        query_start = _get_nth_trading_day_before(reader, plot_start, n)
        if query_start is None:
            raise RuntimeError(f"无法找到 {plot_start} 往前第 {n} 个交易日，数据不足")

        print(f"N日参数: {n} (交易日)")
        print(f"绘图范围: {plot_start} ~ {end_date}")
        print(f"数据查询起始: {query_start} (plot_start 往前第 {n} 个交易日)")

        margin_df = query_margin_data(reader, query_start, end_date)
        alla_df = query_alla_index_data(reader, query_start, end_date)

        # 合并
        merged = pd.merge(margin_df, alla_df, on='trade_date', how='inner')
        if merged.empty:
            raise RuntimeError("合并后数据为空，请检查日期范围是否有交集")

        # 计算N日变化
        merged = calculate_n_day_change(merged, n)

        # 只保留绘图范围内的日期（去掉前导数据）
        merged = merged[merged['trade_date'] >= plot_start].reset_index(drop=True)

        if merged.empty:
            raise RuntimeError(f"数据不足，无法计算{n}日累计变化")

        print(f"有效数据: {len(merged)} 个交易日")
        print(f"两融{n}日变化范围: {merged['margin_change'].min():.0f} ~ {merged['margin_change'].max():.0f} 亿元")
        print(f"中证全指范围: {merged['close'].min():.0f} ~ {merged['close'].max():.0f}")

        # 输出路径
        if args.output:
            output_path = args.output
        else:
            today_str = datetime.now().strftime('%Y%m%d')
            output_path = f"reports/charts/margin_vs_alla_{today_str}.png"

        # 绘图
        plot_chart(merged, n, output_path=output_path, show=not args.no_show)

    finally:
        reader.close()


if __name__ == '__main__':
    main()
