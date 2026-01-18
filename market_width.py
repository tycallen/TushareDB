"""
市场宽度图 - 基于 Tushare-DuckDB 数据源
显示每个行业中上涨股票的比例，帮助判断市场整体强弱
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import warnings
from tushare_db import DataReader

warnings.filterwarnings('ignore')

# 设置中文字体（按优先级尝试不同平台的字体）
def setup_chinese_font(verbose=False):
    """
    配置matplotlib中文字体

    Args:
        verbose: 是否显示字体选择信息
    """
    import matplotlib.font_manager as fm

    # 按优先级列出可能的中文字体
    font_list = [
        'PingFang SC',      # macOS 苹方（简体中文）
        'PingFang HK',      # macOS 苹方（香港）
        'PingFang TC',      # macOS 苹方（繁体中文）
        'Heiti SC',         # macOS 黑体（简体中文）
        'Heiti TC',         # macOS 黑体（繁体中文）
        'STHeiti',          # macOS 华文黑体
        'Songti SC',        # macOS 宋体（简体中文）
        'Kaiti SC',         # macOS 楷体（简体中文）
        'Arial Unicode MS', # macOS/Windows 通用
        'Microsoft YaHei',  # Windows 微软雅黑
        'SimHei',           # Windows 黑体
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'Noto Sans CJK SC', # Linux Noto字体
        'sans-serif'        # 默认字体
    ]

    # 获取系统中可用的字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    # 选择第一个可用的中文字体
    for font in font_list:
        if font in available_fonts or font == 'sans-serif':
            plt.rcParams['font.sans-serif'] = [font]
            if verbose:
                print(f"使用字体: {font}")
            break

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化中文字体（静默模式）
setup_chinese_font(verbose=False)


def get_stock_industry_mapping(reader: DataReader, min_stocks: int = 10):
    """
    获取股票代码与所属行业的映射关系

    Args:
        reader: DataReader 实例
        min_stocks: 行业至少包含的股票数量，少于此数量的行业将被过滤

    Returns:
        pd.Series: 股票代码到行业名称的映射
    """
    # 先统计每个行业的股票数量
    industry_count_query = f"""
        SELECT industry, COUNT(*) as cnt
        FROM stock_basic
        WHERE list_status = 'L'
          AND industry IS NOT NULL
          AND industry != ''
        GROUP BY industry
        HAVING cnt >= {min_stocks}
    """
    valid_industries = reader.query(industry_count_query)['industry'].tolist()

    # 获取这些行业的股票映射
    placeholders = ','.join([f"'{ind}'" for ind in valid_industries])
    query = f"""
        SELECT ts_code, industry
        FROM stock_basic
        WHERE list_status = 'L'
          AND industry IN ({placeholders})
        ORDER BY ts_code
    """
    df = reader.query(query)
    return df.set_index('ts_code')['industry']


def get_trading_days(reader: DataReader, start_date: str, end_date: str):
    """
    获取指定日期范围内的交易日列表

    Args:
        reader: DataReader 实例
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'

    Returns:
        list: 交易日列表
    """
    query = f"""
        SELECT cal_date
        FROM trade_cal
        WHERE is_open = 1
          AND cal_date >= '{start_date}'
          AND cal_date <= '{end_date}'
        ORDER BY cal_date DESC
    """
    df = reader.query(query)
    return df['cal_date'].tolist()


def calculate_industry_width(reader: DataReader, trade_date: str, stock_industry_map: pd.Series):
    """
    计算指定交易日各行业的市场宽度（上涨股票占比）

    Args:
        reader: DataReader 实例
        trade_date: 交易日期，格式 'YYYYMMDD'
        stock_industry_map: 股票到行业的映射

    Returns:
        pd.Series: 各行业的上涨股票占比（0-100）
    """
    # 获取当日所有股票的涨跌情况
    query = f"""
        SELECT ts_code, pct_chg
        FROM daily
        WHERE trade_date = '{trade_date}'
          AND pct_chg IS NOT NULL
    """
    df = reader.query(query)

    if df.empty:
        return pd.Series()

    # 添加行业信息
    df = df[df['ts_code'].isin(stock_industry_map.index)]
    df['industry'] = df['ts_code'].map(stock_industry_map)
    df = df.dropna(subset=['industry'])

    # 按行业统计上涨股票占比
    industry_stats = df.groupby('industry').agg(
        total_count=('pct_chg', 'count'),
        up_count=('pct_chg', lambda x: (x > 0).sum())
    )

    # 计算上涨占比（百分比）
    industry_stats['up_ratio'] = (industry_stats['up_count'] / industry_stats['total_count'] * 100).round(0)

    return industry_stats['up_ratio']


def get_industry_width(reader: DataReader, end_date: str, days: int, min_stocks_per_industry: int = 20):
    """
    获取多个交易日的行业市场宽度数据

    Args:
        reader: DataReader 实例
        end_date: 结束日期，格式 'YYYYMMDD' 或 datetime.date
        days: 向前统计的交易日数量
        min_stocks_per_industry: 行业至少包含的股票数量，用于过滤小行业（默认20）

    Returns:
        pd.DataFrame: 行业宽度数据，行为交易日，列为各行业及总分
    """
    # 处理日期格式
    if isinstance(end_date, datetime.date):
        end_date = end_date.strftime('%Y%m%d')

    # 计算起始日期（预留更多天数以确保有足够的交易日）
    end_dt = datetime.datetime.strptime(end_date, '%Y%m%d')
    start_dt = end_dt - datetime.timedelta(days=days * 2)
    start_date = start_dt.strftime('%Y%m%d')

    # 获取交易日列表
    trading_days = get_trading_days(reader, start_date, end_date)
    trading_days = trading_days[:days]  # 取最近的N天

    if not trading_days:
        raise ValueError(f"未找到交易日数据，日期范围：{start_date} - {end_date}")

    # 获取股票行业映射（过滤小行业）
    stock_industry_map = get_stock_industry_mapping(reader, min_stocks=min_stocks_per_industry)

    # 计算每个交易日的行业宽度
    result_data = {}

    print(f"正在计算市场宽度，共 {len(trading_days)} 个交易日...")
    for i, trade_date in enumerate(trading_days, 1):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(trading_days)}")

        industry_width = calculate_industry_width(reader, trade_date, stock_industry_map)
        result_data[trade_date] = industry_width

    # 转换为 DataFrame
    df = pd.DataFrame(result_data).T
    df.index.name = '交易日'

    # 按日期升序排列（最早的在上面）
    df = df.sort_index()

    # 按行业名称排序列
    df = df[sorted(df.columns)]

    # 确保所有列都是数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 计算总分（所有行业的总和）
    df['总分'] = df.sum(axis=1).round(0)

    # 过滤掉总分为0的异常数据（可能是数据缺失）
    df = df[df['总分'] > 0]

    # 格式化日期显示
    df.index = pd.to_datetime(df.index).strftime('%m-%d')

    print(f"完成！共 {len(df)} 个交易日，{len(df.columns)-1} 个行业")

    return df


def show_industry_width(df: pd.DataFrame, count: int = None):
    """
    可视化展示市场宽度热力图

    Args:
        df: 市场宽度数据
        count: 显示的交易日数量（从最新开始），默认显示全部
    """
    if count is not None and count < len(df):
        df = df.tail(count)

    row_count = len(df)

    # 创建图表
    fig = plt.figure(figsize=(16, max(8, row_count * 0.3)))
    grid = plt.GridSpec(1, 10)

    # 设置颜色映射
    cmap = sns.diverging_palette(200, 10, as_cmap=True)

    # 左侧热力图：各行业的宽度
    heatmap1 = fig.add_subplot(grid[:, :-1])
    heatmap1.xaxis.set_ticks_position('top')
    sns.heatmap(
        df[df.columns[:-1]],  # 不包括"总分"列
        vmin=0, vmax=100,
        annot=True, fmt=".0f",
        cmap=cmap,
        annot_kws={'size': 8},
        cbar=False,
        linewidths=0.5,
        linecolor='white'
    )
    # 旋转 x 轴标签（行业名称）以便更好地显示
    heatmap1.set_xticklabels(heatmap1.get_xticklabels(), rotation=45, ha='left', fontsize=9)

    # 右侧热力图：总分
    heatmap2 = fig.add_subplot(grid[:, -1])
    heatmap2.xaxis.set_ticks_position('top')
    max_score = (len(df.columns) - 1) * 100  # 减去"总分"列
    sns.heatmap(
        df[['总分']],
        vmin=0, vmax=max_score,
        annot=True, fmt=".0f",
        cmap=cmap,
        annot_kws={'size': 10, 'weight': 'bold'},
        linewidths=0.5,
        linecolor='white'
    )

    plt.tight_layout()
    plt.show()

    # 绘制总分趋势图
    plt.style.use({'figure.figsize': (16, 6)})
    plt.figure(figsize=(16, 6))
    plt.plot(df.index, df['总分'], linewidth=2, marker='o', markersize=4)
    plt.title('市场宽度总分趋势', fontsize=16, fontweight='bold')
    plt.xlabel('交易日', fontsize=12)
    plt.ylabel('总分', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 运行市场宽度分析
    """
    # 初始化数据读取器
    reader = DataReader()

    try:
        # 参数设置
        end_date = datetime.date.today()  # 使用今天的日期
        count_days = 100  # 统计最近100个交易日

        print(f"开始计算市场宽度...")
        print(f"结束日期: {end_date}")
        print(f"交易日数量: {count_days}")
        print("-" * 50)

        # 计算市场宽度
        market_width_df = get_industry_width(reader, end_date, count_days)

        # 显示数据统计信息
        print("\n" + "=" * 50)
        print("市场宽度数据概览:")
        print("=" * 50)
        print(f"日期范围: {market_width_df.index[0]} 至 {market_width_df.index[-1]}")
        print(f"行业数量: {len(market_width_df.columns) - 1}")
        print(f"行业列表: {', '.join(market_width_df.columns[:-1])}")
        print("\n最近5个交易日的总分:")
        print(market_width_df['总分'].tail())

        # 可视化展示
        print("\n正在生成可视化图表...")
        show_industry_width(market_width_df, count=min(50, count_days))

    finally:
        reader.close()


if __name__ == '__main__':
    main()
