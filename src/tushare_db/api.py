# -*- coding: utf-8 -*-
"""
此文件旨在为Tushare常用接口提供一层封装。

每个函数针对一个特定的Tushare接口，具有明确的输入参数。
这些函数通过 client.py 中的 TushareDbClient 进行数据获取，
该客户端会自动处理数据库缓存和Tushare网络请求。

对于每个接口，我们都提供一个专门的类，该类将接口返回的
pandas.DataFrame 的列名定义为大写的类属性（字符串常量）。
这样做的好处是，您可以在代码中以 `StockBasic.TS_CODE` 的形式
引用列名，从而获得代码提示、避免拼写错误，并提高代码的可读性和可维护性。

接口函数本身的返回类型依然是 pandas.DataFrame，以便于您进行后续的数据分析。
"""
import pandas as pd
from typing import TYPE_CHECKING, Optional, Union, List, Dict
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from .client import TushareDBClient


class StockBasic:
    """
    `stock_basic` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    TS_CODE = "ts_code"  # TS代码
    SYMBOL = "symbol"  # 股票代码
    NAME = "name"  # 股票名称
    AREA = "area"  # 地域
    INDUSTRY = "industry"  # 所属行业
    FULLNAME = "fullname"  # 股票全称
    ENNAME = "enname"  # 英文全称
    CNSPELL = "cnspell"  # 拼音缩写
    MARKET = "market"  # 市场类型（主板/创业板/科创板/CDR）
    EXCHANGE = "exchange"  # 交易所代码
    CURR_TYPE = "curr_type"  # 交易货币
    LIST_STATUS = "list_status"  # 上市状态 L上市 D退市 P暂停上市
    LIST_DATE = "list_date"  # 上市日期
    DELIST_DATE = "delist_date"  # 退市日期
    IS_HS = "is_hs"  # 是否沪深港通标的，N否 H沪股通 S深股通
    ACT_NAME = "act_name"  # 实控人名称
    ACT_ENT_TYPE = "act_ent_type"  # 实控人企业性质


def stock_basic(
    client: 'TushareDBClient',
    ts_code: str = None,
    name: str = None,
    market: str = None,
    list_status: str = 'L',
    exchange: str = None,
    is_hs: str = None,
    fields: str = 'ts_code,symbol,name,area,industry,list_date,market,list_status'
) -> pd.DataFrame:
    """
    获取股票基础信息数据，包括股票代码、名称、上市日期等。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: TS股票代码 (e.g. '000001.SZ')
    :param name: 股票名称 (支持模糊匹配)
    :param market: 市场类别 (主板/创业板/科创板/CDR/北交所)
    :param list_status: 上市状态 L上市 D退市 P暂停上市，默认是L
    :param exchange: 交易所 SSE上交所 SZSE深交所 BSE北交所
    :param is_hs: 是否沪深港通标的，N否 H沪股通 S深股通
    :param fields: 需要返回的字段，默认包含常用字段。如果传入 `None`，则返回所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    # 构建传递给 client.query 的参数字典
    params = {
        "ts_code": ts_code,
        "name": name,
        "market": market,
        "list_status": list_status,
        "exchange": exchange,
        "is_hs": is_hs,
        "fields": fields
    }
    # 过滤掉值为 None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    return client.get_data('stock_basic', **params)


class TradeCal:
    """
    `trade_cal` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    EXCHANGE = "exchange"  # 交易所 SSE上交所 SZSE深交所
    CAL_DATE = "cal_date"  # 日历日期
    IS_OPEN = "is_open"  # 是否交易 0休市 1交易
    PRETRADE_DATE = "pretrade_date"  # 上一个交易日


def trade_cal(
    client: 'TushareDBClient',
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    is_open: str = None,
    fields: str = 'exchange,cal_date,is_open,pretrade_date'
) -> pd.DataFrame:
    """
    获取各大交易所交易日历数据,默认提取的是上交所。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param exchange: 交易所 SSE上交所,SZSE深交所,CFFEX 中金所,SHFE 上期所,CZCE 郑商所,DCE 大商所,INE 上能源
    :param start_date: 开始日期 (格式：YYYYMMDD)
    :param end_date: 结束日期 (格式：YYYYMMDD)
    :param is_open: 是否交易 '0'休市 '1'交易
    :param fields: 需要返回的字段，默认包含所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    # 验证日期格式
    def _validate_date_format(date_str: str, param_name: str) -> None:
        if date_str is not None:
            import re
            # 检查格式是否为YYYYMMDD
            if not re.match(r'^\d{8}$', date_str):
                raise ValueError(f"{param_name} must be in YYYYMMDD format, got: {date_str}")
            # 检查日期是否有效
            try:
                datetime.strptime(date_str, '%Y%m%d')
            except ValueError as e:
                raise ValueError(f"Invalid date format for {param_name}: {date_str}. {str(e)}")
    
    # 验证输入的日期格式
    _validate_date_format(start_date, "start_date")
    _validate_date_format(end_date, "end_date")
    
    # 构建传递给 client.query 的参数字典
    params = {
        "exchange": exchange,
        "start_date": start_date,
        "end_date": end_date,
        "is_open": is_open,
        "fields": fields
    }
    # 过滤掉值为 None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    df = client.get_data('trade_cal', **params)
    
    # 如果用户指定了 fields，确保返回的 DataFrame 只包含这些列
    if fields:
        requested_fields = fields.split(',')
        # 过滤掉 DataFrame 中不存在的列名
        valid_fields = [f for f in requested_fields if f in df.columns]
        return df[valid_fields]
    return df


class HsConst:
    """
    `hs_const` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    TS_CODE = "ts_code"  # TS代码
    HS_TYPE = "hs_type"  # 沪深港通类型SH沪SZ深
    IN_DATE = "in_date"  # 纳入日期
    OUT_DATE = "out_date"  # 剔除日期
    IS_NEW = "is_new"  # 是否最新 1是 0否


def hs_const(
    client: 'TushareDBClient',
    hs_type: str,
    is_new: str = '1',
    fields: str = 'ts_code,hs_type,in_date,out_date,is_new'
) -> pd.DataFrame:
    """
    获取沪股通、深股通成分数据。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param hs_type: 类型SH沪股通SZ深股通
    :param is_new: 是否最新 1 是 0 否 (默认1)
    :param fields: 需要返回的字段，默认包含所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    # 构建传递给 client.query 的参数字典
    params = {
        "hs_type": hs_type,
        "is_new": is_new,
        "fields": fields
    }
    # 过滤掉值为 None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    return client.get_data('hs_const', **params)


class StockCompany:
    """
    `stock_company` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    TS_CODE = "ts_code"  # 股票代码
    COM_NAME = "com_name"  # 公司全称
    COM_ID = "com_id"  # 统一社会信用代码
    EXCHANGE = "exchange"  # 交易所代码
    CHAIRMAN = "chairman"  # 法人代表
    MANAGER = "manager"  # 总经理
    SECRETARY = "secretary"  # 董秘
    REG_CAPITAL = "reg_capital"  # 注册资本(万元)
    SETUP_DATE = "setup_date"  # 注册日期
    PROVINCE = "province"  # 所在省份
    CITY = "city"  # 所在城市
    INTRODUCTION = "introduction"  # 公司介绍
    WEBSITE = "website"  # 公司主页
    EMAIL = "email"  # 电子邮件
    OFFICE = "office"  # 办公室
    EMPLOYEES = "employees"  # 员工人数
    MAIN_BUSINESS = "main_business"  # 主要业务及产品
    BUSINESS_SCOPE = "business_scope"  # 经营范围


def stock_company(
    client: 'TushareDBClient',
    ts_code: str = None,
    exchange: str = None,
    fields: str = 'ts_code,com_name,com_id,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,website,email,employees,main_business'
) -> pd.DataFrame:
    """
    获取上市公司基础信息。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: 股票代码
    :param exchange: 交易所代码 ，SSE上交所 SZSE深交所 BSE北交所
    :param fields: 需要返回的字段，默认不包含 `introduction`, `office`, `business_scope` 等较长字段。
                   如果传入 `None`，则返回所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    # 构建传递给 client.query 的参数字典
    params = {
        "ts_code": ts_code,
        "exchange": exchange,
        "fields": fields
    }
    # 过滤掉值为 None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    return client.get_data('stock_company', **params)


class DcMember:
    """
    `dc_member` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    TRADE_DATE = "trade_date"  # 交易日期
    TS_CODE = "ts_code"  # 板块指数代码
    CON_CODE = "con_code"  # 成分股票代码
    NAME = "name"  # 成分股名称

def dc_member(
    client: 'TushareDBClient',
    ts_code: str = None,
    con_code: str = None,
    trade_date: str = None,
    fields: str = 'trade_date,ts_code,con_code,name'
) -> pd.DataFrame:
    """
    获取东方财富板块每日成分数据。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: 板块指数代码 (e.g. 'BK1184.DC')
    :param con_code: 成分股票代码 (e.g. '002117.SZ')
    :param trade_date: 交易日期 (YYYYMMDD格式)
    :param fields: 需要返回的字段，默认包含常用字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    params = {
        "ts_code": ts_code,
        "con_code": con_code,
        "trade_date": trade_date,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    return client.get_data('dc_member', **params)

class ProBar:
    """
    `pro_bar` 接口返回的 DataFrame 的列名常量。
    """
    # --- DataFrame 的列名常量 ---
    TS_CODE = "ts_code"  # 证券代码
    TRADE_DATE = "trade_date"  # 交易日期
    OPEN = "open"  # 开盘价
    HIGH = "high"  # 最高价
    LOW = "low"  # 最低价
    CLOSE = "close"  # 收盘价
    PRE_CLOSE = "pre_close"  # 昨收价
    CHANGE = "change"  # 涨跌额
    PCT_CHG = "pct_chg"  # 涨跌幅
    VOL = "vol"  # 成交量 （手）
    AMOUNT = "amount"  # 成交额 （千元）
    # 新增列名
    ADJ_FACTOR = "adj_factor" # 复权因子



class ProBarAsset:
    """
    `pro_bar` 接口的 `asset` 参数常量。
    资产类别：E股票 I沪深指数 C数字货币 FT期货 FD基金 O期权 CB可转债。
    """
    STOCK = 'E'  # 股票
    INDEX = 'I'  # 沪深指数
    CRYPTO = 'C'  # 数字货币
    FUTURES = 'FT'  # 期货
    FUND = 'FD'  # 基金
    OPTIONS = 'O'  # 期权
    CONVERTIBLE_BOND = 'CB' # 可转债


class ProBarAdj:
    """
    `pro_bar` 接口的 `adj` 参数常量。
    复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权。
    """
    NONE = None  # 未复权
    QFQ = 'qfq'  # 前复权
    HFQ = 'hfq'  # 后复权


class ProBarFreq:
    """
    `pro_bar` 接口的 `freq` 参数常量。
    数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线。
    """
    MIN1 = '1min'  # 1分钟
    MIN5 = '5min'  # 5分钟
    MIN15 = '15min'  # 15分钟
    MIN30 = '30min'  # 30分钟
    MIN60 = '60min'  # 60分钟
    DAILY = 'D'  # 日线
    WEEKLY = 'W'  # 周线
    MONTHLY = 'M'  # 月线

"""
通用行情接口，整合了股票（未复权、前复权、后复权）、指数、数字货币、ETF基金、期货、期权的行情数据。
数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

:param client: 'TushareDBClient' 实例。
:param ts_code: 证券代码。
                - 如果为 `None`（默认），则返回数据库中 `pro_bar` 表的全部数据。
                - 如果为字符串（单个代码）或字符串列表（多个代码），则通过 `client.get_data` 获取数据。
:param start_date: 开始日期 (日线格式：YYYYMMDD，提取分钟数据请用2019-09-01 09:00:00这种格式)
:param end_date: 结束日期 (日线格式：YYYYMMDD)
:param asset: 资产类别：E股票 I沪深指数 C数字货币 FT期货 FD基金 O期权 CB可转债（v1.2.39），默认E
:param adj: 复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None，目前只���持日线复权，同时复权机制是根据设定的end_date参数动态复权，采用分红再投模式，具体请参考常见问题列表里的说明，如果获取跟行情软件一致的复权行情，可以参阅股票技术因子接口。
:param freq: 数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，其中1min表示1分钟（类推1/5/15/30/60分钟） ，默认D。
:param ma: 均线，支持任意合理int数值。注：均线是动态计算，要设置一定时间范围才能获得相应的均线，比如5日均线，开始和结束日期参数跨度必须要超过5日。目前只支持单一个股票提取均线，即需要输入ts_code参数。e.g: ma_5表示5日均价，ma_v_5表示5日均量
:param factors: 股票因子（asset='E'有效）支持 tor换手率 vr量比
:param adjfactor: 复权因子，在复权数据时，如果此参数为True，返回的数据中则带复权因子，默认为False。 该功能从1.2.33版本开始生效
:return: 一个 pandas.DataFrame，包含了查询结果。
"""
def pro_bar(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    start_date: str = None,
    end_date: str = None,
    asset: str = 'E',
    adj: str = None,
    freq: str = 'D',
    ma: list = None,
    factors: list = None,
    adjfactor: bool = False,
) -> pd.DataFrame:
    # 如果 ts_code 为 None，则直接查询数据库中的全部 pro_bar 数据
    if ts_code is None:
        ts_code = stock_basic(client, list_status='L')['ts_code'].unique().tolist()
        # return client.duckdb_manager.execute_query("SELECT * FROM pro_bar")

    # 构建传递给 client.query 的参数字典
    params = {
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": end_date,
        "asset": asset,
        "adj": adj,
        "freq": freq,
        "ma": ma,
        "factors": factors,
        "adjfactor": adjfactor,
    }
    # 过滤掉值为 None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    return client.get_data('pro_bar', **params)


class DcIndex:
    """
    `dc_index` 接口返回的 DataFrame 的列名常量。
    """
    TS_CODE = "ts_code"  # 概念代码
    TRADE_DATE = "trade_date"  # 交易日期
    NAME = "name"  # 概念名称
    LEADING = "leading"  # 领涨股票名称
    LEADING_CODE = "leading_code"  # 领涨股票代码
    PCT_CHANGE = "pct_change"  # 涨跌幅
    LEADING_PCT = "leading_pct"  # 领涨股票涨跌幅
    TOTAL_MV = "total_mv"  # 总市值（万元）
    TURNOVER_RATE = "turnover_rate"  # 换手率
    UP_NUM = "up_num"  # 上涨家数
    DOWN_NUM = "down_num"  # 下降家数


"""
获取东方财富每个交易日的概念板块数据，支持按日期查询。
:param client: 'TushareDBClient' 实例。
:param ts_code: 指数代码（支持多个代码同时输入，用逗号分隔）
:param name: 板块名称（例如：人形机器人）
:param trade_date: 交易日期（YYYYMMDD格式）
:param start_date: 开始日期
:param end_date: 结束日期
:param fields: 需要返回的字段，默认返回所有字段。
:return: 一个 pandas.DataFrame，包含了查询结果。
"""
def dc_index(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    name: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None
) -> pd.DataFrame:
    params = {
        "ts_code": ts_code,
        "name": name,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    print(f"dc_index params: {params}")
    return client.get_data('dc_index', **params)

"""
获取指定日期范围内每日最强前n个板块的成员股票列表
:param client: 'TushareDBClient' 实例
:param start_date: 开始日期 (YYYYMMDD格式)
:param end_date: 结束日期 (YYYYMMDD格式)
:param top_n: 获取每日前n个板块，默认为5
:param sort_by: 排序字段，默认为'pct_change'(涨跌幅)
:param ascending: 排序方式，False为降序(默认)，True为升序
:return: 字典，key为日期，value为包含该日前n板块及其成员股票的DataFrame
"""
def get_top_n_sector_members(
    client: 'TushareDBClient',
    start_date: str,
    end_date: str,
    top_n: int = 5,
    sort_by: str = 'pct_change',
    ascending: bool = False
) -> Dict[str, pd.DataFrame]:
    # 获取日期范围内的交易日历
    trade_dates = trade_cal(client, start_date=start_date, end_date=end_date, is_open='1')
    trade_dates = trade_dates['cal_date'].tolist()
    
    result = {}
    
    for date in reversed(trade_dates):
        # 获取当日所有板块数据并按指定字段排序
        print("get_top_n_sector_members 获取日期:", date)
        sectors = dc_index(client, trade_date=date)
        if not sectors.empty:
            # 排序并取前n个板块
            top_sectors = sectors.sort_values(by=sort_by, ascending=ascending).head(top_n)
            
            # 获取每个板块的成员股票
            members_list = []
            for _, sector in top_sectors.iterrows():
                members = dc_member(client, ts_code=sector['ts_code'], trade_date=date)
                if not members.empty:
                    members['sector_name'] = sector['name']
                    members['sector_pct_change'] = sector['pct_change']
                    members_list.append(members)
            
            if members_list:
                result[date] = pd.concat(members_list)
    
    return result


class CyqPerf:
    """
    `cyq_perf` 接口返回的 DataFrame 的列名常量。
    """
    TS_CODE = "ts_code"  # 股票代码
    TRADE_DATE = "trade_date"  # 交易日期
    HIS_LOW = "his_low"  # 历史最低价
    HIS_HIGH = "his_high"  # 历史最高价
    COST_5PCT = "cost_5pct"  # 5分位成本
    COST_15PCT = "cost_15pct"  # 15分位成本
    COST_50PCT = "cost_50pct"  # 50分位成本
    COST_85PCT = "cost_85pct"  # 85分位成本
    COST_95PCT = "cost_95pct"  # 95分位成本
    WEIGHT_AVG = "weight_avg"  # 加权平均成本
    WINNER_RATE = "winner_rate"  # 胜率


def cyq_perf(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None
) -> pd.DataFrame:
    """
    获取A股每日筹码平均成本和胜率情况。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: 股票代码 (e.g. '600000.SH')
    :param trade_date: 交易日期 (YYYYMMDD格式)
    :param start_date: 开始日期 (YYYYMMDD格式)
    :param end_date: 结束日期 (YYYYMMDD格式)
    :param fields: 需要返回的字段，默认返回所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    params = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    return client.get_data('cyq_perf', **params)

class CyqChips:
    """
    `cyq_chips` 接口返回的 DataFrame 的列名常量。
    """
    TS_CODE = "ts_code"  # 股票代码
    TRADE_DATE = "trade_date"  # 交易日期
    PRICE = "price"  # 成本价格
    PERCENT = "percent"  # 价格占比（%）


def cyq_chips(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None
) -> pd.DataFrame:
    """
    获取A股每日的筹码分布情况。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: 股票代码 (e.g. '600000.SH')
    :param trade_date: 交易日期 (YYYYMMDD格式)
    :param start_date: 开始日期 (YYYYMMDD格式)
    :param end_date: 结束日期 (YYYYMMDD格式)
    :param fields: 需要返回的字段，默认返回所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    params = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    return client.get_data('cyq_chips', **params)
