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


class IndexBasic:
    """
    `index_basic` 接口返回的 DataFrame 的列名常量。
    """
    TS_CODE = "ts_code"  # TS代码
    NAME = "name"  # 简称
    FULLNAME = "fullname"  # 指数全称
    MARKET = "market"  # 市场
    PUBLISHER = "publisher"  # 发布方
    INDEX_TYPE = "index_type"  # 指数风格
    CATEGORY = "category"  # 指数类别
    BASE_DATE = "base_date"  # 基期
    BASE_POINT = "base_point"  # 基点
    LIST_DATE = "list_date"  # 发布日期
    WEIGHT_RULE = "weight_rule"  # 加权方式
    DESC = "desc"  # 描述
    EXP_DATE = "exp_date"  # 终止日期


def index_basic(
    client: 'TushareDBClient',
    ts_code: str = None,
    name: str = None,
    market: str = None,
    publisher: str = None,
    category: str = None,
    fields: str = None
) -> pd.DataFrame:
    """
    获取指数基础信息。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: 'TushareDBClient' 实例。
    :param ts_code: 指数代码
    :param name: 指数简称
    :param market: 交易所或服务商(默认SSE)
    :param publisher: 发布商
    :param category: 指数类别
    :param fields: 需要返回的字段，默认返回所有字段。
    :return: 一个 pandas.DataFrame，包含了查询结果。
    """
    params = {
        "ts_code": ts_code,
        "name": name,
        "market": market,
        "publisher": publisher,
        "category": category,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    return client.get_data('index_basic', **params)


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
:param adj: 复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None，目前只支持日线复权，同时复权机制是根据设定的end_date参数动态复权，采用分红再投模式，具体请参考常见问题列表里的说明，如果获取跟行情软件一致的复权行情，可以参阅股票技术因子接口。
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


class StkFactorPro:
    """
    `stk_factor_pro` 接口返回的 DataFrame 的列名常量。 参考 https://tushare.pro/document/2?doc_id=328
    """
    TS_CODE = "ts_code"
    TRADE_DATE = "trade_date"
    OPEN = "open" # 开盘价
    OPEN_HFQ = "open_hfq"
    OPEN_QFQ = "open_qfq"
    HIGH = "high"
    HIGH_HFQ = "high_hfq"
    HIGH_QFQ = "high_qfq"
    LOW = "low"
    LOW_HFQ = "low_hfq"
    LOW_QFQ = "low_qfq"
    CLOSE = "close"
    CLOSE_HFQ = "close_hfq"
    CLOSE_QFQ = "close_qfq"
    PRE_CLOSE = "pre_close"
    CHANGE = "change" # 涨跌额
    PCT_CHG = "pct_chg"
    VOL = "vol"
    AMOUNT = "amount" # 成交额 （千元）
    TURNOVER_RATE = "turnover_rate"
    TURNOVER_RATE_F = "turnover_rate_f" # 换手率（自由流通股）
    VOLUME_RATIO = "volume_ratio" # 量比
    PE = "pe" # 市盈率（总市值/净利润， 亏损的PE为空）
    PE_TTM = "pe_ttm" # 市盈率（TTM，亏损的PE为空）
    PB = "pb" # 市净率（总市值/净资产）
    PS = "ps" # 市销率
    PS_TTM = "ps_ttm" # 市销率（TTM）
    DV_RATIO = "dv_ratio" # 股息率 （%）
    DV_TTM = "dv_ttm" # 股息率（TTM）
    TOTAL_SHARE = "total_share" # 总股本 （万股）
    FLOAT_SHARE = "float_share" # 流通股本 （万股）
    FREE_SHARE = "free_share" # 自由流通股本 （万股）
    TOTAL_MV = "total_mv" # 总市值 （万元）
    CIRC_MV = "circ_mv" # 流通市值（万元）
    ADJ_FACTOR = "adj_factor" # 复权因子
    ASI_BFQ = "asi_bfq" # 振动升降指标-OPEN, CLOSE, HIGH, LOW, M1=26, M2=10
    ASI_HFQ = "asi_hfq"
    ASI_QFQ = "asi_qfq"
    ASIT_BFQ = "asit_bfq" # 振动升降指标-OPEN, CLOSE, HIGH, LOW, M1=26, M2=10
    ASIT_HFQ = "asit_hfq"
    ASIT_QFQ = "asit_qfq"
    ATR_BFQ = "atr_bfq" # 真实波动N日平均值-CLOSE, HIGH, LOW, N=20
    ATR_HFQ = "atr_hfq"
    ATR_QFQ = "atr_qfq"
    BBI_BFQ = "bbi_bfq" # BBI多空指标-CLOSE, M1=3, M2=6, M3=12, M4=20
    BBI_HFQ = "bbi_hfq"
    BBI_QFQ = "bbi_qfq"
    BIAS1_BFQ = "bias1_bfq" # BIAS乖离率-CLOSE, L1=6, L2=12, L3=24
    BIAS1_HFQ = "bias1_hfq"
    BIAS1_QFQ = "bias1_qfq"
    BIAS2_BFQ = "bias2_bfq"
    BIAS2_HFQ = "bias2_hfq"
    BIAS2_QFQ = "bias2_qfq"
    BIAS3_BFQ = "bias3_bfq"
    BIAS3_HFQ = "bias3_hfq"
    BIAS3_QFQ = "bias3_qfq"
    BOLL_LOWER_BFQ = "boll_lower_bfq" # BOLL指标，布林带-CLOSE, N=20, P=2
    BOLL_LOWER_HFQ = "boll_lower_hfq"
    BOLL_LOWER_QFQ = "boll_lower_qfq"
    BOLL_MID_BFQ = "boll_mid_bfq"
    BOLL_MID_HFQ = "boll_mid_hfq"
    BOLL_MID_QFQ = "boll_mid_qfq"
    BOLL_UPPER_BFQ = "boll_upper_bfq"
    BOLL_UPPER_HFQ = "boll_upper_hfq"
    BOLL_UPPER_QFQ = "boll_upper_qfq"
    BRAR_AR_BFQ = "brar_ar_bfq" # BRAR情绪指标-OPEN, CLOSE, HIGH, LOW, M1=26
    BRAR_AR_HFQ = "brar_ar_hfq"
    BRAR_AR_QFQ = "brar_ar_qfq"
    BRAR_BR_BFQ = "brar_br_bfq"
    BRAR_BR_HFQ = "brar_br_hfq"
    BRAR_BR_QFQ = "brar_br_qfq"
    CCI_BFQ = "cci_bfq" # 顺势指标又叫CCI指标-CLOSE, HIGH, LOW, N=14
    CCI_HFQ = "cci_hfq"
    CCI_QFQ = "cci_qfq"
    CR_BFQ = "cr_bfq" # CR价格动量指标-CLOSE, HIGH, LOW, N=20
    CR_HFQ = "cr_hfq"
    CR_QFQ = "cr_qfq"
    DFMA_DIF_BFQ = "dfma_dif_bfq" # 平行线差指标-CLOSE, N1=10, N2=50, M=10
    DFMA_DIF_HFQ = "dfma_dif_hfq"
    DFMA_DIF_QFQ = "dfma_dif_qfq"
    DFMA_DIFMA_BFQ = "dfma_difma_bfq"
    DFMA_DIFMA_HFQ = "dfma_difma_hfq"
    DFMA_DIFMA_QFQ = "dfma_difma_qfq"
    DMI_ADX_BFQ = "dmi_adx_bfq" # 动向指标-CLOSE, HIGH, LOW, M1=14, M2=6
    DMI_ADX_HFQ = "dmi_adx_hfq"
    DMI_ADX_QFQ = "dmi_adx_qfq"
    DMI_ADXR_BFQ = "dmi_adxr_bfq"
    DMI_ADXR_HFQ = "dmi_adxr_hfq"
    DMI_ADXR_QFQ = "dmi_adxr_qfq"
    DMI_MDI_BFQ = "dmi_mdi_bfq"
    DMI_MDI_HFQ = "dmi_mdi_hfq"
    DMI_MDI_QFQ = "dmi_mdi_qfq"
    DMI_PDI_BFQ = "dmi_pdi_bfq"
    DMI_PDI_HFQ = "dmi_pdi_hfq"
    DMI_PDI_QFQ = "dmi_pdi_qfq"
    DOWNDAYS = "downdays"
    UPDAYS = "updays"
    DPO_BFQ = "dpo_bfq" # 区间震荡线-CLOSE, M1=20, M2=10, M3=6
    DPO_HFQ = "dpo_hfq"
    DPO_QFQ = "dpo_qfq"
    MADPO_BFQ = "madpo_bfq"
    MADPO_HFQ = "madpo_hfq"
    MADPO_QFQ = "madpo_qfq"
    EMA_BFQ_10 = "ema_bfq_10" # 指数移动平均-N=10
    EMA_BFQ_20 = "ema_bfq_20"
    EMA_BFQ_250 = "ema_bfq_250"
    EMA_BFQ_30 = "ema_bfq_30"
    EMA_BFQ_5 = "ema_bfq_5"
    EMA_BFQ_60 = "ema_bfq_60"
    EMA_BFQ_90 = "ema_bfq_90"
    EMA_HFQ_10 = "ema_hfq_10"
    EMA_HFQ_20 = "ema_hfq_20"
    EMA_HFQ_250 = "ema_hfq_250"
    EMA_HFQ_30 = "ema_hfq_30"
    EMA_HFQ_5 = "ema_hfq_5"
    EMA_HFQ_60 = "ema_hfq_60"
    EMA_HFQ_90 = "ema_hfq_90"
    EMA_QFQ_10 = "ema_qfq_10"
    EMA_QFQ_20 = "ema_qfq_20"
    EMA_QFQ_250 = "ema_qfq_250"
    EMA_QFQ_30 = "ema_qfq_30"
    EMA_QFQ_5 = "ema_qfq_5"
    EMA_QFQ_60 = "ema_qfq_60"
    EMA_QFQ_90 = "ema_qfq_90"
    EMV_BFQ = "emv_bfq" # 简易波动指标-HIGH, LOW, VOL, N=14, M=9
    EMV_HFQ = "emv_hfq"
    EMV_QFQ = "emv_qfq"
    MAEMV_BFQ = "maemv_bfq"
    MAEMV_HFQ = "maemv_hfq"
    MAEMV_QFQ = "maemv_qfq"
    EXPMA_12_BFQ = "expma_12_bfq" # EMA指数平均数指标-CLOSE, N1=12, N2=50
    EXPMA_12_HFQ = "expma_12_hfq"
    EXPMA_12_QFQ = "expma_12_qfq"
    EXPMA_50_BFQ = "expma_50_bfq"
    EXPMA_50_HFQ = "expma_50_hfq"
    EXPMA_50_QFQ = "expma_50_qfq"
    KDJ_BFQ = "kdj_bfq" # KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3
    KDJ_HFQ = "kdj_hfq"
    KDJ_QFQ = "kdj_qfq"
    KDJ_D_BFQ = "kdj_d_bfq"
    KDJ_D_HFQ = "kdj_d_hfq"
    KDJ_D_QFQ = "kdj_d_qfq"
    KDJ_K_BFQ = "kdj_k_bfq"
    KDJ_K_HFQ = "kdj_k_hfq"
    KDJ_K_QFQ = "kdj_k_qfq"
    KTN_DOWN_BFQ = "ktn_down_bfq" # 肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10
    KTN_DOWN_HFQ = "ktn_down_hfq"
    KTN_DOWN_QFQ = "ktn_down_qfq"
    KTN_MID_BFQ = "ktn_mid_bfq"
    KTN_MID_HFQ = "ktn_mid_hfq"
    KTN_MID_QFQ = "ktn_mid_qfq"
    KTN_UPPER_BFQ = "ktn_upper_bfq"
    KTN_UPPER_HFQ = "ktn_upper_hfq"
    KTN_UPPER_QFQ = "ktn_upper_qfq"
    LOWDAYS = "lowdays" # LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值
    TOPDAYS = "topdays" # TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值
    MA_BFQ_10 = "ma_bfq_10" # 简单移动平均-N=10
    MA_BFQ_20 = "ma_bfq_20"
    MA_BFQ_250 = "ma_bfq_250"
    MA_BFQ_30 = "ma_bfq_30"
    MA_BFQ_5 = "ma_bfq_5"
    MA_BFQ_60 = "ma_bfq_60"
    MA_BFQ_90 = "ma_bfq_90"
    MA_HFQ_10 = "ma_hfq_10"
    MA_HFQ_20 = "ma_hfq_20"
    MA_HFQ_250 = "ma_hfq_250"
    MA_HFQ_30 = "ma_hfq_30"
    MA_HFQ_5 = "ma_hfq_5"
    MA_HFQ_60 = "ma_hfq_60"
    MA_HFQ_90 = "ma_hfq_90"
    MA_QFQ_10 = "ma_qfq_10"
    MA_QFQ_20 = "ma_qfq_20"
    MA_QFQ_250 = "ma_qfq_250"
    MA_QFQ_30 = "ma_qfq_30"
    MA_QFQ_5 = "ma_qfq_5"
    MA_QFQ_60 = "ma_qfq_60"
    MA_QFQ_90 = "ma_qfq_90"
    MACD_BFQ = "macd_bfq" # MACD指标-CLOSE, SHORT=12, LONG=26, M=9
    MACD_HFQ = "macd_hfq"
    MACD_QFQ = "macd_qfq"
    MACD_DEA_BFQ = "macd_dea_bfq"
    MACD_DEA_HFQ = "macd_dea_hfq"
    MACD_DEA_QFQ = "macd_dea_qfq"
    MACD_DIF_BFQ = "macd_dif_bfq"
    MACD_DIF_HFQ = "macd_dif_hfq"
    MACD_DIF_QFQ = "macd_dif_qfq"
    MASS_BFQ = "mass_bfq" # 梅斯线-HIGH, LOW, N1=9, N2=25, M=6
    MASS_HFQ = "mass_hfq"
    MASS_QFQ = "mass_qfq"
    MA_MASS_BFQ = "ma_mass_bfq"
    MA_MASS_HFQ = "ma_mass_hfq"
    MA_MASS_QFQ = "ma_mass_qfq"
    MFI_BFQ = "mfi_bfq" # MFI指标是成交量的RSI指标-CLOSE, HIGH, LOW, VOL, N=14
    MFI_HFQ = "mfi_hfq"
    MFI_QFQ = "mfi_qfq"
    MTM_BFQ = "mtm_bfq" # 动量指标-CLOSE, N=12, M=6
    MTM_HFQ = "mtm_hfq"
    MTM_QFQ = "mtm_qfq"
    MTMMA_BFQ = "mtmma_bfq"
    MTMMA_HFQ = "mtmma_hfq"
    MTMMA_QFQ = "mtmma_qfq"
    OBV_BFQ = "obv_bfq" # 能量潮指标-CLOSE, VOL
    OBV_HFQ = "obv_hfq"
    OBV_QFQ = "obv_qfq"
    PSY_BFQ = "psy_bfq" # 投资者对股市涨跌产生心理波动的情绪指标-CLOSE, N=12, M=6
    PSY_HFQ = "psy_hfq"
    PSY_QFQ = "psy_qfq"
    PSYMA_BFQ = "psyma_bfq"
    PSYMA_HFQ = "psyma_hfq"
    PSYMA_QFQ = "psyma_qfq"
    ROC_BFQ = "roc_bfq" # 变动率指标-CLOSE, N=12, M=6
    ROC_HFQ = "roc_hfq"
    ROC_QFQ = "roc_qfq"
    MAROC_BFQ = "maroc_bfq"
    MAROC_HFQ = "maroc_hfq"
    MAROC_QFQ = "maroc_qfq"
    RSI_BFQ_12 = "rsi_bfq_12" # RSI指标-CLOSE, N=12
    RSI_BFQ_24 = "rsi_bfq_24"
    RSI_BFQ_6 = "rsi_bfq_6"
    RSI_HFQ_12 = "rsi_hfq_12"
    RSI_HFQ_24 = "rsi_hfq_24"
    RSI_HFQ_6 = "rsi_hfq_6"
    RSI_QFQ_12 = "rsi_qfq_12"
    RSI_QFQ_24 = "rsi_qfq_24"
    RSI_QFQ_6 = "rsi_qfq_6"
    TAQ_DOWN_BFQ = "taq_down_bfq" # 唐安奇通道(海龟)交易指标-HIGH, LOW, 20
    TAQ_DOWN_HFQ = "taq_down_hfq"
    TAQ_DOWN_QFQ = "taq_down_qfq"
    TAQ_MID_BFQ = "taq_mid_bfq"
    TAQ_MID_HFQ = "taq_mid_hfq"
    TAQ_MID_QFQ = "taq_mid_qfq"
    TAQ_UP_BFQ = "taq_up_bfq"
    TAQ_UP_HFQ = "taq_up_hfq"
    TAQ_UP_QFQ = "taq_up_qfq"
    TRIX_BFQ = "trix_bfq" # 三重指数平滑平均线-CLOSE, M1=12, M2=20
    TRIX_HFQ = "trix_hfq"
    TRIX_QFQ = "trix_qfq"
    TRMA_BFQ = "trma_bfq"
    TRMA_HFQ = "trma_hfq"
    TRMA_QFQ = "trma_qfq"
    VR_BFQ = "vr_bfq" # VR容量比率-CLOSE, VOL, M1=26
    VR_HFQ = "vr_hfq"
    VR_QFQ = "vr_qfq"
    WR_BFQ = "wr_bfq" # W&R 威廉指标-CLOSE, HIGH, LOW, N=10, N1=6
    WR_HFQ = "wr_hfq"
    WR_QFQ = "wr_qfq"
    WR1_BFQ = "wr1_bfq"
    WR1_HFQ = "wr1_hfq"
    WR1_QFQ = "wr1_qfq"
    XSII_TD1_BFQ = "xsii_td1_bfq" # 薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7
    XSII_TD1_HFQ = "xsii_td1_hfq"
    XSII_TD1_QFQ = "xsii_td1_qfq"
    XSII_TD2_BFQ = "xsii_td2_bfq"
    XSII_TD2_HFQ = "xsii_td2_hfq"
    XSII_TD2_QFQ = "xsii_td2_qfq"
    XSII_TD3_BFQ = "xsii_td3_bfq"
    XSII_TD3_HFQ = "xsii_td3_hfq"
    XSII_TD3_QFQ = "xsii_td3_qfq"
    XSII_TD4_BFQ = "xsii_td4_bfq"
    XSII_TD4_HFQ = "xsii_td4_hfq"
    XSII_TD4_QFQ = "xsii_td4_qfq"


def stk_factor_pro(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None
) -> pd.DataFrame:
    """
    获取股票每日技术面因子数据。
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
    return client.get_data('stk_factor_pro', **params)

def fina_indicator_vip(
        self,
        period: str,
        ann_date: str = None,
        start_date: str = None,
        end_date: str = None,
        **kwargs,
    ):
        """
        获取某一季度全部上市公司财务指标数据。
        此接口为VIP接口，需要较高积分。与fina_indicator不同，本接口不支持按股票代码查询，而是按季度返回所有股票的数据。

        :param period: 报告期，格式为YYYYMMDD，例如'20171231'表示年报。此参数为必需参数。
        :param ann_date: 公告日期 (可选)
        :param start_date: 报告期开始日期 (可选)
        :param end_date: 报告期结束日期 (可选)
        :return: DataFrame
        ts_code: str, TS代码
        ann_date: str, 公告日期
        end_date: str, 报告期
        eps: float, 基本每股收益
        dt_eps: float, 稀释每股收益
        total_revenue_ps: float, 每股营业总收入
        revenue_ps: float, 每股营业收入
        capital_rese_ps: float, 每股资本公积
        surplus_rese_ps: float, 每股盈余公积
        undist_profit_ps: float, 每股未分配利润
        extra_item: float, 非经常性损益
        profit_dedt: float, 扣除非经常性损益后的净利润（扣非净利润）
        gross_margin: float, 毛利
        current_ratio: float, 流动比率
        quick_ratio: float, 速动比率
        cash_ratio: float, 保守速动比率
        invturn_days: float, 存货周转天数
        arturn_days: float, 应收账款周转天数
        inv_turn: float, 存货周转率
        ar_turn: float, 应收账款周转率
        ca_turn: float, 流动资产周转率
        fa_turn: float, 固定资产周转率
        assets_turn: float, 总资产周转率
        op_income: float, 经营活动净收益
        valuechange_income: float, 价值变动净收益
        interst_income: float, 利息费用
        daa: float, 折旧与摊销
        ebit: float, 息税前利润
        ebitda: float, 息税折旧摊销前利润
        fcff: float, 企业自由现金流量
        fcfe: float, 股权自由现金流量
        current_exint: float, 无息流动负债
        noncurrent_exint: float, 无息非流动负债
        interestdebt: float, 带息债务
        netdebt: float, 净债务
        tangible_asset: float, 有形资产
        working_capital: float, 营运资金
        networking_capital: float, 营运流动资本
        invest_capital: float, 全部投入资本
        retained_earnings: float, 留存收益
        diluted2_eps: float, 期末摊薄每股收益
        bps: float, 每股净资产
        ocfps: float, 每股经营活动产生的现金流量净额
        retainedps: float, 每股留存收益
        cfps: float, 每股现金流量净额
        ebit_ps: float, 每股息税前利润
        fcff_ps: float, 每股企业自由现金流量
        fcfe_ps: float, 每股股东自由现金流量
        netprofit_margin: float, 销售净利率
        grossprofit_margin: float, 销售毛利率
        cogs_of_sales: float, 销售成本率
        expense_of_sales: float, 销售期间费用率
        profit_to_gr: float, 净利润/营业总收入
        saleexp_to_gr: float, 销售费用/营业总收入
        adminexp_of_gr: float, 管理费用/营业总收入
        finaexp_of_gr: float, 财务费用/营业总收入
        impai_ttm: float, 资产减值损失/营业总收入
        gc_of_gr: float, 营业总成本/营业总收入
        op_of_gr: float, 营业利润/营业总收入
        ebit_of_gr: float, 息税前利润/营业总收入
        roe: float, 净资产收益率
        roe_waa: float, 加权平均净资产收益率
        roe_dt: float, 净资产收益率(扣除非经常损益)
        roa: float, 总资产报酬率
        npta: float, 总资产净利润
        roic: float, 投入资本回报率
        roe_yearly: float, 年化净资产收益率
        roa2_yearly: float, 年化总资产报酬率
        roe_avg: float, 平均净资产收益率(增发条件)
        opincome_of_ebt: float, 经营活动净收益/利润总额
        investincome_of_ebt: float, 价值变动净收益/利润总额
        n_op_profit_of_ebt: float, 营业外收支净额/利润总额
        tax_to_ebt: float, 所得税/利润总额
        dtprofit_to_profit: float, 扣除非经常损益后的净利润/净利润
        salescash_to_or: float, 销售商品提供劳务收到的现金/营业收入
        ocf_to_or: float, 经营活动产生的现金流量净额/营业收入
        ocf_to_opincome: float, 经营活动产生的现金流量净额/经营活动净收益
        capitalized_to_da: float, 资本支出/折旧和摊销
        debt_to_assets: float, 资产负债率
        assets_to_eqt: float, 权益乘数
        dp_assets_to_eqt: float, 权益乘数(杜邦分析)
        ca_to_assets: float, 流动资产/总资产
        nca_to_assets: float, 非流动资产/总资产
        tbassets_to_totalassets: float, 有形资产/总资产
        int_to_talcap: float, 带息债务/全部投入资本
        eqt_to_talcapital: float, 归属于母公司的股东权益/全部投入资本
        currentdebt_to_debt: float, 流动负债/负债合计
        longdeb_to_debt: float, 非流动负债/负债合计
        ocf_to_shortdebt: float, 经营活动产生的现金流量净额/流动负债
        debt_to_eqt: float, 产权比率
        eqt_to_debt: float, 归属于母公司的股东权益/负债合计
        eqt_to_interestdebt: float, 归属于母公司的股东权益/带息债务
        tangibleasset_to_debt: float, 有形资产/负债合计
        tangasset_to_intdebt: float, 有形资产/带息债务
        tangibleasset_to_netdebt: float, 有形资产/净债务
        ocf_to_debt: float, 经营活动产生的现金流量净额/负债合计
        ocf_to_interestdebt: float, 经营活动产生的现金流量净额/带息债务
        ocf_to_netdebt: float, 经营活动产生的现金流量净额/净债务
        ebit_to_interest: float, 已获利息倍数(EBIT/利息费用)
        longdebt_to_workingcapital: float, 长期债务与营运资金比率
        ebitda_to_debt: float, 息税折旧摊销前利润/负债合计
        turn_days: float, 营业周期
        roa_yearly: float, 年化总资产净利率
        roa_dp: float, 总资产净利率(杜邦分析)
        fixed_assets: float, 固定资产合计
        profit_prefin_exp: float, 扣除财务费用前营业利润
        non_op_profit: float, 非营业利润
        op_to_ebt: float, 营业利润／利润总额
        nop_to_ebt: float, 非营业利润／利润总额
        ocf_to_profit: float, 经营活动产生的现金流量净额／营业利润
        cash_to_liqdebt: float, 货币资金／流动负债
        cash_to_liqdebt_withinterest: float, 货币资金／带息流动负债
        op_to_liqdebt: float, 营业利润／流动负债
        op_to_debt: float, 营业利润／负债合计
        roic_yearly: float, 年化投入资本回报率
        total_fa_trun: float, 固定资产合计周转率
        profit_to_op: float, 利润总额／营业收入
        q_opincome: float, 经营活动单季度净收益
        q_investincome: float, 价值变动单季度净收益
        q_dtprofit: float, 扣除非经常损益后的单季度净利润
        q_eps: float, 每股收益(单季度)
        q_netprofit_margin: float, 销售净利率(单季度)
        q_gsprofit_margin: float, 销售毛利率(单季度)
        q_exp_to_sales: float, 销售期间费用率(单季度)
        q_profit_to_gr: float, 净利润／营业总收入(单季度)
        q_saleexp_to_gr: float, 销售费用／营业总收入 (单季度)
        q_adminexp_to_gr: float, 管理费用／营业总收入 (单季度)
        q_finaexp_to_gr: float, 财务费用／营业总收入 (单季度)
        q_impair_to_gr_ttm: float, 资产减值损失／营业总收入(单季度)
        q_gc_to_gr: float, 营业总成本／营业总收入 (单季度)
        q_op_to_gr: float, 营业利润／营业总收入(单季度)
        q_roe: float, 净资产收益率(单季度)
        q_dt_roe: float, 净资产单季度收益率(扣除非经常损益)
        q_npta: float, 总资产净利润(单季度)
        q_opincome_to_ebt: float, 经营活动净收益／利润总额(单季度)
        q_investincome_to_ebt: float, 价值变动净收益／利润总额(单季度)
        q_dtprofit_to_profit: float, 扣除非经常损益后的净利润／净利润(单季度)
        q_salescash_to_or: float, 销售商品提供劳务收到的现金／营业收入(单季度)
        q_ocf_to_sales: float, 经营活动产生的现金流量净额／营业收入(单季度)
        q_ocf_to_or: float, 经营活动产生的现金流量净额／经营活动净收益(单季度)
        basic_eps_yoy: float, 基本每股收益同比增长率(%)
        dt_eps_yoy: float, 稀释每股收益同比增长率(%)
        cfps_yoy: float, 每股经营活动产生的现金流量净额同比增长率(%)
        op_yoy: float, 营业利润同比增长率(%)
        ebt_yoy: float, 利润总额同比增长率(%)
        netprofit_yoy: float, 归属母公司股东的净利润同比增长率(%)
        dt_netprofit_yoy: float, 归属母公司股东的净利润-扣除非经常损益同比增长率(%)
        ocf_yoy: float, 经营活动产生的现金流量净额同比增长率(%)
        roe_yoy: float, 净资产收益率(摊薄)同比增长率(%)
        bps_yoy: float, 每股净资产相对年初增长率(%)
        assets_yoy: float, 资产总计相对年初增长率(%)
        eqt_yoy: float, 归属母公司的股东权益相对年初增长率(%)
        tr_yoy: float, 营业总收入同比增长率(%)
        or_yoy: float, 营业收入同比增长率(%)
        q_gr_yoy: float, 营业总收入同比增长率(%)(单季度)
        q_gr_qoq: float, 营业总收入环比增长率(%)(单季度)
        q_sales_yoy: float, 营业收入同比增长率(%)(单季度)
        q_sales_qoq: float, 营业收入环比增长率(%)(单季度)
        q_op_yoy: float, 营业利润同比增长率(%)(单季度)
        q_op_qoq: float, 营业利润环比增长率(%)(单季度)
        q_profit_yoy: float, 净利润同比增长率(%)(单季度)
        q_profit_qoq: float, 净利润环比增长率(%)(单季度)
        equity_yoy: float, 净资产同比增长率
        rd_exp: float, 研发费用
        update_flag: str, 更新标识
        """
        if not period:
            raise ValueError("The 'period' parameter is required for fina_indicator_vip.")
        if "ts_code" in kwargs:
            logger.warning(
                "The 'ts_code' parameter is not applicable for the 'fina_indicator_vip' API and will be ignored."
            )
            del kwargs["ts_code"]

        return self.client.fetch(
            "fina_indicator_vip",
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            **kwargs,
        )