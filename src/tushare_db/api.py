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
from typing import TYPE_CHECKING, Optional, Union, List

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
    # 构建传递给 client.query 的参数字典
    params = {
        "exchange": exchange,
        "start_date": start_date,
        "end_date": end_date,
        "is_open": is_open,
        "fields": fields
    }
    # 过滤掉值�� None 的参数
    params = {k: v for k, v in params.items() if v is not None}

    # 通过 client 获取数据
    return client.get_data('trade_cal', **params)


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


def dc_index(
    client: 'TushareDBClient',
    ts_code: Optional[Union[str, List[str]]] = None,
    name: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None
) -> pd.DataFrame:
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
    params = {
        "ts_code": ts_code,
        "name": name,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "fields": fields
    }
    params = {k: v for k, v in params.items() if v is not None}
    return client.get_data('dc_index', **params)