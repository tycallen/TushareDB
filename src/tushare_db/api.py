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
    client: TushareDBClient,
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

    :param client: TushareDBClient 实例。
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
    client: TushareDBClient,
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    is_open: str = None,
    fields: str = 'exchange,cal_date,is_open,pretrade_date'
) -> pd.DataFrame:
    """
    获取各大交易所交易日历数据,默认提取的是上交所。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: TushareDBClient 实例。
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
    client: TushareDBClient,
    hs_type: str,
    is_new: str = '1',
    fields: str = 'ts_code,hs_type,in_date,out_date,is_new'
) -> pd.DataFrame:
    """
    获取沪股通、深股通成分数据。
    数据将首先尝试从本地缓存获取，如果缓存中不存在，则通过Tushare API获取并存入缓存。

    :param client: TushareDBClient 实例。
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
