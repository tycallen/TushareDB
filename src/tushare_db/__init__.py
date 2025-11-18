# src/tushare_db/__init__.py

# 新架构 - 职责分离
from .downloader import DataDownloader  # 数据下载
from .reader import DataReader          # 数据查询

# 工具
from .logger import setup_logging, get_logger

# Define what is exposed when a user does 'from tushare_db import *'
__all__ = [
    # 核心接口
    'DataDownloader',
    'DataReader',
    # 工具
    'setup_logging',
    'get_logger',
]