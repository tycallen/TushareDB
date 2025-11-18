# src/tushare_db/__init__.py

# 新架构：推荐使用
from .downloader import DataDownloader
from .reader import DataReader

# 旧架构：保持向后兼容（将来会废弃）
from .client import TushareDBClient

# 工具
from .logger import setup_logging, get_logger

# Define what is exposed when a user does 'from tushare_db import *'
__all__ = [
    # 新架构（推荐）
    'DataDownloader',
    'DataReader',
    # 旧架构（兼容）
    'TushareDBClient',
    # 工具
    'setup_logging',
    'get_logger',
]