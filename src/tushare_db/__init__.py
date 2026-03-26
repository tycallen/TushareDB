# src/tushare_db/__init__.py

# 新架构 - 职责分离
from .downloader import DataDownloader  # 数据下载
from .reader import DataReader          # 数据查询

# 概念板块数据管理器 (jquant_data_sync)
from .concept_manager import ConceptDataManager

# 工具
from .logger import setup_logging, get_logger

# Define what is exposed when a user does 'from tushare_db import *'
__all__ = [
    # 核心接口
    'DataDownloader',
    'DataReader',
    'ConceptDataManager',
    # 工具
    'setup_logging',
    'get_logger',
]