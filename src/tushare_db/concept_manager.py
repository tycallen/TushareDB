# -*- coding: utf-8 -*-
"""
A股概念板块数据管理器 (jquant_data_sync)

职责：从 GitHub Release 动态拉取概念板块 PIT 数据，支持本地缓存
设计理念：零数据库依赖，CSV 文件缓存，按需下载
"""
import os
import re
from pathlib import Path
from datetime import date
from typing import Optional, List
import pandas as pd
import requests

from .logger import get_logger

logger = get_logger(__name__)


class ConceptDataManagerError(Exception):
    """概念板块数据管理器异常"""
    pass


class ConceptDataManager:
    """
    A股概念板块数据管理器

    数据来源：https://github.com/tycallen/jquant_data_sync
    数据结构：SCD (缓慢变化维度) 区间生效
    缓存位置：与 tushare.db 同级目录下的 .concept_cache/

    使用示例：
        >>> manager = ConceptDataManager(db_path="data/tushare.db")
        >>> # 自动下载/更新数据
        >>> manager.pull_data()
        >>> # 查询某天属于人工智能板块的所有股票
        >>> stocks = manager.get_concept_stocks(trade_date='20240115', concept_name='人工智能')
        >>> # 查询某只股票在某天的所有概念
        >>> concepts = manager.get_stock_concepts(trade_date='20240115', ts_code='000001.SZ')
    """

    GITHUB_REPO = "tycallen/jquant_data_sync"
    DATA_FILENAME = "all_concepts_pit_scd.csv"

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化概念板块数据管理器

        Args:
            db_path: DuckDB 数据库路径（用于确定缓存目录位置）
                    默认为当前目录下的 tushare.db
        """
        if db_path is None:
            db_path = os.getenv("DB_PATH", "tushare.db")

        # 缓存目录：与 tushare.db 同级目录下的 .concept_cache/
        db_path = Path(db_path).resolve()
        self.cache_dir = db_path.parent / ".concept_cache"
        self.cache_file = self.cache_dir / self.DATA_FILENAME

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存中的数据
        self._df: Optional[pd.DataFrame] = None
        self._load_date: Optional[date] = None

        logger.info(f"ConceptDataManager 初始化完成: cache_dir={self.cache_dir}")

    def _get_download_url(self, target_date: Optional[date] = None) -> str:
        """
        构建下载链接

        Args:
            target_date: 目标日期，默认为今天

        Returns:
            GitHub Release 下载直链
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.strftime('%Y-%m-%d')
        return (
            f"https://github.com/{self.GITHUB_REPO}/releases/download/"
            f"data-{date_str}/{self.DATA_FILENAME}"
        )

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效（今日已下载）"""
        if not self.cache_file.exists():
            return False

        # 获取文件修改日期
        file_mtime = date.fromtimestamp(self.cache_file.stat().st_mtime)
        today = date.today()

        return file_mtime == today

    def _convert_stock_code(self, code: str, to_tushare: bool = True) -> str:
        """
        转换股票代码格式

        Args:
            code: 股票代码
            to_tushare: True=转换为 .SZ/.SH 格式, False=转换为 .XSHE/.XSHG 格式

        Returns:
            转换后的股票代码
        """
        if to_tushare:
            # .XSHE/.XSHG -> .SZ/.SH
            code = code.replace('.XSHE', '.SZ').replace('.XSHG', '.SH')
        else:
            # .SZ/.SH -> .XSHE/.XSHG
            code = code.replace('.SZ', '.XSHE').replace('.SH', '.XSHG')
        return code

    def pull_data(self, force: bool = False) -> bool:
        """
        拉取最新概念板块数据

        Args:
            force: 是否强制重新下载（忽略缓存）

        Returns:
            是否成功获取数据
        """
        # 检查缓存
        if not force and self._is_cache_valid():
            logger.info(f"✅ 发现今日缓存 ({self.cache_file})，跳过网络请求")
            return self._load_data()

        # 尝试下载今日数据
        today = date.today()
        download_url = self._get_download_url(today)

        logger.info(f"🔄 准备拉取今日数据...\n🔗 链接: {download_url}")

        try:
            response = requests.get(download_url, timeout=30)

            if response.status_code == 200:
                # 保存到缓存
                self.cache_file.write_bytes(response.content)
                logger.info(f"✅ 数据下载成功: {len(response.content)} bytes")
                return self._load_data()

            elif response.status_code == 404:
                logger.warning("⚠️ 今日数据尚未发布（或周末休市无更新）")
                # 尝试使用历史缓存
                if self.cache_file.exists():
                    logger.info(f"📂 使用历史缓存: {self.cache_file}")
                    return self._load_data()
                return False

            else:
                logger.error(f"❌ 下载失败，HTTP 状态码: {response.status_code}")
                # 尝试使用历史缓存
                if self.cache_file.exists():
                    logger.info(f"📂 使用历史缓存: {self.cache_file}")
                    return self._load_data()
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 请求异常: {e}")
            # 尝试使用历史缓存
            if self.cache_file.exists():
                logger.info(f"📂 使用历史缓存: {self.cache_file}")
                return self._load_data()
            return False

    def _load_data(self) -> bool:
        """
        从缓存加载数据到内存

        Returns:
            是否成功加载
        """
        if not self.cache_file.exists():
            logger.error(f"❌ 缓存文件不存在: {self.cache_file}")
            return False

        try:
            # 强制指定字符串类型，防止吞掉前导0
            self._df = pd.read_csv(
                self.cache_file,
                dtype={'stock_code': str, 'concept_code': str}
            )

            # 统一日期格式为 YYYYMMDD
            self._df['in_date'] = pd.to_datetime(self._df['in_date']).dt.strftime('%Y%m%d')
            self._df['out_date'] = pd.to_datetime(self._df['out_date']).dt.strftime('%Y%m%d')

            # 添加 ts_code 列（Tushare 格式）
            self._df['ts_code'] = self._df['stock_code'].apply(
                lambda x: self._convert_stock_code(x, to_tushare=True)
            )

            self._load_date = date.fromtimestamp(self.cache_file.stat().st_mtime)

            logger.info(
                f"✅ 数据加载完成: {len(self._df)} 行, "
                f"{self._df['concept_code'].nunique()} 个概念, "
                f"{self._df['ts_code'].nunique()} 只股票"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 加载数据失败: {e}")
            return False

    def _ensure_data_loaded(self) -> None:
        """确保数据已加载"""
        if self._df is None:
            if not self.pull_data():
                raise ConceptDataManagerError("数据加载失败，请先调用 pull_data()")

    def get_cross_section(self, trade_date: str) -> pd.DataFrame:
        """
        获取指定日期的概念板块截面数据

        Args:
            trade_date: 交易日期 YYYYMMDD

        Returns:
            该日期的概念板块数据 DataFrame
            列: concept_code, concept_name, stock_code, ts_code, in_date, out_date
        """
        self._ensure_data_loaded()

        # 格式统一
        trade_date_fmt = trade_date.replace('-', '')

        # 核心过滤逻辑：纳入日期 <= 目标日 且 剔除日期 >= 目标日
        mask = (
            (self._df['in_date'] <= trade_date_fmt) &
            (self._df['out_date'] >= trade_date_fmt)
        )

        result = self._df[mask].copy()
        logger.debug(f"截面数据查询: {trade_date} -> {len(result)} 条记录")

        return result

    def get_concept_stocks(
        self,
        trade_date: str,
        concept_name: Optional[str] = None,
        concept_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取指定日期某概念板块的所有成分股

        Args:
            trade_date: 交易日期 YYYYMMDD
            concept_name: 概念板块名称（如 '人工智能'）
            concept_code: 概念板块代码（如 'GN036'）

        Returns:
            成分股 DataFrame，列: ts_code, name, in_date, out_date

        Example:
            >>> df = manager.get_concept_stocks('20240115', concept_name='人工智能')
            >>> print(df['ts_code'].tolist())  # ['000001.SZ', '000002.SZ', ...]
        """
        self._ensure_data_loaded()

        if concept_name is None and concept_code is None:
            raise ValueError("concept_name 和 concept_code 至少提供一个")

        # 获取截面数据
        df = self.get_cross_section(trade_date)

        # 筛选概念
        if concept_name:
            df = df[df['concept_name'] == concept_name]
        if concept_code:
            df = df[df['concept_code'] == concept_code]

        if df.empty:
            return pd.DataFrame(columns=['ts_code', 'concept_name', 'in_date', 'out_date'])

        # 返回简洁格式
        return df[['ts_code', 'concept_name', 'in_date', 'out_date']].reset_index(drop=True)

    def get_stock_concepts(
        self,
        trade_date: str,
        ts_code: str
    ) -> pd.DataFrame:
        """
        获取指定日期某股票所属的所有概念板块

        Args:
            trade_date: 交易日期 YYYYMMDD
            ts_code: 股票代码（支持 .SZ/.SH 或 .XSHE/.XSHG 格式）

        Returns:
            概念板块 DataFrame，列: concept_code, concept_name, in_date, out_date

        Example:
            >>> df = manager.get_stock_concepts('20240115', ts_code='000001.SZ')
            >>> print(df['concept_name'].tolist())  # ['人工智能', '区块链', ...]
        """
        self._ensure_data_loaded()

        # 统一代码格式
        ts_code = ts_code.replace('.XSHE', '.SZ').replace('.XSHG', '.SH')

        # 获取截面数据
        df = self.get_cross_section(trade_date)

        # 筛选股票
        df = df[df['ts_code'] == ts_code]

        if df.empty:
            return pd.DataFrame(columns=['concept_code', 'concept_name', 'in_date', 'out_date'])

        # 返回简洁格式
        return df[['concept_code', 'concept_name', 'in_date', 'out_date']].reset_index(drop=True)

    def get_all_concepts(self) -> pd.DataFrame:
        """
        获取所有概念板块列表（去重）

        Returns:
            概念板块 DataFrame，列: concept_code, concept_name
        """
        self._ensure_data_loaded()

        concepts = self._df[['concept_code', 'concept_name']].drop_duplicates()
        return concepts.sort_values('concept_code').reset_index(drop=True)

    def search_concepts(self, keyword: str) -> pd.DataFrame:
        """
        搜索概念板块

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的概念板块 DataFrame
        """
        self._ensure_data_loaded()

        mask = self._df['concept_name'].str.contains(keyword, case=False, na=False)
        concepts = self._df[mask][['concept_code', 'concept_name']].drop_duplicates()
        return concepts.reset_index(drop=True)

    def get_cache_info(self) -> dict:
        """
        获取缓存信息

        Returns:
            包含缓存状态的字典
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'cache_file': str(self.cache_file),
            'exists': self.cache_file.exists(),
            'valid_today': self._is_cache_valid(),
        }

        if self.cache_file.exists():
            stat = self.cache_file.stat()
            info['file_size'] = stat.st_size
            info['modify_time'] = date.fromtimestamp(stat.st_mtime).isoformat()

        if self._df is not None:
            info['loaded'] = True
            info['loaded_date'] = self._load_date.isoformat() if self._load_date else None
            info['total_records'] = len(self._df)
            info['total_concepts'] = self._df['concept_code'].nunique()
            info['total_stocks'] = self._df['ts_code'].nunique()
        else:
            info['loaded'] = False

        return info

    def clear_cache(self) -> bool:
        """
        清除缓存文件

        Returns:
            是否成功清除
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"🗑️ 缓存已清除: {self.cache_file}")
            self._df = None
            self._load_date = None
            return True
        except Exception as e:
            logger.error(f"❌ 清除缓存失败: {e}")
            return False
