"""
申万板块关系分析模块

提供板块涨跌幅计算、相关性分析、传导关系检测、联动强度计算等功能。
"""

from .calculator import SectorCalculator
from .analyzer import SectorAnalyzer
from .reporter import OutputManager

__all__ = [
    'SectorCalculator',
    'SectorAnalyzer',
    'OutputManager',
]

__version__ = '0.1.0'
