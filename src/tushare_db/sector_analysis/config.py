"""
配置管理模块
"""

# 自适应滞后窗口配置
MAX_LAG_MAP = {
    'daily': 5,      # 1-5日滞后
    'weekly': 4,     # 1-4周滞后
    'monthly': 3,    # 1-3月滞后
}

# 默认输出目录
DEFAULT_OUTPUT_DIR = 'output'

# 日志配置
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL_CONSOLE = 'INFO'
LOG_LEVEL_FILE = 'DEBUG'
