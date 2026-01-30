"""运行参数优化"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from backtest_money_signal import parameter_optimization

if __name__ == '__main__':
    parameter_optimization()
