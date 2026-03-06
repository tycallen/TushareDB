import numpy as np
import pytest


def test_gbm_simulator_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator
    assert GBMSimulator is not None


def test_gbm_simulator_init():
    """测试初始化参数"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    assert sim.n_paths == 100
    assert sim.n_steps == 50
    assert sim.limit_up == 0.10
    assert sim.limit_down == -0.10


def test_gbm_simulate_shape():
    """测试模拟结果形状正确"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert result.shape == (100, 50)
    assert isinstance(result, np.ndarray)


def test_gbm_price_positive():
    """测试价格始终为正"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.all(result > 0)


def test_gbm_first_column_is_s0():
    """测试第一列是初始价格"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.allclose(result[:, 0], 100)


def test_gbm_price_limits():
    """测试涨跌停约束生效"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50, limit_up=0.10, limit_down=-0.10)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.5)  # 高波动率更容易触发限制

    # 计算日收益率
    returns = np.diff(result, axis=1) / result[:, :-1]

    # 检查没有超过涨跌停限制的收益率
    assert np.all(returns <= 0.10 + 1e-6)
    assert np.all(returns >= -0.10 - 1e-6)


def test_gbm_reproducibility():
    """测试可重复性（设置随机种子）"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim1 = GBMSimulator(n_paths=100, n_steps=50, random_seed=42)
    result1 = sim1.simulate(s0=100, mu=0.1, sigma=0.2)

    sim2 = GBMSimulator(n_paths=100, n_steps=50, random_seed=42)
    result2 = sim2.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.allclose(result1, result2)
