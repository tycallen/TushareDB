#!/usr/bin/env python3
"""
增强版蒙特卡洛因子验证脚本

改进点：
1. 全A股数据（5000+股票，10年历史）
2. 多年模拟（3-5年一个周期）
3. 针对低频信号的贝叶斯估计
4. 隔夜跳空模拟模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tushare_db import DataReader
from src.tushare_db.factor_validation import FactorRegistry
from scipy import stats


@dataclass
class ValidationConfig:
    """验证配置"""
    # 数据配置
    n_stocks: int = 5000  # 使用股票数量（全A）
    history_years: int = 10  # 历史数据年限

    # 模拟配置
    n_paths: int = 100_000  # 模拟路径数（增加到10万）
    simulation_years: int = 5  # 单次模拟年限（从1年增加到5年）

    # 低频信号特殊处理
    min_samples_threshold: int = 100  # 低频因子最小样本数
    use_bayesian: bool = True  # 对低频信号使用贝叶斯估计

    # 隔夜跳空模拟
    simulate_gap: bool = True  # 是否模拟隔夜跳空
    gap_prob: float = 0.3  # 出现跳空的概率
    gap_mean: float = 0.0  # 跳空均值
    gap_std: float = 0.01  # 跳空标准差（1%）

    # 输出配置
    output_dir: str = "./validation_results"

    @property
    def n_steps(self) -> int:
        """每年的交易日数"""
        return 252

    @property
    def total_simulation_days(self) -> int:
        """总模拟天数"""
        return self.simulation_years * self.n_steps


class GapAwareGBM:
    """
    带隔夜跳跳的GBM模型

    标准GBM：连续交易，无跳空
    改进版：每天开盘时可能产生跳空
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def generate(self, S0: float = 100.0, mu: float = 0.0, sigma: float = 0.2) -> np.ndarray:
        """
        生成带跳空的价格路径

        Args:
            S0: 初始价格
            mu: 预期收益率
            sigma: 波动率

        Returns:
            (n_steps,) 价格序列
        """
        n_steps = self.config.total_simulation_days
        dt = 1 / self.config.n_steps

        prices = np.zeros(n_steps)
        prices[0] = S0

        for t in range(1, n_steps):
            # 隔夜跳空（只在每天开盘时）
            if self.config.simulate_gap:
                gap = self._generate_gap()
            else:
                gap = 0.0

            # 日内GBM
            Z = np.random.standard_normal()
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z

            # 价格更新：前收盘 × (1+跳空) × GBM
            prices[t] = prices[t-1] * (1 + gap) * np.exp(drift + diffusion)

        return prices

    def generate_ohlc(self, S0: float = 100.0, mu: float = 0.0,
                      sigma: float = 0.2) -> Dict[str, np.ndarray]:
        """
        生成OHLC数据（带跳空）

        改进：更真实的微观结构
        - 开盘价 = 前收盘价 × (1 ± 跳空)
        - 日内高/低基于波动率
        - 收盘价 = 基于GBM
        """
        n_steps = self.config.total_simulation_days
        dt = 1 / self.config.n_steps

        opens = np.zeros(n_steps)
        highs = np.zeros(n_steps)
        lows = np.zeros(n_steps)
        closes = np.zeros(n_steps)
        volumes = np.zeros(n_steps)

        prev_close = S0

        for t in range(n_steps):
            # 隔夜跳空
            if self.config.simulate_gap and t > 0:
                gap = self._generate_gap()
                open_price = prev_close * (1 + gap)
            else:
                open_price = prev_close

            # 日内波动
            Z = np.random.standard_normal()
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z

            close_price = open_price * np.exp(drift + diffusion)

            # 日内高/低（基于波动率）
            intraday_vol = sigma * np.sqrt(dt) * np.abs(np.random.standard_normal())
            high_price = max(open_price, close_price) * (1 + intraday_vol * 0.5)
            low_price = min(open_price, close_price) * (1 - intraday_vol * 0.5)

            # 成交量（与波动率相关）
            volume = np.random.lognormal(mean=10, sigma=0.5) * (1 + intraday_vol * 10)

            opens[t] = open_price
            highs[t] = high_price
            lows[t] = low_price
            closes[t] = close_price
            volumes[t] = volume

            prev_close = close_price

        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'vol': volumes
        }

    def _generate_gap(self) -> float:
        """生成隔夜跳空（有概率无跳空）"""
        if np.random.random() < self.config.gap_prob:
            return np.random.normal(self.config.gap_mean, self.config.gap_std)
        return 0.0


class BayesianAlphaEstimator:
    """
    贝叶斯Alpha估计器

    针对低频信号，使用Beta先验 + 二项似然
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 99.0):
        """
        Args:
            prior_alpha: Beta分布alpha参数（预期成功次数）
            prior_beta: Beta分布beta参数（预期失败次数）
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def estimate(self, n_success: int, n_total: int,
                 confidence: float = 0.95) -> Dict[str, float]:
        """
        估计触发概率的后验分布

        Args:
            n_success: 成功次数（触发次数）
            n_total: 总观测数
            confidence: 置信水平

        Returns:
            包含后验均值、中位数、置信区间的字典
        """
        # Beta后验分布
        posterior_alpha = self.prior_alpha + n_success
        posterior_beta = self.prior_beta + (n_total - n_success)

        # 后验统计量
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

        # 置信区间
        ci_lower = stats.beta.ppf((1 - confidence) / 2, posterior_alpha, posterior_beta)
        ci_upper = stats.beta.ppf((1 + confidence) / 2, posterior_alpha, posterior_beta)

        return {
            'mean': posterior_mean,
            'median': stats.beta.median(posterior_alpha, posterior_beta),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.sqrt(posterior_alpha * posterior_beta /
                          ((posterior_alpha + posterior_beta)**2 *
                           (posterior_alpha + posterior_beta + 1)))
        }


class EnhancedMonteCarloValidator:
    """增强版蒙特卡洛验证器"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.reader = DataReader()
        self.gbm = GapAwareGBM(config)
        self.bayesian = BayesianAlphaEstimator()

    def load_historical_data(self) -> pd.DataFrame:
        """
        加载历史数据

        使用10年全A股数据
        """
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.history_years * 365)

        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        print(f"加载历史数据: {start_str} 至 {end_str}")
        print(f"股票数量: 约 {self.config.n_stocks} 只")

        # 获取活跃股票列表
        stocks = self.reader.get_stock_basic(list_status='L')

        # 限制股票数量
        if len(stocks) > self.config.n_stocks:
            # 按市值排序，取大盘股
            stocks = stocks.head(self.config.n_stocks)

        # 获取日线数据
        ts_codes = stocks['ts_code'].tolist()

        # 分批查询
        all_data = []
        batch_size = 500

        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i+batch_size]
            codes_str = "','".join(batch)

            query = f"""
                SELECT ts_code, trade_date, open, high, low, close, vol
                FROM daily
                WHERE ts_code IN ('{codes_str}')
                AND trade_date BETWEEN '{start_str}' AND '{end_str}'
            """
            df = self.reader.db.execute_query(query)
            all_data.append(df)

            if (i // batch_size + 1) % 5 == 0:
                print(f"  已加载 {min(i+batch_size, len(ts_codes))}/{len(ts_codes)} 只股票...")

        df_all = pd.concat(all_data, ignore_index=True)

        print(f"\\n数据加载完成: {len(df_all):,} 条记录")
        print(f"  股票数: {df_all['ts_code'].nunique()}")
        print(f"  交易日: {df_all['trade_date'].nunique()}")

        return df_all

    def calculate_p_actual(self, df: pd.DataFrame, factor_name: str) -> Tuple[float, int]:
        """
        计算真实触发概率

        Returns:
            (p_actual, n_triggers)
        """
        factor = FactorRegistry.get(factor_name)

        total_triggers = 0
        total_observations = 0

        # 按股票分组计算
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')

            # 准备OHLCV数据
            ohlcv = group[['open', 'high', 'low', 'close', 'vol']].copy()

            # 计算因子信号
            try:
                signals = factor.evaluate(ohlcv)
                total_triggers += signals.sum()
                total_observations += len(signals)
            except Exception as e:
                # 某些股票数据可能不完整，跳过
                continue

        p_actual = total_triggers / total_observations if total_observations > 0 else 0

        return p_actual, int(total_triggers)

    def calculate_p_random(self, factor_name: str, n_samples: int = 5000) -> Tuple[float, int]:
        """
        计算随机触发概率（分层采样）

        改进：使用多年模拟
        """
        factor = FactorRegistry.get(factor_name)

        total_triggers = 0
        total_observations = 0

        # 分层参数
        mu_values = [-0.2, -0.1, 0.0, 0.1, 0.2]  # 不同市场环境
        sigma_values = [0.1, 0.2, 0.3, 0.4]

        samples_per_layer = n_samples // (len(mu_values) * len(sigma_values))

        print(f"  分层采样: {len(mu_values)} 种趋势 × {len(sigma_values)} 种波动率 = {len(mu_values)*len(sigma_values)} 层")
        print(f"  每层 {samples_per_layer} 条路径，共 {n_samples} 条路径")
        print(f"  每路径 {self.config.total_simulation_days} 个交易日（{self.config.simulation_years}年）")

        for mu in mu_values:
            for sigma in sigma_values:
                for _ in range(samples_per_layer):
                    # 生成OHLC数据
                    ohlc = self.gbm.generate_ohlc(S0=100.0, mu=mu, sigma=sigma)

                    # 转换为DataFrame
                    df_sim = pd.DataFrame(ohlc)

                    # 计算因子信号
                    try:
                        signals = factor.evaluate(df_sim)
                        total_triggers += signals.sum()
                        total_observations += len(signals)
                    except Exception as e:
                        continue

        p_random = total_triggers / total_observations if total_observations > 0 else 0

        return p_random, int(total_triggers)

    def validate_factor(self, factor_name: str, df_historical: pd.DataFrame) -> Dict:
        """
        验证单个因子
        """
        print(f"\\n{'='*60}")
        print(f"验证因子: {factor_name}")
        print(f"{'='*60}")

        # 1. 计算 P_actual
        print("\\n[1/3] 计算真实触发概率...")
        p_actual, n_actual = self.calculate_p_actual(df_historical, factor_name)
        print(f"  P_actual = {p_actual:.4%} ({n_actual} 次触发)")

        # 2. 计算 P_random
        print("\\n[2/3] 计算随机触发概率（蒙特卡洛模拟）...")
        p_random, n_random = self.calculate_p_random(factor_name)
        print(f"  P_random = {p_random:.4%} ({n_random} 次触发)")

        # 3. 计算Alpha
        print("\\n[3/3] 计算Alpha比率...")

        if p_random == 0:
            # 低频信号：使用贝叶斯估计
            if self.config.use_bayesian:
                print("  低频信号检测，使用贝叶斯估计...")

                # P_actual 的后验
                n_total = len(df_historical)
                posterior_actual = self.bayesian.estimate(n_actual, n_total)

                # P_random 的后验（假设在模拟中也观测到0次）
                n_sim_total = self.config.total_simulation_days * 5000  # 5000条路径 × 天数
                posterior_random = self.bayesian.estimate(n_random, n_sim_total)

                # Alpha = E[P_actual] / E[P_random]
                alpha_mean = posterior_actual['mean'] / max(posterior_random['mean'], 1e-10)
                alpha_lower = posterior_actual['ci_lower'] / max(posterior_random['ci_upper'], 1e-10)
                alpha_upper = posterior_actual['ci_upper'] / max(posterior_random['ci_lower'], 1e-10)

                print(f"  贝叶斯估计结果:")
                print(f"    P_actual: {posterior_actual['mean']:.4%} [{posterior_actual['ci_lower']:.4%}, {posterior_actual['ci_upper']:.4%}]")
                print(f"    P_random: {posterior_random['mean']:.4%} [{posterior_random['ci_lower']:.4%}, {posterior_random['ci_upper']:.4%}]")
                print(f"    Alpha: {alpha_mean:.2f} [{alpha_lower:.2f}, {alpha_upper:.2f}]")

                result = {
                    'factor_name': factor_name,
                    'p_actual': p_actual,
                    'p_actual_ci': [posterior_actual['ci_lower'], posterior_actual['ci_upper']],
                    'n_actual': n_actual,
                    'p_random': posterior_random['mean'],
                    'p_random_ci': [posterior_random['ci_lower'], posterior_random['ci_upper']],
                    'n_random': n_random,
                    'alpha_ratio': alpha_mean,
                    'alpha_ci': [alpha_lower, alpha_upper],
                    'method': 'bayesian',
                    'is_low_frequency': True
                }
            else:
                # 不使用贝叶斯，标记为不可评估
                result = {
                    'factor_name': factor_name,
                    'p_actual': p_actual,
                    'n_actual': n_actual,
                    'p_random': 0,
                    'n_random': n_random,
                    'alpha_ratio': float('inf'),
                    'method': 'point_estimate',
                    'is_low_frequency': True,
                    'warning': 'P_random=0，点估计不可靠'
                }
        else:
            # 正常频率：标准Alpha计算
            alpha = p_actual / p_random if p_random > 0 else float('inf')

            # 计算标准误
            se = np.sqrt(p_actual * (1 - p_actual) / len(df_historical) +
                        p_random * (1 - p_random) / (self.config.total_simulation_days * 5000))

            print(f"  Alpha = {alpha:.2f}")
            print(f"  标准误 = {se:.6f}")

            result = {
                'factor_name': factor_name,
                'p_actual': p_actual,
                'n_actual': n_actual,
                'p_random': p_random,
                'n_random': n_random,
                'alpha_ratio': alpha,
                'se': se,
                'method': 'frequentist',
                'is_low_frequency': False
            }

        # 评估建议
        if result.get('alpha_ratio', 0) >= 2.0:
            result['recommendation'] = 'KEEP'
            result['confidence'] = 'HIGH' if not result.get('is_low_frequency') else 'MEDIUM'
        elif result.get('alpha_ratio', 0) >= 1.5:
            result['recommendation'] = 'OPTIMIZE'
            result['confidence'] = 'MEDIUM'
        else:
            result['recommendation'] = 'DISCARD'
            result['confidence'] = 'HIGH'

        print(f"\\n  建议: {result['recommendation']} (置信度: {result['confidence']})")

        return result

    def run_validation(self, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        运行完整验证
        """
        # 加载历史数据
        df_historical = self.load_historical_data()

        # 获取因子列表
        if factor_names is None:
            factor_names = FactorRegistry.list_builtin()

        print(f"\\n将验证 {len(factor_names)} 个因子")
        print(f"配置:")
        print(f"  历史数据: {self.config.history_years}年, {self.config.n_stocks}只股票")
        print(f"  模拟: {self.config.n_paths:,}路径 × {self.config.simulation_years}年")
        print(f"  跳空模拟: {'开启' if self.config.simulate_gap else '关闭'}")
        print(f"  贝叶斯估计: {'开启' if self.config.use_bayesian else '关闭'}")

        # 验证每个因子
        results = []
        for i, factor_name in enumerate(factor_names, 1):
            print(f"\\n{'#'*60}")
            print(f"# 进度: {i}/{len(factor_names)} - {factor_name}")
            print(f"{'#'*60}")

            try:
                result = self.validate_factor(factor_name, df_historical)
                results.append(result)
            except Exception as e:
                print(f"  ❌ 验证失败: {e}")
                results.append({
                    'factor_name': factor_name,
                    'error': str(e)
                })

        # 创建结果DataFrame
        df_results = pd.DataFrame(results)

        # 保存结果
        output_file = Path(self.config.output_dir) / f"enhanced_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_file, index=False)

        print(f"\\n{'='*60}")
        print("验证完成！")
        print(f"{'='*60}")
        print(f"结果保存至: {output_file}")

        # 打印摘要
        print(f"\\n摘要:")
        print(f"  通过因子 (Alpha>=2.0): {len(df_results[df_results['alpha_ratio'] >= 2.0])}")
        print(f"  优化因子 (1.5<=Alpha<2.0): {len(df_results[(df_results['alpha_ratio'] >= 1.5) & (df_results['alpha_ratio'] < 2.0)])}")
        print(f"  废弃因子 (Alpha<1.5): {len(df_results[df_results['alpha_ratio'] < 1.5])}")

        return df_results


def main():
    parser = argparse.ArgumentParser(description='增强版蒙特卡洛因子验证')
    parser.add_argument('--factors', nargs='+', help='要验证的因子名称（默认全部）')
    parser.add_argument('--n-stocks', type=int, default=5000, help='使用股票数量')
    parser.add_argument('--history-years', type=int, default=10, help='历史数据年限')
    parser.add_argument('--n-paths', type=int, default=100000, help='模拟路径数')
    parser.add_argument('--sim-years', type=int, default=5, help='单次模拟年限')
    parser.add_argument('--no-bayesian', action='store_true', help='关闭贝叶斯估计')
    parser.add_argument('--no-gap', action='store_true', help='关闭跳空模拟')
    parser.add_argument('--output', default='./validation_results', help='输出目录')

    args = parser.parse_args()

    # 创建配置
    config = ValidationConfig(
        n_stocks=args.n_stocks,
        history_years=args.history_years,
        n_paths=args.n_paths,
        simulation_years=args.sim_years,
        use_bayesian=not args.no_bayesian,
        simulate_gap=not args.no_gap,
        output_dir=args.output
    )

    # 运行验证
    validator = EnhancedMonteCarloValidator(config)
    validator.run_validation(args.factors)


if __name__ == '__main__':
    main()
