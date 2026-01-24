"""
报告生成器
"""

import os
import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from .config import DEFAULT_OUTPUT_DIR, LOG_FORMAT, LOG_LEVEL_CONSOLE, LOG_LEVEL_FILE

logger = logging.getLogger(__name__)


class OutputManager:
    """输出管理器"""

    def __init__(self, output_dir: str = None):
        """
        初始化输出管理器

        Args:
            output_dir: 输出目录路径
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y-%m-%d')
            output_dir = f"{DEFAULT_OUTPUT_DIR}/{timestamp}_analysis"

        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        self.plots_dir = os.path.join(output_dir, 'plots')

        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        """配置日志"""
        log_file = os.path.join(self.output_dir, 'analysis.log')

        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(getattr(logging, LOG_LEVEL_FILE))
        fh.setFormatter(logging.Formatter(LOG_FORMAT))

        # 控制台handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, LOG_LEVEL_CONSOLE))
        ch.setFormatter(logging.Formatter(LOG_FORMAT))

        # 添加到logger
        root_logger = logging.getLogger('tushare_db.sector_analysis')
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(fh)
        root_logger.addHandler(ch)

        logger.info(f"输出目录: {self.output_dir}")

    def save_dataframe(self, df: pd.DataFrame, name: str, format: str = 'csv'):
        """
        保存DataFrame

        Args:
            df: 数据
            name: 文件名（不含扩展名）
            format: 格式 (csv/excel)
        """
        if format == 'csv':
            file_path = os.path.join(self.data_dir, f"{name}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif format == 'excel':
            file_path = os.path.join(self.data_dir, f"{name}.xlsx")
            df.to_excel(file_path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"已保存数据: {file_path} ({len(df)} 行)")
        return file_path

    def save_plot(self, fig, name: str, dpi: int = 300):
        """
        保存图表

        Args:
            fig: matplotlib figure对象
            name: 文件名（不含扩展名）
            dpi: 分辨率
        """
        file_path = os.path.join(self.plots_dir, f"{name}.png")
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"已保存图表: {file_path}")
        return file_path

    def generate_report(self, results: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        生成Markdown报告

        Args:
            results: 分析结果字典，可包含:
                - returns: 板块涨跌幅DataFrame
                - correlation: 相关性矩阵DataFrame
                - lead_lag: 传导关系DataFrame
                - linkage: 联动强度DataFrame
            metadata: 元数据字典，包含分析参数
        """
        report_path = os.path.join(self.output_dir, 'report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            # 标题
            f.write("# 申万行业板块关系分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 分析周期
            if metadata:
                f.write("## 分析参数\n\n")
                f.write(f"- **数据范围**: {metadata.get('start_date', 'N/A')} ~ {metadata.get('end_date', 'N/A')}\n")
                f.write(f"- **分析层级**: {metadata.get('level', 'N/A')}\n")
                f.write(f"- **周期**: {metadata.get('period', 'N/A')}\n")
                f.write(f"- **计算方法**: {metadata.get('method', 'N/A')}\n\n")

            # 板块涨跌幅
            if 'returns' in results:
                df_returns = results['returns']
                f.write("## 板块涨跌幅统计\n\n")
                f.write(f"共计算 {df_returns['sector_code'].nunique()} 个板块，")
                f.write(f"{df_returns['trade_date'].nunique()} 个交易日。\n\n")

                # 涨跌幅排行
                avg_returns = df_returns.groupby('sector_code')['return'].mean().sort_values(ascending=False)
                f.write("### 平均涨跌幅排行（前10）\n\n")
                f.write("| 排名 | 板块代码 | 平均涨跌幅(%) |\n")
                f.write("|------|----------|---------------|\n")
                for i, (code, ret) in enumerate(avg_returns.head(10).items(), 1):
                    f.write(f"| {i} | {code} | {ret:.2f} |\n")
                f.write("\n")

            # 相关性分析
            if 'correlation' in results:
                corr_matrix = results['correlation']
                f.write("## 相关性分析\n\n")
                f.write(f"相关性矩阵: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}\n\n")

                # 找出最强相关性（非对角线）
                corr_values = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_values.append({
                            'sector_a': corr_matrix.index[i],
                            'sector_b': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
                top_corr = sorted(corr_values, key=lambda x: abs(x['correlation']), reverse=True)[:10]

                f.write("### 相关性最强的板块对（前10）\n\n")
                f.write("| 排名 | 板块A | 板块B | 相关系数 |\n")
                f.write("|------|-------|-------|----------|\n")
                for i, item in enumerate(top_corr, 1):
                    f.write(f"| {i} | {item['sector_a']} | {item['sector_b']} | {item['correlation']:.3f} |\n")
                f.write("\n")

            # 传导关系
            if 'lead_lag' in results:
                df_lead_lag = results['lead_lag']
                if len(df_lead_lag) > 0:
                    f.write("## 传导关系分析\n\n")
                    f.write(f"发现 {len(df_lead_lag)} 对传导关系。\n\n")

                    f.write("### 传导性最强的板块对（前10）\n\n")
                    f.write("| 排名 | 领涨板块 | 跟随板块 | 滞后天数 | 相关系数 | p值 |\n")
                    f.write("|------|----------|----------|----------|----------|-----|\n")
                    for i, row in df_lead_lag.head(10).iterrows():
                        f.write(f"| {i+1} | {row['sector_lead']} | {row['sector_lag']} | "
                               f"{row['lag_days']} | {row['correlation']:.3f} | {row['p_value']:.4f} |\n")
                    f.write("\n")

            # 联动强度
            if 'linkage' in results:
                df_linkage = results['linkage']
                if len(df_linkage) > 0:
                    f.write("## 联动强度分析\n\n")
                    f.write(f"发现 {len(df_linkage)} 对联动关系。\n\n")

                    f.write("### 联动最强的板块对（前10）\n\n")
                    f.write("| 排名 | 板块A | 板块B | Beta系数 | R² | 含义 |\n")
                    f.write("|------|-------|-------|----------|----|-----------|\n")
                    for i, row in df_linkage.head(10).iterrows():
                        meaning = f"A涨1%，B涨{row['beta']:.2f}%"
                        f.write(f"| {i+1} | {row['sector_a']} | {row['sector_b']} | "
                               f"{row['beta']:.3f} | {row['r_squared']:.3f} | {meaning} |\n")
                    f.write("\n")

            # 数据文件说明
            f.write("## 数据文件\n\n")
            f.write("本次分析生成的数据文件保存在 `data/` 目录下：\n\n")
            if os.path.exists(self.data_dir):
                for filename in sorted(os.listdir(self.data_dir)):
                    if filename.endswith(('.csv', '.xlsx')):
                        f.write(f"- `{filename}`\n")
            f.write("\n")

            # 图表说明
            if os.path.exists(self.plots_dir) and len(os.listdir(self.plots_dir)) > 0:
                f.write("## 可视化图表\n\n")
                f.write("本次分析生成的图表保存在 `plots/` 目录下：\n\n")
                for filename in sorted(os.listdir(self.plots_dir)):
                    if filename.endswith('.png'):
                        f.write(f"- `{filename}`\n")
                f.write("\n")

        logger.info(f"已生成报告: {report_path}")
        return report_path
