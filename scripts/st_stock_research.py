#!/usr/bin/env python3
"""
ST股票特征研究分析
===================
研究内容:
1. ST股识别与分类
2. ST效应分析（被ST前后走势、摘帽效应、波动特征）
3. 策略研究（摘帽博弈、风险规避、困境反转）
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os
import re

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
REPORT_DIR = "/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research"

# 确保报告目录存在
os.makedirs(REPORT_DIR, exist_ok=True)


class STStockResearch:
    """ST股票研究类"""

    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_content = []

    def add_section(self, title, content):
        """添加报告章节"""
        self.report_content.append(f"\n## {title}\n")
        self.report_content.append(content)

    def add_subsection(self, title, content):
        """添加子章节"""
        self.report_content.append(f"\n### {title}\n")
        self.report_content.append(content)

    def get_st_stocks(self):
        """获取当前所有ST股票"""
        query = """
        SELECT ts_code, name, industry, area, list_date, market, list_status
        FROM stock_basic
        WHERE name LIKE '%ST%'
        ORDER BY ts_code
        """
        return self.conn.execute(query).fetchdf()

    def get_all_stocks_with_history(self):
        """获取所有股票的历史数据，用于识别曾经ST过的股票"""
        # 由于没有专门的ST历史表，我们通过股票名称的变化来推断
        # 这里使用当前的ST股票列表
        return self.get_st_stocks()

    def classify_st_types(self, st_stocks):
        """ST原因分类"""
        # 根据名称前缀分类
        st_stocks['st_type'] = st_stocks['name'].apply(self._classify_st_type)
        return st_stocks

    def _classify_st_type(self, name):
        """根据股票名称分类ST类型"""
        if '*ST' in name:
            return '*ST（退市风险警示）'
        elif 'ST' in name:
            return 'ST（其他风险警示）'
        else:
            return '非ST'

    def analyze_st_by_industry(self, st_stocks):
        """按行业分析ST股票分布"""
        return st_stocks.groupby('industry').agg({
            'ts_code': 'count',
            'name': lambda x: ', '.join(x.head(3))
        }).rename(columns={'ts_code': '数量', 'name': '代表股票'}).sort_values('数量', ascending=False)

    def analyze_st_by_area(self, st_stocks):
        """按地区分析ST股票分布"""
        return st_stocks.groupby('area').agg({
            'ts_code': 'count'
        }).rename(columns={'ts_code': '数量'}).sort_values('数量', ascending=False)

    def analyze_st_by_market(self, st_stocks):
        """按市场分析ST股票分布"""
        return st_stocks.groupby('market').agg({
            'ts_code': 'count'
        }).rename(columns={'ts_code': '数量'}).sort_values('数量', ascending=False)

    def get_stock_daily_data(self, ts_codes, start_date=None, end_date=None):
        """获取股票日线数据"""
        codes_str = "', '".join(ts_codes)
        query = f"""
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               d.pre_close, d.pct_chg, d.vol, d.amount
        FROM daily d
        WHERE d.ts_code IN ('{codes_str}')
        """
        if start_date:
            query += f" AND d.trade_date >= '{start_date}'"
        if end_date:
            query += f" AND d.trade_date <= '{end_date}'"
        query += " ORDER BY d.ts_code, d.trade_date"
        return self.conn.execute(query).fetchdf()

    def analyze_st_volatility(self, st_stocks, period_days=252):
        """分析ST股票波动特征"""
        # 获取最近一年的数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y%m%d')

        st_codes = st_stocks['ts_code'].tolist()[:50]  # 取前50只分析
        daily_data = self.get_stock_daily_data(st_codes, start_date, end_date)

        if daily_data.empty:
            return pd.DataFrame()

        # 计算波动率统计
        volatility_stats = daily_data.groupby('ts_code').agg({
            'pct_chg': ['mean', 'std', 'min', 'max', 'count'],
            'vol': 'mean',
            'amount': 'mean'
        })
        volatility_stats.columns = ['日均涨跌幅', '涨跌幅标准差', '最大跌幅', '最大涨幅',
                                    '交易天数', '日均成交量', '日均成交额']

        # 计算涨停跌停次数
        daily_data['is_limit_up'] = daily_data['pct_chg'] >= 4.9  # ST股5%涨停
        daily_data['is_limit_down'] = daily_data['pct_chg'] <= -4.9

        limit_stats = daily_data.groupby('ts_code').agg({
            'is_limit_up': 'sum',
            'is_limit_down': 'sum'
        }).rename(columns={'is_limit_up': '涨停次数', 'is_limit_down': '跌停次数'})

        result = volatility_stats.join(limit_stats)
        return result

    def analyze_st_vs_market(self, st_stocks, period='2024'):
        """对比ST股票与大盘走势"""
        # 获取ST股票数据
        st_codes = st_stocks['ts_code'].tolist()[:30]
        start_date = f'{period}0101'
        end_date = f'{period}1231'

        st_daily = self.get_stock_daily_data(st_codes, start_date, end_date)

        # 获取上证指数数据
        index_query = f"""
        SELECT trade_date, close, pct_chg
        FROM index_daily
        WHERE ts_code = '000001.SH'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        index_data = self.conn.execute(index_query).fetchdf()

        return st_daily, index_data

    def calculate_st_premium(self):
        """计算ST股票的估值特征"""
        query = """
        SELECT sb.ts_code, sb.name, sb.industry,
               db.pe_ttm, db.pb, db.total_mv, db.circ_mv,
               db.turnover_rate, db.trade_date
        FROM stock_basic sb
        LEFT JOIN daily_basic db ON sb.ts_code = db.ts_code
        WHERE sb.name LIKE '%ST%'
        AND db.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
        """
        return self.conn.execute(query).fetchdf()

    def analyze_delisting_risk(self, st_stocks):
        """分析退市风险"""
        # *ST股票是退市风险警示股票
        star_st = st_stocks[st_stocks['name'].str.contains(r'\*ST', regex=True)]
        normal_st = st_stocks[~st_stocks['name'].str.contains(r'\*ST', regex=True)]

        return {
            '退市风险警示(*ST)': len(star_st),
            '其他风险警示(ST)': len(normal_st),
            '退市风险占比': f"{len(star_st) / len(st_stocks) * 100:.1f}%" if len(st_stocks) > 0 else "0%"
        }

    def generate_trading_strategy_analysis(self, st_stocks):
        """生成交易策略分析"""
        strategies = {
            '摘帽博弈策略': self._analyze_uncap_strategy(st_stocks),
            'ST股风险规避': self._analyze_risk_avoidance(),
            '困境反转策略': self._analyze_turnaround_strategy(st_stocks)
        }
        return strategies

    def _analyze_uncap_strategy(self, st_stocks):
        """摘帽博弈策略分析"""
        return """
**策略逻辑：**
- 在ST股票有望摘帽前布局，等待摘帽后的估值修复
- 核心是判断公司基本面是否真正改善

**选股标准：**
1. 财务指标改善：连续两个会计年度净利润为正
2. 净资产为正且每股净资产高于股票面值
3. 审计意见为标准无保留意见
4. 无其他导致ST的情形

**进场时机：**
- 年报公布前1-3个月（通常4月30日前公布）
- 密切关注业绩预告、快报
- 关注公司是否有资产重组、债务重组计划

**风险控制：**
- 严格控制仓位，单只ST股不超过总仓位5%
- 设置止损线（通常-15%至-20%）
- 避免参与连续亏损超过3年的ST股

**历史收益特征：**
- 摘帽成功：通常有20%-50%的短期涨幅
- 摘帽失败：可能面临退市风险，损失惨重
"""

    def _analyze_risk_avoidance(self):
        """ST股风险规避策略"""
        return """
**风险规避原则：**

1. **财务风险警示**
   - 连续亏损两年被ST
   - 净资产为负被*ST
   - 审计意见异常被ST

2. **应规避的ST股类型：**
   - 连续亏损3年以上
   - 涉及重大违规（财务造假、信息披露违规）
   - 主营业务严重萎缩
   - 资不抵债且无重组预期
   - 面临多起诉讼和债务纠纷

3. **识别风险信号：**
   - 股东大幅减持
   - 高管频繁变动
   - 审计机构更换
   - 业绩预告大幅下调
   - 被交易所问询函

4. **退市风险指标：**
   - *ST股票面临直接退市风险
   - 关注年报、半年报的盈亏情况
   - 净资产是否为正
"""

    def _analyze_turnaround_strategy(self, st_stocks):
        """困境反转策略"""
        return """
**策略核心：**
- 寻找基本面出现实质性改善的ST公司
- 关注资产重组、债务重整、新业务注入

**选股维度：**

1. **重组预期**
   - 国资入主：地方国资接盘的ST公司
   - 产业整合：被同行业优质企业收购
   - 借壳上市：有优质资产注入预期

2. **业务转型**
   - 剥离亏损业务
   - 切入新赛道（如新能源、半导体等热门行业）
   - 新管理团队带来的变革

3. **债务重整**
   - 债转股
   - 债务豁免
   - 资产出售偿债

**投资时机：**
- 重组公告后的调整期
- 定增预案公布后
- 业绩扭亏预期确立时

**案例特征：**
- 壳价值仍有吸引力
- 注册制背景下壳价值下降
- 需严格评估重组方实力
"""

    def create_visualizations(self, st_stocks, industry_dist, area_dist):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. ST类型分布
        ax1 = axes[0, 0]
        st_type_counts = st_stocks['st_type'].value_counts()
        colors = ['#ff6b6b', '#feca57']
        ax1.pie(st_type_counts.values, labels=st_type_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('ST股票类型分布', fontsize=14, fontweight='bold')

        # 2. 行业分布（Top 15）
        ax2 = axes[0, 1]
        top_industries = industry_dist.head(15)
        bars = ax2.barh(range(len(top_industries)), top_industries['数量'].values, color='#54a0ff')
        ax2.set_yticks(range(len(top_industries)))
        ax2.set_yticklabels(top_industries.index)
        ax2.set_xlabel('ST股票数量')
        ax2.set_title('ST股票行业分布 (Top 15)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        for i, v in enumerate(top_industries['数量'].values):
            ax2.text(v + 0.1, i, str(v), va='center')

        # 3. 地区分布（Top 15）
        ax3 = axes[1, 0]
        top_areas = area_dist.head(15)
        bars = ax3.barh(range(len(top_areas)), top_areas['数量'].values, color='#5f27cd')
        ax3.set_yticks(range(len(top_areas)))
        ax3.set_yticklabels(top_areas.index)
        ax3.set_xlabel('ST股票数量')
        ax3.set_title('ST股票地区分布 (Top 15)', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        for i, v in enumerate(top_areas['数量'].values):
            ax3.text(v + 0.1, i, str(v), va='center')

        # 4. 市场分布
        ax4 = axes[1, 1]
        market_counts = st_stocks.groupby('market').size()
        ax4.pie(market_counts.values, labels=market_counts.index, autopct='%1.1f%%',
                colors=['#00d2d3', '#ff9ff3', '#ffeaa7', '#dfe6e9'], startangle=90)
        ax4.set_title('ST股票市场板块分布', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}/st_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_volatility_chart(self, volatility_stats):
        """创建波动率图表"""
        if volatility_stats.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 涨跌幅标准差分布
        ax1 = axes[0, 0]
        ax1.hist(volatility_stats['涨跌幅标准差'].dropna(), bins=20, color='#3498db', edgecolor='black')
        ax1.set_xlabel('涨跌幅标准差 (%)')
        ax1.set_ylabel('股票数量')
        ax1.set_title('ST股票波动率分布', fontsize=12, fontweight='bold')
        ax1.axvline(volatility_stats['涨跌幅标准差'].mean(), color='red', linestyle='--',
                    label=f'均值: {volatility_stats["涨跌幅标准差"].mean():.2f}%')
        ax1.legend()

        # 2. 涨停跌停次数
        ax2 = axes[0, 1]
        x = range(min(20, len(volatility_stats)))
        sample = volatility_stats.head(20)
        width = 0.35
        ax2.bar([i - width/2 for i in x], sample['涨停次数'].values, width,
                label='涨停次数', color='#e74c3c')
        ax2.bar([i + width/2 for i in x], sample['跌停次数'].values, width,
                label='跌停次数', color='#27ae60')
        ax2.set_xlabel('股票')
        ax2.set_ylabel('次数')
        ax2.set_title('ST股票涨跌停分布 (样本)', fontsize=12, fontweight='bold')
        ax2.legend()

        # 3. 日均涨跌幅分布
        ax3 = axes[1, 0]
        ax3.hist(volatility_stats['日均涨跌幅'].dropna(), bins=20, color='#9b59b6', edgecolor='black')
        ax3.set_xlabel('日均涨跌幅 (%)')
        ax3.set_ylabel('股票数量')
        ax3.set_title('ST股票日均收益分布', fontsize=12, fontweight='bold')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(volatility_stats['日均涨跌幅'].mean(), color='red', linestyle='--',
                    label=f'均值: {volatility_stats["日均涨跌幅"].mean():.3f}%')
        ax3.legend()

        # 4. 最大涨跌幅散点图
        ax4 = axes[1, 1]
        ax4.scatter(volatility_stats['最大跌幅'], volatility_stats['最大涨幅'],
                   alpha=0.6, c='#e67e22', s=50)
        ax4.set_xlabel('最大跌幅 (%)')
        ax4.set_ylabel('最大涨幅 (%)')
        ax4.set_title('ST股票极端收益分布', fontsize=12, fontweight='bold')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}/st_volatility.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成完整研究报告"""
        print("=" * 60)
        print("ST股票特征研究分析")
        print("=" * 60)

        # 报告头部
        self.report_content.append("# ST股票特征研究报告\n")
        self.report_content.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.report_content.append(f"**数据来源：** Tushare-DuckDB\n")

        # ========== 1. ST股识别 ==========
        print("\n[1/5] 获取ST股票列表...")
        st_stocks = self.get_st_stocks()
        st_stocks = self.classify_st_types(st_stocks)

        # 基础统计
        total_listed = self.conn.execute(
            "SELECT COUNT(*) FROM stock_basic WHERE list_status = 'L'"
        ).fetchone()[0]

        st_summary = f"""
**基础统计：**
- 当前A股上市公司总数：{total_listed}
- 当前ST股票总数：{len(st_stocks)}
- ST股票占比：{len(st_stocks)/total_listed*100:.2f}%

**ST类型分布：**
"""
        st_type_counts = st_stocks['st_type'].value_counts()
        for st_type, count in st_type_counts.items():
            st_summary += f"- {st_type}：{count}只 ({count/len(st_stocks)*100:.1f}%)\n"

        self.add_section("一、ST股识别与分类", st_summary)

        # 行业分布
        print("[2/5] 分析行业分布...")
        industry_dist = self.analyze_st_by_industry(st_stocks)

        industry_content = "\n**行业分布 (Top 10)：**\n\n"
        industry_content += "| 行业 | 数量 | 代表股票 |\n"
        industry_content += "|------|------|----------|\n"
        for idx in industry_dist.head(10).index:
            row = industry_dist.loc[idx]
            industry_content += f"| {idx} | {row['数量']} | {row['代表股票'][:30]}... |\n"

        self.add_subsection("1.1 行业分布", industry_content)

        # 地区分布
        area_dist = self.analyze_st_by_area(st_stocks)

        area_content = "\n**地区分布 (Top 10)：**\n\n"
        area_content += "| 地区 | 数量 |\n"
        area_content += "|------|------|\n"
        for idx in area_dist.head(10).index:
            area_content += f"| {idx} | {area_dist.loc[idx, '数量']} |\n"

        self.add_subsection("1.2 地区分布", area_content)

        # 退市风险分析
        delisting_risk = self.analyze_delisting_risk(st_stocks)

        risk_content = "\n**退市风险统计：**\n"
        for key, value in delisting_risk.items():
            risk_content += f"- {key}：{value}\n"

        self.add_subsection("1.3 退市风险分布", risk_content)

        # 创建分布图表
        self.create_visualizations(st_stocks, industry_dist, area_dist)
        self.report_content.append(f"\n![ST股票分布图](st_distribution.png)\n")

        # ========== 2. ST效应分析 ==========
        print("[3/5] 分析ST股票波动特征...")

        volatility_stats = self.analyze_st_volatility(st_stocks)

        if not volatility_stats.empty:
            vol_content = f"""
**波动特征统计（近一年）：**
- 平均日涨跌幅标准差：{volatility_stats['涨跌幅标准差'].mean():.2f}%
- 平均日涨跌幅：{volatility_stats['日均涨跌幅'].mean():.3f}%
- 平均涨停次数：{volatility_stats['涨停次数'].mean():.1f}次
- 平均跌停次数：{volatility_stats['跌停次数'].mean():.1f}次

**极端收益特征：**
- 最大单日涨幅（样本中）：{volatility_stats['最大涨幅'].max():.2f}%
- 最大单日跌幅（样本中）：{volatility_stats['最大跌幅'].min():.2f}%

**波动率排名 (Top 5 最高波动)：**
"""
            top_volatile = volatility_stats.nlargest(5, '涨跌幅标准差')
            for ts_code in top_volatile.index:
                row = top_volatile.loc[ts_code]
                vol_content += f"- {ts_code}：标准差 {row['涨跌幅标准差']:.2f}%\n"

            self.add_section("二、ST效应分析", vol_content)

            # 创建波动率图表
            self.create_volatility_chart(volatility_stats)
            self.report_content.append(f"\n![ST股票波动特征](st_volatility.png)\n")
        else:
            self.add_section("二、ST效应分析", "\n*数据不足，无法进行波动分析*\n")

        # ST股票与涨跌停限制
        limit_content = """
### 2.1 ST股票交易规则特点

**涨跌幅限制：**
- ST股票涨跌幅限制为5%（普通股票为10%）
- *ST股票同样为5%涨跌幅限制
- 科创板、创业板ST股票仍为20%涨跌幅

**交易影响：**
1. 流动性较差，买卖差价大
2. 容易出现连续涨停/跌停
3. 散户参与难度大
4. 机构持仓比例通常较低

**信息披露要求：**
- 定期披露风险提示公告
- 重大事项需及时公告
- 年报审计要求更严格
"""
        self.add_subsection("2.1 ST股票交易规则特点", limit_content)

        # ========== 3. 策略研究 ==========
        print("[4/5] 生成策略研究...")

        strategies = self.generate_trading_strategy_analysis(st_stocks)

        self.add_section("三、策略研究", "")

        for strategy_name, strategy_content in strategies.items():
            self.add_subsection(f"3.{list(strategies.keys()).index(strategy_name)+1} {strategy_name}",
                              strategy_content)

        # 添加量化指标建议
        quant_content = """
### 3.4 量化筛选指标

**摘帽预期筛选指标：**
```
1. 最近两个季度净利润 > 0
2. 净资产 > 0
3. 营业收入同比增长 > 0
4. 无审计异常意见
5. 无重大诉讼或违规
```

**困境反转筛选指标：**
```
1. PE < 行业平均（对于盈利公司）
2. PB < 1.5（相对低估）
3. 大股东持股比例稳定
4. 近期有增持公告
5. 有重组预期或国资入主
```

**风险规避清单：**
```
避免的特征：
- 连续3年以上亏损
- 审计意见为保留/否定/无法表示
- 被证监会立案调查
- 资不抵债（净资产为负）
- 主营业务停滞
```
"""
        self.report_content.append(quant_content)

        # ========== 4. 风险提示 ==========
        risk_warning = """
## 四、风险提示

**ST股票投资的主要风险：**

1. **退市风险**
   - *ST股票随时可能被暂停上市或终止上市
   - 退市后股票价值可能归零
   - 2020年后退市新规加速退市进程

2. **流动性风险**
   - ST股票成交量通常较小
   - 连续跌停时无法卖出
   - 买卖价差较大

3. **信息不对称风险**
   - ST公司信息披露可能不完整
   - 财务数据可信度存疑
   - 重组信息内幕交易风险

4. **估值风险**
   - 传统估值方法可能失效
   - 壳价值难以准确评估
   - 注册制下壳价值持续下降

5. **时间成本**
   - 困境反转周期可能很长
   - 资金占用成本高
   - 机会成本大

**投资建议：**
- ST股票属于高风险投资，仅适合风险承受能力强的投资者
- 严格控制仓位，单只ST股不超过总仓位5%
- 做好充分的基本面研究
- 设置严格的止损纪律
"""
        self.report_content.append(risk_warning)

        # ========== 5. ST股票清单 ==========
        print("[5/5] 生成ST股票清单...")

        st_list_content = "\n## 五、当前ST股票清单\n\n"
        st_list_content += "| 代码 | 名称 | 行业 | 地区 | 市场 |\n"
        st_list_content += "|------|------|------|------|------|\n"

        for _, row in st_stocks.head(50).iterrows():
            st_list_content += f"| {row['ts_code']} | {row['name']} | {row['industry']} | {row['area']} | {row['market']} |\n"

        if len(st_stocks) > 50:
            st_list_content += f"\n*（共{len(st_stocks)}只ST股票，此处仅展示前50只）*\n"

        self.report_content.append(st_list_content)

        # 保存报告
        report_path = f'{REPORT_DIR}/st_stock_research_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))

        print(f"\n报告已保存至：{report_path}")
        print(f"图表已保存至：{REPORT_DIR}/")

        return st_stocks

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def main():
    """主函数"""
    research = STStockResearch(DB_PATH)
    try:
        st_stocks = research.generate_report()
        print("\n" + "=" * 60)
        print("研究报告生成完成！")
        print("=" * 60)
    finally:
        research.close()


if __name__ == "__main__":
    main()
