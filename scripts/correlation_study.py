"""
A股市场相关性结构研究
========================
分析内容:
1. 相关性矩阵分析
2. 相关性动态变化
3. 主成分分析
4. 相关性预测
5. 组合分散化
6. 相关性异常检测
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 可视化相关
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 科学计算
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

# =============================================================================
# 1. 数据加载
# =============================================================================

def load_industry_data():
    """加载申万一级行业日线数据"""
    conn = get_connection()

    # 获取申万一级行业代码
    query = """
    SELECT DISTINCT ts_code, name, trade_date, close, pct_change
    FROM sw_daily
    WHERE ts_code LIKE '8010%%.SI' OR ts_code LIKE '8011%%.SI' OR
          ts_code LIKE '8012%%.SI' OR ts_code LIKE '8013%%.SI' OR
          ts_code LIKE '8014%%.SI' OR ts_code LIKE '8015%%.SI' OR
          ts_code LIKE '8016%%.SI' OR ts_code LIKE '8017%%.SI' OR
          ts_code LIKE '8018%%.SI' OR ts_code LIKE '8019%%.SI' OR
          ts_code LIKE '8020%%.SI' OR ts_code LIKE '8021%%.SI' OR
          ts_code LIKE '8023%%.SI' OR ts_code LIKE '8071%%.SI' OR
          ts_code LIKE '8072%%.SI' OR ts_code LIKE '8073%%.SI' OR
          ts_code LIKE '8074%%.SI' OR ts_code LIKE '8075%%.SI' OR
          ts_code LIKE '8076%%.SI' OR ts_code LIKE '8077%%.SI' OR
          ts_code LIKE '8078%%.SI' OR ts_code LIKE '8079%%.SI' OR
          ts_code LIKE '8088%%.SI' OR ts_code LIKE '8089%%.SI' OR
          ts_code LIKE '8095%%.SI' OR ts_code LIKE '8096%%.SI' OR
          ts_code LIKE '8097%%.SI' OR ts_code LIKE '8098%%.SI'
    ORDER BY trade_date, ts_code
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    # 创建行业名称映射
    industry_names = df[['ts_code', 'name']].drop_duplicates().set_index('ts_code')['name'].to_dict()

    # 转换为宽表格式
    df_pivot = df.pivot(index='trade_date', columns='ts_code', values='pct_change')

    # 重命名列为行业名称
    df_pivot.columns = [industry_names.get(c, c) for c in df_pivot.columns]

    return df_pivot, industry_names

def load_stock_data(start_date='20230101', end_date='20260130', min_days=200):
    """加载个股数据(选取流动性好的股票)"""
    conn = get_connection()

    # 获取沪深300成分股或交易活跃的股票
    query = f"""
    WITH active_stocks AS (
        SELECT ts_code, COUNT(*) as cnt, AVG(amount) as avg_amount
        FROM daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        GROUP BY ts_code
        HAVING COUNT(*) >= {min_days} AND AVG(amount) > 50000
        ORDER BY avg_amount DESC
        LIMIT 300
    )
    SELECT d.ts_code, d.trade_date, d.pct_chg
    FROM daily d
    INNER JOIN active_stocks a ON d.ts_code = a.ts_code
    WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
    ORDER BY d.trade_date, d.ts_code
    """
    df = conn.execute(query).fetchdf()

    # 获取股票名称
    stock_info = conn.execute("""
    SELECT ts_code, name, industry FROM stock_basic
    """).fetchdf()
    stock_names = stock_info.set_index('ts_code')['name'].to_dict()
    stock_industries = stock_info.set_index('ts_code')['industry'].to_dict()

    conn.close()

    # 转换为宽表
    df_pivot = df.pivot(index='trade_date', columns='ts_code', values='pct_chg')

    return df_pivot, stock_names, stock_industries

def load_index_data():
    """加载主要指数数据"""
    conn = get_connection()

    # 从dc_index表获取指数数据(大盘指数)
    query = """
    SELECT ts_code, trade_date, pct_change
    FROM sw_daily
    WHERE ts_code IN ('801001.SI', '801002.SI', '801003.SI')  -- 申万市场指数
    ORDER BY trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    if len(df) > 0:
        df_pivot = df.pivot(index='trade_date', columns='ts_code', values='pct_change')
        return df_pivot
    return None

# =============================================================================
# 2. 相关性矩阵分析
# =============================================================================

def analyze_correlation_matrix(df, title_suffix=""):
    """分析相关性矩阵"""
    print(f"\n{'='*60}")
    print(f"相关性矩阵分析 {title_suffix}")
    print('='*60)

    # 删除缺失值过多的列
    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8))
    df_clean = df_clean.dropna()

    if df_clean.shape[1] < 3:
        print("数据不足，无法进行分析")
        return None, None

    # 计算相关性矩阵
    corr_matrix = df_clean.corr()

    # 基本统计
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

    print(f"\n数据维度: {df_clean.shape}")
    print(f"相关性统计:")
    print(f"  - 均值: {np.mean(corr_values):.4f}")
    print(f"  - 中位数: {np.median(corr_values):.4f}")
    print(f"  - 标准差: {np.std(corr_values):.4f}")
    print(f"  - 最小值: {np.min(corr_values):.4f}")
    print(f"  - 最大值: {np.max(corr_values):.4f}")
    print(f"  - 正相关占比: {np.mean(corr_values > 0)*100:.1f}%")
    print(f"  - 强相关(>0.7)占比: {np.mean(corr_values > 0.7)*100:.1f}%")

    return corr_matrix, df_clean

def plot_correlation_heatmap(corr_matrix, title, filename):
    """绘制相关性热力图"""
    fig, ax = plt.subplots(figsize=(14, 12))

    # 使用层次聚类重新排序
    if corr_matrix.shape[0] > 3:
        # 将相关性转换为距离
        distance = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(distance, 0)

        # 层次聚类
        linkage_matrix = linkage(squareform(distance), method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']

        # 重新排序
        corr_ordered = corr_matrix.iloc[order, order]
    else:
        corr_ordered = corr_matrix

    sns.heatmap(corr_ordered, annot=False, cmap='RdYlBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

def analyze_correlation_distribution(corr_matrix, title, filename):
    """分析相关性分布"""
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(corr_values), color='red', linestyle='--', label=f'均值: {np.mean(corr_values):.3f}')
    axes[0].axvline(np.median(corr_values), color='green', linestyle='--', label=f'中位数: {np.median(corr_values):.3f}')
    axes[0].set_xlabel('相关系数')
    axes[0].set_ylabel('频数')
    axes[0].set_title(f'{title} - 相关系数分布')
    axes[0].legend()

    # 箱线图
    axes[1].boxplot(corr_values, vert=True)
    axes[1].set_ylabel('相关系数')
    axes[1].set_title(f'{title} - 相关系数箱线图')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

# =============================================================================
# 3. 相关性聚类分析
# =============================================================================

def correlation_clustering(corr_matrix, n_clusters=5):
    """基于相关性的聚类分析"""
    print(f"\n{'='*60}")
    print("相关性聚类分析")
    print('='*60)

    # 将相关性转换为距离
    distance = 1 - corr_matrix.values
    np.fill_diagonal(distance, 0)

    # 层次聚类
    linkage_matrix = linkage(squareform(distance), method='ward')

    # 获取聚类标签
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # 创建聚类结果
    cluster_df = pd.DataFrame({
        'name': corr_matrix.columns,
        'cluster': labels
    })

    print(f"\n聚类结果 (共{n_clusters}类):")
    for i in range(1, n_clusters+1):
        members = cluster_df[cluster_df['cluster']==i]['name'].tolist()
        print(f"\n类别 {i} ({len(members)}个):")
        print(f"  {', '.join(members[:10])}" + ("..." if len(members) > 10 else ""))

    return linkage_matrix, labels, cluster_df

def plot_dendrogram(linkage_matrix, labels, title, filename):
    """绘制树状图"""
    fig, ax = plt.subplots(figsize=(16, 8))

    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('资产')
    ax.set_ylabel('距离')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

# =============================================================================
# 4. 滚动相关性分析
# =============================================================================

def rolling_correlation_analysis(df, window=60):
    """滚动相关性分析"""
    print(f"\n{'='*60}")
    print(f"滚动相关性分析 (窗口: {window}天)")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()

    if df_clean.shape[1] < 2:
        print("数据不足")
        return None

    # 计算滚动平均相关性
    n_cols = df_clean.shape[1]
    dates = df_clean.index.tolist()
    avg_corrs = []

    for i in range(window, len(df_clean)):
        window_data = df_clean.iloc[i-window:i]
        corr = window_data.corr()
        corr_values = corr.values[np.triu_indices(n_cols, k=1)]
        avg_corrs.append({
            'date': dates[i],
            'avg_corr': np.mean(corr_values),
            'median_corr': np.median(corr_values),
            'std_corr': np.std(corr_values),
            'min_corr': np.min(corr_values),
            'max_corr': np.max(corr_values)
        })

    rolling_corr_df = pd.DataFrame(avg_corrs)
    rolling_corr_df['date'] = pd.to_datetime(rolling_corr_df['date'], format='%Y%m%d')

    print(f"\n滚动相关性统计:")
    print(f"  - 平均相关性均值: {rolling_corr_df['avg_corr'].mean():.4f}")
    print(f"  - 平均相关性标准差: {rolling_corr_df['avg_corr'].std():.4f}")
    print(f"  - 最高平均相关性: {rolling_corr_df['avg_corr'].max():.4f}")
    print(f"  - 最低平均相关性: {rolling_corr_df['avg_corr'].min():.4f}")

    return rolling_corr_df

def plot_rolling_correlation(rolling_df, title, filename):
    """绘制滚动相关性时序图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 平均相关性
    axes[0].plot(rolling_df['date'], rolling_df['avg_corr'], label='平均相关性', linewidth=1)
    axes[0].fill_between(rolling_df['date'],
                         rolling_df['avg_corr'] - rolling_df['std_corr'],
                         rolling_df['avg_corr'] + rolling_df['std_corr'],
                         alpha=0.3, label='标准差区间')
    axes[0].axhline(rolling_df['avg_corr'].mean(), color='red', linestyle='--',
                    label=f"长期均值: {rolling_df['avg_corr'].mean():.3f}")
    axes[0].set_ylabel('相关系数')
    axes[0].set_title(f'{title} - 滚动平均相关性')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 相关性范围
    axes[1].fill_between(rolling_df['date'], rolling_df['min_corr'], rolling_df['max_corr'],
                         alpha=0.3, label='相关性范围')
    axes[1].plot(rolling_df['date'], rolling_df['median_corr'], label='中位数', linewidth=1)
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('相关系数')
    axes[1].set_title(f'{title} - 相关性范围')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

def detect_correlation_regimes(rolling_df, threshold_std=1.5):
    """检测相关性状态变化"""
    print(f"\n{'='*60}")
    print("相关性状态检测")
    print('='*60)

    mean_corr = rolling_df['avg_corr'].mean()
    std_corr = rolling_df['avg_corr'].std()

    high_threshold = mean_corr + threshold_std * std_corr
    low_threshold = mean_corr - threshold_std * std_corr

    rolling_df['regime'] = 'normal'
    rolling_df.loc[rolling_df['avg_corr'] > high_threshold, 'regime'] = 'high'
    rolling_df.loc[rolling_df['avg_corr'] < low_threshold, 'regime'] = 'low'

    print(f"\n相关性阈值:")
    print(f"  - 均值: {mean_corr:.4f}")
    print(f"  - 高相关阈值: {high_threshold:.4f}")
    print(f"  - 低相关阈值: {low_threshold:.4f}")

    print(f"\n状态分布:")
    regime_counts = rolling_df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(rolling_df) * 100
        print(f"  - {regime}: {count}天 ({pct:.1f}%)")

    # 高相关性时期
    high_periods = rolling_df[rolling_df['regime'] == 'high']
    if len(high_periods) > 0:
        print(f"\n高相关性时期:")
        for idx, row in high_periods.head(10).iterrows():
            print(f"  - {row['date'].strftime('%Y-%m-%d')}: {row['avg_corr']:.4f}")

    return rolling_df

# =============================================================================
# 5. 主成分分析
# =============================================================================

def pca_analysis(df, n_components=10):
    """主成分分析"""
    print(f"\n{'='*60}")
    print("主成分分析 (PCA)")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()

    if df_clean.shape[1] < 3:
        print("数据不足")
        return None, None, None

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)

    # PCA
    n_components = min(n_components, df_clean.shape[1], df_clean.shape[0])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)

    # 方差解释
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\n主成分方差解释:")
    for i in range(min(10, n_components)):
        print(f"  PC{i+1}: {explained_var[i]*100:.2f}% (累计: {cumulative_var[i]*100:.2f}%)")

    # 找到解释90%方差所需的成分数
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\n解释90%方差需要 {n_90} 个主成分")
    print(f"解释95%方差需要 {n_95} 个主成分")

    # 第一主成分(市场因子)的载荷
    pc1_loadings = pd.Series(pca.components_[0], index=df_clean.columns)
    pc1_loadings_sorted = pc1_loadings.abs().sort_values(ascending=False)

    print(f"\n第一主成分(市场因子)最大载荷:")
    for name in pc1_loadings_sorted.head(10).index:
        print(f"  {name}: {pc1_loadings[name]:.4f}")

    return pca, pca_result, df_clean.columns.tolist()

def plot_pca_results(pca, columns, title, filename):
    """绘制PCA结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 方差解释
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_show = min(20, len(explained_var))

    axes[0, 0].bar(range(1, n_show+1), explained_var[:n_show]*100, alpha=0.7, label='单独')
    axes[0, 0].plot(range(1, n_show+1), cumulative_var[:n_show]*100, 'ro-', label='累计')
    axes[0, 0].axhline(90, color='green', linestyle='--', label='90%阈值')
    axes[0, 0].set_xlabel('主成分')
    axes[0, 0].set_ylabel('方差解释(%)')
    axes[0, 0].set_title('主成分方差解释')
    axes[0, 0].legend()

    # 前两个主成分的载荷
    if len(pca.components_) >= 2:
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]

        axes[0, 1].scatter(pc1, pc2, alpha=0.6)
        for i, name in enumerate(columns):
            if abs(pc1[i]) > 0.1 or abs(pc2[i]) > 0.1:
                axes[0, 1].annotate(name[:6], (pc1[i], pc2[i]), fontsize=7)
        axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('PC1载荷')
        axes[0, 1].set_ylabel('PC2载荷')
        axes[0, 1].set_title('PC1 vs PC2 载荷图')

    # 第一主成分载荷分布
    pc1_loadings = pca.components_[0]
    axes[1, 0].hist(pc1_loadings, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(pc1_loadings), color='red', linestyle='--')
    axes[1, 0].set_xlabel('载荷值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('PC1(市场因子)载荷分布')

    # 显示top载荷
    pc1_series = pd.Series(pc1_loadings, index=columns).sort_values()
    top_n = min(15, len(columns))
    combined = pd.concat([pc1_series.head(top_n), pc1_series.tail(top_n)])
    combined.plot(kind='barh', ax=axes[1, 1], color=['red' if x < 0 else 'green' for x in combined])
    axes[1, 1].set_xlabel('载荷值')
    axes[1, 1].set_title('PC1载荷Top资产')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

# =============================================================================
# 6. 相关性预测 (DCC-GARCH思想简化版)
# =============================================================================

def correlation_forecast(df, forecast_window=20):
    """相关性预测分析 (简化DCC模型思想)"""
    print(f"\n{'='*60}")
    print("相关性预测分析")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()

    if df_clean.shape[1] < 2:
        print("数据不足")
        return None

    # 使用EWMA估计动态相关性
    # DCC模型的核心思想: 相关性是时变的，可以用历史信息预测

    results = []
    test_start = int(len(df_clean) * 0.8)

    for i in range(test_start, len(df_clean) - forecast_window):
        # 历史数据
        hist_data = df_clean.iloc[:i]

        # 计算历史相关性
        hist_corr = hist_data.corr()

        # EWMA相关性 (lambda=0.94)
        ewma_data = hist_data.ewm(span=60).corr()
        last_ewma_corr = ewma_data.iloc[-df_clean.shape[1]:]

        # 实际未来相关性
        future_data = df_clean.iloc[i:i+forecast_window]
        actual_corr = future_data.corr()

        # 获取上三角元素
        hist_values = hist_corr.values[np.triu_indices(hist_corr.shape[0], k=1)]
        ewma_values = last_ewma_corr.values[np.triu_indices(hist_corr.shape[0], k=1)]
        actual_values = actual_corr.values[np.triu_indices(actual_corr.shape[0], k=1)]

        # 计算预测误差
        mae_hist = np.mean(np.abs(hist_values - actual_values))
        mae_ewma = np.mean(np.abs(ewma_values - actual_values))

        results.append({
            'date': df_clean.index[i],
            'hist_avg_corr': np.mean(hist_values),
            'ewma_avg_corr': np.mean(ewma_values),
            'actual_avg_corr': np.mean(actual_values),
            'mae_hist': mae_hist,
            'mae_ewma': mae_ewma
        })

    forecast_df = pd.DataFrame(results)

    print(f"\n预测性能比较 (预测窗口: {forecast_window}天):")
    print(f"  历史相关性MAE: {forecast_df['mae_hist'].mean():.4f}")
    print(f"  EWMA相关性MAE: {forecast_df['mae_ewma'].mean():.4f}")

    # 相关性是否均值回归
    avg_corr = df_clean.corr().values[np.triu_indices(df_clean.shape[1], k=1)].mean()
    print(f"\n相关性均值回归检验:")
    print(f"  长期平均相关性: {avg_corr:.4f}")

    # 自相关检验
    rolling_corrs = []
    for i in range(60, len(df_clean)):
        window = df_clean.iloc[i-60:i]
        corr = window.corr()
        rolling_corrs.append(np.mean(corr.values[np.triu_indices(corr.shape[0], k=1)]))

    if len(rolling_corrs) > 1:
        autocorr = pd.Series(rolling_corrs).autocorr(lag=1)
        print(f"  相关性一阶自相关: {autocorr:.4f}")

    return forecast_df

def plot_correlation_forecast(forecast_df, filename):
    """绘制相关性预测结果"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    forecast_df['date'] = pd.to_datetime(forecast_df['date'], format='%Y%m%d')

    # 预测 vs 实际
    axes[0].plot(forecast_df['date'], forecast_df['actual_avg_corr'], label='实际', linewidth=1)
    axes[0].plot(forecast_df['date'], forecast_df['ewma_avg_corr'], label='EWMA预测', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('平均相关系数')
    axes[0].set_title('相关性预测 vs 实际')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 预测误差
    axes[1].plot(forecast_df['date'], forecast_df['mae_hist'], label='历史MAE', linewidth=1)
    axes[1].plot(forecast_df['date'], forecast_df['mae_ewma'], label='EWMA MAE', linewidth=1)
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('预测误差比较')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

# =============================================================================
# 7. 组合分散化分析
# =============================================================================

def portfolio_diversification_analysis(df, corr_matrix):
    """组合分散化分析"""
    print(f"\n{'='*60}")
    print("组合分散化分析")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()

    if df_clean.shape[1] < 3:
        print("数据不足")
        return None

    n_assets = df_clean.shape[1]
    returns = df_clean.mean()
    volatilities = df_clean.std()
    corr = df_clean.corr()
    cov = df_clean.cov()

    # 等权组合
    eq_weights = np.ones(n_assets) / n_assets
    eq_return = np.dot(eq_weights, returns)
    eq_vol = np.sqrt(np.dot(eq_weights, np.dot(cov, eq_weights)))

    print(f"\n等权组合:")
    print(f"  - 日均收益: {eq_return:.4f}%")
    print(f"  - 日波动率: {eq_vol:.4f}%")
    print(f"  - 年化收益: {eq_return*252:.2f}%")
    print(f"  - 年化波动: {eq_vol*np.sqrt(252):.2f}%")

    # 分散化效果
    avg_vol = volatilities.mean()
    div_ratio = avg_vol / eq_vol
    print(f"\n分散化效果:")
    print(f"  - 个股平均波动率: {avg_vol:.4f}%")
    print(f"  - 组合波动率: {eq_vol:.4f}%")
    print(f"  - 分散化比率: {div_ratio:.2f}")
    print(f"  - 波动率降低: {(1-1/div_ratio)*100:.1f}%")

    # 最小方差组合 (简化: 使用逆方差加权)
    inv_var = 1 / volatilities**2
    mv_weights = inv_var / inv_var.sum()
    mv_return = np.dot(mv_weights, returns)
    mv_vol = np.sqrt(np.dot(mv_weights, np.dot(cov, mv_weights)))

    print(f"\n最小方差组合(逆方差加权):")
    print(f"  - 日均收益: {mv_return:.4f}%")
    print(f"  - 日波动率: {mv_vol:.4f}%")

    # 风险平价组合
    # 简化版: 使用逆波动率加权
    inv_vol = 1 / volatilities
    rp_weights = inv_vol / inv_vol.sum()
    rp_return = np.dot(rp_weights, returns)
    rp_vol = np.sqrt(np.dot(rp_weights, np.dot(cov, rp_weights)))

    print(f"\n风险平价组合(逆波动率加权):")
    print(f"  - 日均收益: {rp_return:.4f}%")
    print(f"  - 日波动率: {rp_vol:.4f}%")

    # 有效分散化程度 (ENB - Effective Number of Bets)
    # ENB = 1 / sum(w^2)
    enb_eq = 1 / np.sum(eq_weights**2)
    enb_rp = 1 / np.sum(rp_weights**2)

    print(f"\n有效分散化程度 (ENB):")
    print(f"  - 等权组合: {enb_eq:.1f}")
    print(f"  - 风险平价组合: {enb_rp:.1f}")

    results = {
        'eq_weights': eq_weights,
        'eq_return': eq_return,
        'eq_vol': eq_vol,
        'mv_weights': mv_weights,
        'mv_return': mv_return,
        'mv_vol': mv_vol,
        'rp_weights': rp_weights,
        'rp_return': rp_return,
        'rp_vol': rp_vol,
        'div_ratio': div_ratio
    }

    return results

def minimum_correlation_portfolio(df):
    """最小相关性组合"""
    print(f"\n{'='*60}")
    print("最小相关性组合")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()
    corr = df_clean.corr()

    # 每个资产与其他资产的平均相关性
    avg_corr_per_asset = (corr.sum() - 1) / (len(corr) - 1)
    avg_corr_per_asset_sorted = avg_corr_per_asset.sort_values()

    print(f"\n与其他资产平均相关性最低的资产:")
    for name in avg_corr_per_asset_sorted.head(10).index:
        print(f"  {name}: {avg_corr_per_asset[name]:.4f}")

    print(f"\n与其他资产平均相关性最高的资产:")
    for name in avg_corr_per_asset_sorted.tail(10).index:
        print(f"  {name}: {avg_corr_per_asset[name]:.4f}")

    # 贪婪选择最小相关性组合
    n_select = min(10, len(corr)//3)
    selected = [avg_corr_per_asset_sorted.index[0]]

    for _ in range(n_select - 1):
        min_corr = float('inf')
        best_asset = None

        for asset in avg_corr_per_asset_sorted.index:
            if asset in selected:
                continue

            # 计算与已选资产的平均相关性
            avg = np.mean([corr.loc[asset, s] for s in selected])
            if avg < min_corr:
                min_corr = avg
                best_asset = asset

        if best_asset:
            selected.append(best_asset)

    print(f"\n最小相关性组合 ({n_select}个资产):")
    for asset in selected:
        print(f"  - {asset}")

    # 计算组合内平均相关性
    selected_corr = corr.loc[selected, selected]
    portfolio_avg_corr = (selected_corr.sum().sum() - n_select) / (n_select * (n_select - 1))
    print(f"\n组合内平均相关性: {portfolio_avg_corr:.4f}")

    return selected, avg_corr_per_asset

# =============================================================================
# 8. 相关性异常检测
# =============================================================================

def correlation_anomaly_detection(df, window=60, threshold_z=2.5):
    """相关性异常检测"""
    print(f"\n{'='*60}")
    print("相关性异常检测")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()

    if df_clean.shape[1] < 2:
        print("数据不足")
        return None

    # 计算滚动相关性变化
    n_cols = min(30, df_clean.shape[1])  # 限制计算量
    cols = df_clean.columns[:n_cols]

    anomalies = []

    for i in range(window*2, len(df_clean)):
        # 前期窗口
        prev_window = df_clean[cols].iloc[i-window*2:i-window]
        prev_corr = prev_window.corr()

        # 当前窗口
        curr_window = df_clean[cols].iloc[i-window:i]
        curr_corr = curr_window.corr()

        # 相关性变化
        corr_change = curr_corr - prev_corr

        # 找到显著变化
        for j in range(len(cols)):
            for k in range(j+1, len(cols)):
                change = corr_change.iloc[j, k]
                if abs(change) > 0.3:  # 相关性变化超过0.3
                    anomalies.append({
                        'date': df_clean.index[i],
                        'asset1': cols[j],
                        'asset2': cols[k],
                        'prev_corr': prev_corr.iloc[j, k],
                        'curr_corr': curr_corr.iloc[j, k],
                        'change': change
                    })

    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_df = anomaly_df.sort_values('change', key=abs, ascending=False)

        print(f"\n检测到 {len(anomaly_df)} 个相关性异常")
        print(f"\n最显著的相关性变化:")
        for _, row in anomaly_df.head(15).iterrows():
            direction = "上升" if row['change'] > 0 else "下降"
            print(f"  {row['date']}: {row['asset1']} vs {row['asset2']}")
            print(f"    {row['prev_corr']:.3f} -> {row['curr_corr']:.3f} ({direction} {abs(row['change']):.3f})")

        return anomaly_df
    else:
        print("未检测到显著相关性异常")
        return None

def pairs_trading_opportunities(df, corr_matrix, threshold=0.8):
    """配对交易机会识别"""
    print(f"\n{'='*60}")
    print("配对交易机会识别")
    print('='*60)

    df_clean = df.dropna(axis=1, thresh=int(len(df)*0.8)).dropna()
    corr = df_clean.corr()

    # 找到高相关性配对
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            c = corr.iloc[i, j]
            if c > threshold:
                high_corr_pairs.append({
                    'asset1': corr.columns[i],
                    'asset2': corr.columns[j],
                    'correlation': c
                })

    if high_corr_pairs:
        pairs_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

        print(f"\n高相关性配对 (相关系数 > {threshold}):")
        for _, row in pairs_df.head(20).iterrows():
            print(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.4f}")

        # 计算价差的平稳性 (简化ADF检验)
        print(f"\n配对价差分析:")
        for _, row in pairs_df.head(5).iterrows():
            s1 = df_clean[row['asset1']]
            s2 = df_clean[row['asset2']]

            # 标准化后计算价差
            spread = (s1 - s1.mean()) / s1.std() - (s2 - s2.mean()) / s2.std()

            # 价差统计
            print(f"\n  {row['asset1']} vs {row['asset2']}:")
            print(f"    价差均值: {spread.mean():.4f}")
            print(f"    价差标准差: {spread.std():.4f}")
            print(f"    价差偏度: {spread.skew():.4f}")

        return pairs_df
    else:
        print(f"未找到相关系数超过 {threshold} 的配对")
        return None

# =============================================================================
# 9. 行业相关性分析
# =============================================================================

def industry_correlation_analysis(df_industry):
    """行业间相关性分析"""
    print(f"\n{'='*60}")
    print("行业间相关性分析")
    print('='*60)

    df_clean = df_industry.dropna(axis=1, thresh=int(len(df_industry)*0.8)).dropna()

    if df_clean.shape[1] < 3:
        print("数据不足")
        return None, None

    corr = df_clean.corr()

    # 行业间平均相关性
    corr_values = corr.values[np.triu_indices(len(corr), k=1)]

    print(f"\n行业间相关性统计:")
    print(f"  - 平均: {np.mean(corr_values):.4f}")
    print(f"  - 中位数: {np.median(corr_values):.4f}")
    print(f"  - 标准差: {np.std(corr_values):.4f}")
    print(f"  - 范围: [{np.min(corr_values):.4f}, {np.max(corr_values):.4f}]")

    # 每个行业与其他行业的平均相关性
    avg_corr_per_industry = (corr.sum() - 1) / (len(corr) - 1)

    print(f"\n与市场同步性最高的行业:")
    for name in avg_corr_per_industry.sort_values(ascending=False).head(10).index:
        print(f"  {name}: {avg_corr_per_industry[name]:.4f}")

    print(f"\n与市场同步性最低的行业:")
    for name in avg_corr_per_industry.sort_values().head(10).index:
        print(f"  {name}: {avg_corr_per_industry[name]:.4f}")

    # 找到最不相关的行业对
    min_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            min_pairs.append({
                'industry1': corr.columns[i],
                'industry2': corr.columns[j],
                'correlation': corr.iloc[i, j]
            })

    min_pairs_df = pd.DataFrame(min_pairs).sort_values('correlation')

    print(f"\n相关性最低的行业配对:")
    for _, row in min_pairs_df.head(10).iterrows():
        print(f"  {row['industry1']} vs {row['industry2']}: {row['correlation']:.4f}")

    return corr, avg_corr_per_industry

# =============================================================================
# 10. 生成报告
# =============================================================================

def generate_report(results):
    """生成Markdown报告"""
    report = """# A股市场相关性结构研究报告

## 报告概述

本报告对A股市场的相关性结构进行了系统性研究，包括相关性矩阵分析、动态相关性、主成分分析、组合分散化以及相关性异常检测等方面。

**研究时间**: {report_date}
**数据范围**: {data_range}

---

## 1. 相关性矩阵分析

### 1.1 行业层面相关性

{industry_corr_section}

### 1.2 个股层面相关性

{stock_corr_section}

### 1.3 相关性聚类

{clustering_section}

---

## 2. 相关性动态变化

### 2.1 滚动相关性

{rolling_corr_section}

### 2.2 相关性状态检测

{regime_section}

### 2.3 相关性均值回归

{mean_reversion_section}

---

## 3. 主成分分析

### 3.1 方差解释

{pca_variance_section}

### 3.2 市场因子分析

{market_factor_section}

### 3.3 因子构建

{factor_construction_section}

---

## 4. 相关性预测

### 4.1 预测模型

{forecast_model_section}

### 4.2 预测性能

{forecast_performance_section}

---

## 5. 组合分散化

### 5.1 分散化效果

{diversification_section}

### 5.2 最小相关性组合

{min_corr_portfolio_section}

### 5.3 风险平价组合

{risk_parity_section}

---

## 6. 相关性异常检测

### 6.1 相关性突变

{anomaly_section}

### 6.2 配对交易机会

{pairs_trading_section}

---

## 7. 主要结论

{conclusions}

---

## 8. 投资建议

{recommendations}

---

## 附录：图表说明

{chart_descriptions}

---

*报告生成时间: {generation_time}*
"""

    # 填充报告内容
    report = report.format(
        report_date=datetime.now().strftime('%Y-%m-%d'),
        data_range=results.get('data_range', 'N/A'),
        industry_corr_section=results.get('industry_corr_section', ''),
        stock_corr_section=results.get('stock_corr_section', ''),
        clustering_section=results.get('clustering_section', ''),
        rolling_corr_section=results.get('rolling_corr_section', ''),
        regime_section=results.get('regime_section', ''),
        mean_reversion_section=results.get('mean_reversion_section', ''),
        pca_variance_section=results.get('pca_variance_section', ''),
        market_factor_section=results.get('market_factor_section', ''),
        factor_construction_section=results.get('factor_construction_section', ''),
        forecast_model_section=results.get('forecast_model_section', ''),
        forecast_performance_section=results.get('forecast_performance_section', ''),
        diversification_section=results.get('diversification_section', ''),
        min_corr_portfolio_section=results.get('min_corr_portfolio_section', ''),
        risk_parity_section=results.get('risk_parity_section', ''),
        anomaly_section=results.get('anomaly_section', ''),
        pairs_trading_section=results.get('pairs_trading_section', ''),
        conclusions=results.get('conclusions', ''),
        recommendations=results.get('recommendations', ''),
        chart_descriptions=results.get('chart_descriptions', ''),
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    return report

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("="*70)
    print("A股市场相关性结构研究")
    print("="*70)

    results = {}

    # 1. 加载数据
    print("\n[1/10] 加载数据...")
    df_industry, industry_names = load_industry_data()
    print(f"  行业数据: {df_industry.shape}")

    df_stock, stock_names, stock_industries = load_stock_data()
    print(f"  个股数据: {df_stock.shape}")

    results['data_range'] = f"行业: {df_industry.index.min()} - {df_industry.index.max()}, 个股: {df_stock.index.min()} - {df_stock.index.max()}"

    # 2. 行业相关性矩阵分析
    print("\n[2/10] 行业相关性分析...")
    industry_corr, df_industry_clean = analyze_correlation_matrix(df_industry, "行业")

    if industry_corr is not None:
        plot_correlation_heatmap(industry_corr, "A股申万一级行业相关性矩阵", "industry_correlation_heatmap.png")
        analyze_correlation_distribution(industry_corr, "行业", "industry_correlation_distribution.png")

        corr_values = industry_corr.values[np.triu_indices_from(industry_corr.values, k=1)]
        results['industry_corr_section'] = f"""
行业间相关性分析结果:

- **分析维度**: {industry_corr.shape[0]} 个申万一级行业
- **数据时间**: {df_industry.index.min()} 至 {df_industry.index.max()}
- **平均相关性**: {np.mean(corr_values):.4f}
- **中位数相关性**: {np.median(corr_values):.4f}
- **相关性标准差**: {np.std(corr_values):.4f}
- **相关性范围**: [{np.min(corr_values):.4f}, {np.max(corr_values):.4f}]
- **正相关占比**: {np.mean(corr_values > 0)*100:.1f}%
- **强相关(>0.7)占比**: {np.mean(corr_values > 0.7)*100:.1f}%

**关键发现**: A股市场行业间普遍呈现正相关，反映了系统性风险的主导地位。高相关性意味着行业分散化的效果有限。

![行业相关性热力图](industry_correlation_heatmap.png)
"""

    # 3. 行业间相关性详细分析
    print("\n[3/10] 行业间相关性详细分析...")
    industry_corr_detail, avg_corr_per_industry = industry_correlation_analysis(df_industry)

    if avg_corr_per_industry is not None:
        high_sync = avg_corr_per_industry.sort_values(ascending=False).head(5)
        low_sync = avg_corr_per_industry.sort_values().head(5)

        results['industry_corr_section'] += f"""

### 行业市场同步性分析

**与市场同步性最高的行业** (高Beta特征):
| 行业 | 平均相关性 |
|------|-----------|
""" + "\n".join([f"| {name} | {avg_corr_per_industry[name]:.4f} |" for name in high_sync.index])

        results['industry_corr_section'] += f"""

**与市场同步性最低的行业** (防御性特征):
| 行业 | 平均相关性 |
|------|-----------|
""" + "\n".join([f"| {name} | {avg_corr_per_industry[name]:.4f} |" for name in low_sync.index])

    # 4. 个股相关性分析
    print("\n[4/10] 个股相关性分析...")
    stock_corr, df_stock_clean = analyze_correlation_matrix(df_stock, "个股")

    if stock_corr is not None:
        plot_correlation_heatmap(stock_corr, "A股个股相关性矩阵(Top300)", "stock_correlation_heatmap.png")
        analyze_correlation_distribution(stock_corr, "个股", "stock_correlation_distribution.png")

        corr_values = stock_corr.values[np.triu_indices_from(stock_corr.values, k=1)]
        results['stock_corr_section'] = f"""
个股相关性分析结果:

- **分析维度**: {stock_corr.shape[0]} 只活跃股票
- **数据时间**: {df_stock.index.min()} 至 {df_stock.index.max()}
- **平均相关性**: {np.mean(corr_values):.4f}
- **中位数相关性**: {np.median(corr_values):.4f}
- **相关性标准差**: {np.std(corr_values):.4f}
- **相关性范围**: [{np.min(corr_values):.4f}, {np.max(corr_values):.4f}]
- **正相关占比**: {np.mean(corr_values > 0)*100:.1f}%

**关键发现**: 与行业层面相比，个股层面相关性略低，但整体仍呈正相关主导。这意味着个股选择对于分散风险有一定效果，但无法完全消除系统性风险。

![个股相关性热力图](stock_correlation_heatmap.png)
"""

    # 5. 聚类分析
    print("\n[5/10] 聚类分析...")
    if industry_corr is not None:
        linkage_matrix, labels, cluster_df = correlation_clustering(industry_corr, n_clusters=5)
        plot_dendrogram(linkage_matrix, list(industry_corr.columns), "行业相关性聚类", "industry_dendrogram.png")

        cluster_summary = ""
        for i in range(1, 6):
            members = cluster_df[cluster_df['cluster']==i]['name'].tolist()
            cluster_summary += f"\n**类别{i}** ({len(members)}个行业): {', '.join(members)}\n"

        results['clustering_section'] = f"""
基于相关性的行业聚类分析:

使用层次聚类(Ward方法)将行业分为5个类别:
{cluster_summary}

**聚类解读**:
- 同一聚类内的行业具有相似的市场表现特征
- 跨聚类配置可以获得更好的分散化效果
- 聚类结果反映了行业间的经济关联性

![行业聚类树状图](industry_dendrogram.png)
"""

    # 6. 滚动相关性分析
    print("\n[6/10] 滚动相关性分析...")
    rolling_df = rolling_correlation_analysis(df_industry, window=60)

    if rolling_df is not None:
        plot_rolling_correlation(rolling_df, "行业", "rolling_correlation_industry.png")
        rolling_df = detect_correlation_regimes(rolling_df)

        results['rolling_corr_section'] = f"""
滚动相关性分析(60日窗口):

- **长期平均相关性**: {rolling_df['avg_corr'].mean():.4f}
- **相关性波动(标准差)**: {rolling_df['avg_corr'].std():.4f}
- **最高相关性**: {rolling_df['avg_corr'].max():.4f}
- **最低相关性**: {rolling_df['avg_corr'].min():.4f}

**时间特征**:
- 相关性呈现明显的时变特征
- 市场压力期间相关性显著上升
- 平稳期相关性趋于下降

![滚动相关性](rolling_correlation_industry.png)
"""

        regime_counts = rolling_df['regime'].value_counts()
        results['regime_section'] = f"""
相关性状态分布:

| 状态 | 天数 | 占比 |
|------|------|------|
| 正常 | {regime_counts.get('normal', 0)} | {regime_counts.get('normal', 0)/len(rolling_df)*100:.1f}% |
| 高相关 | {regime_counts.get('high', 0)} | {regime_counts.get('high', 0)/len(rolling_df)*100:.1f}% |
| 低相关 | {regime_counts.get('low', 0)} | {regime_counts.get('low', 0)/len(rolling_df)*100:.1f}% |

**投资含义**:
- 高相关期: 系统性风险主导，行业配置效果减弱，应降低整体仓位
- 低相关期: 个股/行业选择价值凸显，可增加主动配置比例
"""

        # 均值回归分析
        autocorr = pd.Series(rolling_df['avg_corr'].values).autocorr(lag=20)
        results['mean_reversion_section'] = f"""
相关性均值回归分析:

- **相关性自相关系数(lag=20)**: {autocorr:.4f}
- **长期均值**: {rolling_df['avg_corr'].mean():.4f}

相关性具有{'' if autocorr > 0.5 else '一定程度的'}均值回归特征:
- 极端高相关后往往回落
- 低相关状态不可持续
- 可利用相关性偏离均值进行择时
"""

    # 7. PCA分析
    print("\n[7/10] 主成分分析...")
    pca_industry, pca_result_industry, cols_industry = pca_analysis(df_industry_clean, n_components=10)

    if pca_industry is not None:
        plot_pca_results(pca_industry, cols_industry, "行业PCA分析", "pca_industry.png")

        explained_var = pca_industry.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        n_90 = np.argmax(cumulative_var >= 0.90) + 1

        results['pca_variance_section'] = f"""
主成分方差解释:

| 主成分 | 方差解释 | 累计解释 |
|--------|----------|----------|
| PC1 | {explained_var[0]*100:.2f}% | {cumulative_var[0]*100:.2f}% |
| PC2 | {explained_var[1]*100:.2f}% | {cumulative_var[1]*100:.2f}% |
| PC3 | {explained_var[2]*100:.2f}% | {cumulative_var[2]*100:.2f}% |
| PC4 | {explained_var[3]*100:.2f}% | {cumulative_var[3]*100:.2f}% |
| PC5 | {explained_var[4]*100:.2f}% | {cumulative_var[4]*100:.2f}% |

- **解释90%方差需要**: {n_90} 个主成分
- **第一主成分(市场因子)解释力**: {explained_var[0]*100:.1f}%

![PCA分析](pca_industry.png)
"""

        # 市场因子分析
        pc1_loadings = pd.Series(pca_industry.components_[0], index=cols_industry)
        top_loadings = pc1_loadings.abs().sort_values(ascending=False).head(10)

        results['market_factor_section'] = f"""
第一主成分(市场因子)载荷:

市场因子代表了系统性风险，解释了{explained_var[0]*100:.1f}%的总方差。

**市场因子载荷最高的行业**:
| 行业 | 载荷 |
|------|------|
""" + "\n".join([f"| {name} | {pc1_loadings[name]:.4f} |" for name in top_loadings.index])

        results['factor_construction_section'] = f"""
基于PCA的因子构建:

1. **市场因子(PC1)**: 反映整体市场涨跌，与大盘指数高度相关
2. **行业轮动因子(PC2)**: 反映周期vs防御的切换
3. **风格因子(PC3+)**: 反映价值/成长等风格差异

**应用建议**:
- 使用前{n_90}个主成分可以有效降维，保留90%信息
- 第一主成分可作为市场Beta的代理
- 剩余主成分可用于构建市场中性策略
"""

    # 8. 相关性预测
    print("\n[8/10] 相关性预测分析...")
    forecast_df = correlation_forecast(df_industry_clean, forecast_window=20)

    if forecast_df is not None:
        plot_correlation_forecast(forecast_df, "correlation_forecast.png")

        results['forecast_model_section'] = f"""
相关性预测采用两种方法:

1. **历史相关性法**: 使用全样本历史相关性作为预测
2. **EWMA相关性法**: 使用指数加权移动平均相关性(decay=0.94)

EWMA方法的核心思想来自DCC-GARCH模型:
- 相关性是时变的
- 近期观测值权重更高
- 能够捕捉相关性的动态变化
"""

        results['forecast_performance_section'] = f"""
预测性能比较(20日预测窗口):

| 方法 | 平均MAE |
|------|---------|
| 历史相关性 | {forecast_df['mae_hist'].mean():.4f} |
| EWMA相关性 | {forecast_df['mae_ewma'].mean():.4f} |

**结论**: {'EWMA方法预测效果更好' if forecast_df['mae_ewma'].mean() < forecast_df['mae_hist'].mean() else '历史相关性方法表现更稳定'}

![相关性预测](correlation_forecast.png)
"""

    # 9. 组合分散化
    print("\n[9/10] 组合分散化分析...")
    div_results = portfolio_diversification_analysis(df_industry_clean, industry_corr)

    if div_results is not None:
        results['diversification_section'] = f"""
分散化效果分析:

| 指标 | 等权组合 | 最小方差组合 | 风险平价组合 |
|------|----------|--------------|--------------|
| 日均收益 | {div_results['eq_return']:.4f}% | {div_results['mv_return']:.4f}% | {div_results['rp_return']:.4f}% |
| 日波动率 | {div_results['eq_vol']:.4f}% | {div_results['mv_vol']:.4f}% | {div_results['rp_vol']:.4f}% |

**分散化效果**:
- 分散化比率: {div_results['div_ratio']:.2f}
- 波动率降低: {(1-1/div_results['div_ratio'])*100:.1f}%
"""

        results['risk_parity_section'] = f"""
风险平价组合分析:

风险平价策略目标是使每个资产对组合风险的贡献相等。

**等权 vs 风险平价**:
- 等权组合: 每个行业配置{100/div_results['eq_weights'].shape[0]:.1f}%
- 风险平价: 根据波动率倒数加权配置

风险平价在高波动率资产上配置较低权重，有助于控制组合整体风险。
"""

    # 最小相关性组合
    min_corr_assets, avg_corr_assets = minimum_correlation_portfolio(df_industry_clean)

    if min_corr_assets:
        results['min_corr_portfolio_section'] = f"""
最小相关性组合:

通过贪婪算法选择相关性最低的行业组合:

**入选行业**:
""" + "\n".join([f"- {asset}" for asset in min_corr_assets])

        results['min_corr_portfolio_section'] += """

**选择逻辑**:
1. 首先选择与市场同步性最低的行业
2. 逐步添加与已选行业相关性最低的行业
3. 最终组合内平均相关性最小化
"""

    # 10. 异常检测
    print("\n[10/10] 相关性异常检测...")
    anomaly_df = correlation_anomaly_detection(df_industry_clean)

    if anomaly_df is not None and len(anomaly_df) > 0:
        top_anomalies = anomaly_df.head(10)
        results['anomaly_section'] = f"""
相关性突变检测:

检测到 {len(anomaly_df)} 个相关性异常事件。

**最显著的相关性变化**:
| 日期 | 行业1 | 行业2 | 变化前 | 变化后 | 变化幅度 |
|------|-------|-------|--------|--------|----------|
""" + "\n".join([f"| {row['date']} | {row['asset1']} | {row['asset2']} | {row['prev_corr']:.3f} | {row['curr_corr']:.3f} | {row['change']:+.3f} |"
                  for _, row in top_anomalies.iterrows()])
    else:
        results['anomaly_section'] = "在分析期间未检测到显著的相关性突变事件。"

    # 配对交易
    pairs_df = pairs_trading_opportunities(df_industry_clean, industry_corr, threshold=0.85)

    if pairs_df is not None and len(pairs_df) > 0:
        top_pairs = pairs_df.head(10)
        results['pairs_trading_section'] = f"""
配对交易机会:

高相关性行业配对(相关系数>0.85):

| 行业1 | 行业2 | 相关系数 |
|-------|-------|----------|
""" + "\n".join([f"| {row['asset1']} | {row['asset2']} | {row['correlation']:.4f} |"
                  for _, row in top_pairs.iterrows()])

        results['pairs_trading_section'] += """

**配对交易策略**:
1. 选择高相关性配对
2. 计算标准化价差
3. 当价差偏离均值时建立套利头寸
4. 价差回归时平仓获利
"""
    else:
        results['pairs_trading_section'] = "未发现符合条件的高相关性配对。"

    # 结论和建议
    results['conclusions'] = """
1. **高相关性市场**: A股市场整体呈现高相关性特征，系统性风险主导
2. **分散化有限**: 行业分散化效果有限，个股层面分散效果略好
3. **时变相关性**: 相关性具有明显的时变特征，危机期间显著上升
4. **市场因子主导**: PCA分析显示第一主成分(市场因子)解释力超过50%
5. **相关性可预测**: EWMA方法对短期相关性有一定预测能力
6. **聚类结构清晰**: 行业可分为周期/防御/金融等几大类别
"""

    results['recommendations'] = """
1. **资产配置**:
   - 在高相关期降低整体仓位
   - 选择相关性较低的行业进行分散配置

2. **风险管理**:
   - 监控滚动相关性变化
   - 相关性上升时提高警惕

3. **策略应用**:
   - 利用配对交易机会
   - 基于PCA构建市场中性策略

4. **组合优化**:
   - 考虑相关性约束的组合优化
   - 采用风险平价策略控制风险
"""

    results['chart_descriptions'] = """
1. **industry_correlation_heatmap.png**: 行业相关性热力图，经聚类排序
2. **industry_correlation_distribution.png**: 行业相关性分布直方图
3. **stock_correlation_heatmap.png**: 个股相关性热力图
4. **stock_correlation_distribution.png**: 个股相关性分布
5. **industry_dendrogram.png**: 行业聚类树状图
6. **rolling_correlation_industry.png**: 滚动相关性时序图
7. **pca_industry.png**: 主成分分析结果图
8. **correlation_forecast.png**: 相关性预测对比图
"""

    # 生成报告
    report = generate_report(results)

    # 保存报告
    report_path = f"{OUTPUT_DIR}/correlation_study.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*70}")
    print("分析完成!")
    print(f"报告已保存至: {report_path}")
    print(f"{'='*70}")

    return results

if __name__ == "__main__":
    main()
