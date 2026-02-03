#!/usr/bin/env python3
"""
A股市场股票聚类分析
====================
1. 基于收益率的聚类 - 发现隐含的"风格"分组
2. 基于基本面的聚类 - 识别价值股/成长股/周期股
3. 基于量价特征的聚类 - 识别活跃股/冷门股
4. 多维度综合聚类 - PCA/t-SNE降维可视化
5. 聚类应用研究 - 组合分散化、配对交易、风格轮动
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/stock_clustering_analysis.md'

# 分析时间范围：最近一年
START_DATE = '20240101'
END_DATE = '20260130'

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

# ===== 1. 数据准备 =====
def prepare_return_data():
    """准备收益率数据"""
    print("准备收益率数据...")
    con = get_connection()

    # 获取日收益率数据
    df = con.execute(f"""
        SELECT ts_code, trade_date, pct_chg
        FROM daily
        WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
        ORDER BY ts_code, trade_date
    """).fetchdf()

    # 转换为收益率矩阵
    returns = df.pivot(index='trade_date', columns='ts_code', values='pct_chg')

    # 筛选有足够数据的股票（至少80%数据完整）
    valid_stocks = returns.columns[returns.notna().sum() >= len(returns) * 0.8]
    returns = returns[valid_stocks].fillna(0)

    print(f"  有效股票数: {len(valid_stocks)}, 交易日数: {len(returns)}")
    con.close()
    return returns

def prepare_fundamental_data():
    """准备基本面数据"""
    print("准备基本面数据...")
    con = get_connection()

    # 获取最新的日线基础数据
    latest_date = con.execute(f"""
        SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= '{END_DATE}'
    """).fetchone()[0]

    basic_df = con.execute(f"""
        SELECT ts_code, pe_ttm, pb, turnover_rate_f, total_mv, circ_mv
        FROM daily_basic
        WHERE trade_date = '{latest_date}'
    """).fetchdf()

    # 获取最新财务指标（2024年报或最新季报）
    fina_df = con.execute("""
        SELECT DISTINCT ON (ts_code)
            ts_code, roe, roa, gross_margin, netprofit_margin,
            debt_to_assets, current_ratio, assets_turn,
            netprofit_yoy, or_yoy, eps
        FROM fina_indicator_vip
        WHERE end_date >= '20240101'
        ORDER BY ts_code, end_date DESC
    """).fetchdf()

    # 合并数据
    fund_df = basic_df.merge(fina_df, on='ts_code', how='inner')

    print(f"  基本面数据股票数: {len(fund_df)}")
    con.close()
    return fund_df

def prepare_technical_data():
    """准备量价特征数据"""
    print("准备量价特征数据...")
    con = get_connection()

    # 计算近一年的技术指标
    tech_df = con.execute(f"""
        WITH daily_stats AS (
            SELECT
                ts_code,
                AVG(pct_chg) as avg_return,
                STDDEV(pct_chg) as volatility,
                AVG(vol) as avg_volume,
                AVG(amount) as avg_amount,
                SUM(pct_chg) as total_return,
                COUNT(*) as trading_days
            FROM daily
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
            GROUP BY ts_code
            HAVING COUNT(*) >= 200
        ),
        turnover_stats AS (
            SELECT
                ts_code,
                AVG(turnover_rate_f) as avg_turnover,
                STDDEV(turnover_rate_f) as turnover_std,
                AVG(volume_ratio) as avg_volume_ratio
            FROM daily_basic
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
            GROUP BY ts_code
        )
        SELECT
            d.ts_code,
            d.avg_return,
            d.volatility,
            d.avg_volume,
            d.avg_amount,
            d.total_return,
            t.avg_turnover,
            t.turnover_std,
            t.avg_volume_ratio
        FROM daily_stats d
        JOIN turnover_stats t ON d.ts_code = t.ts_code
    """).fetchdf()

    print(f"  量价特征数据股票数: {len(tech_df)}")
    con.close()
    return tech_df

def get_industry_data():
    """获取申万行业分类"""
    print("获取行业分类...")
    con = get_connection()

    industry_df = con.execute("""
        SELECT ts_code, name, l1_name, l2_name, l3_name
        FROM index_member_all
        WHERE is_new = 'Y'
    """).fetchdf()

    print(f"  行业分类股票数: {len(industry_df)}")
    con.close()
    return industry_df

# ===== 2. 基于收益率的聚类 =====
def return_based_clustering(returns_df, n_clusters=10):
    """基于收益率相关性的聚类"""
    print("\n=== 基于收益率的聚类分析 ===")

    # 计算相关性矩阵
    print("计算收益率相关性矩阵...")
    corr_matrix = returns_df.corr()

    # 转换为距离矩阵 (1 - correlation)
    dist_matrix = 1 - corr_matrix
    dist_matrix = dist_matrix.clip(lower=0)  # 确保非负
    np.fill_diagonal(dist_matrix.values, 0)  # 对角线为0

    # 层次聚类
    print("执行层次聚类...")
    try:
        # 将距离矩阵转换为压缩形式
        dist_condensed = squareform(dist_matrix.values, checks=False)
        linkage_matrix = linkage(dist_condensed, method='ward')

        # 切割树得到聚类
        clusters_hier = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        hier_result = pd.DataFrame({
            'ts_code': returns_df.columns,
            'cluster_hier': clusters_hier
        })
    except Exception as e:
        print(f"  层次聚类出错: {e}")
        hier_result = pd.DataFrame({
            'ts_code': returns_df.columns,
            'cluster_hier': [1] * len(returns_df.columns)
        })

    # K-Means聚类（基于收益率时间序列）
    print("执行K-Means聚类...")
    returns_T = returns_df.T.values  # 股票 x 时间
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_T)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(returns_scaled)

    # 合并结果
    hier_result['cluster_kmeans'] = clusters_kmeans

    # 计算轮廓系数
    try:
        silhouette_hier = silhouette_score(dist_matrix.values, clusters_hier, metric='precomputed')
        silhouette_kmeans = silhouette_score(returns_scaled, clusters_kmeans)
        print(f"  层次聚类轮廓系数: {silhouette_hier:.4f}")
        print(f"  K-Means轮廓系数: {silhouette_kmeans:.4f}")
    except:
        silhouette_hier = silhouette_kmeans = 0

    return hier_result, corr_matrix, {
        'silhouette_hier': silhouette_hier,
        'silhouette_kmeans': silhouette_kmeans,
        'n_clusters': n_clusters
    }

# ===== 3. 基于基本面的聚类 =====
def fundamental_clustering(fund_df, n_clusters=6):
    """基于基本面特征的聚类"""
    print("\n=== 基于基本面的聚类分析 ===")

    # 选择特征
    features = ['pe_ttm', 'pb', 'roe', 'roa', 'gross_margin',
                'netprofit_margin', 'debt_to_assets', 'total_mv',
                'netprofit_yoy', 'or_yoy']

    # 准备数据
    df = fund_df[['ts_code'] + features].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 对PE/PB做截断处理（去除极端值）
    for col in ['pe_ttm', 'pb']:
        q01, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=q01, upper=q99)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # K-Means聚类
    print("执行K-Means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # 计算轮廓系数
    silhouette = silhouette_score(X, df['cluster'])
    print(f"  轮廓系数: {silhouette:.4f}")

    # 分析各聚类的特征
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'median', 'std'])

    # 聚类解释
    cluster_labels = interpret_fundamental_clusters(df, features)
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    return df, cluster_stats, {
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels
    }

def interpret_fundamental_clusters(df, features):
    """解释基本面聚类的含义"""
    labels = {}
    cluster_means = df.groupby('cluster')[features].mean()

    for cluster in df['cluster'].unique():
        means = cluster_means.loc[cluster]

        # 判断风格
        pe_rank = (cluster_means['pe_ttm'] <= means['pe_ttm']).sum() / len(cluster_means)
        pb_rank = (cluster_means['pb'] <= means['pb']).sum() / len(cluster_means)
        roe_rank = (cluster_means['roe'] <= means['roe']).sum() / len(cluster_means)
        growth_rank = (cluster_means['netprofit_yoy'] <= means['netprofit_yoy']).sum() / len(cluster_means)
        mv_rank = (cluster_means['total_mv'] <= means['total_mv']).sum() / len(cluster_means)

        # 分类逻辑
        if pe_rank < 0.3 and pb_rank < 0.3:
            style = "深度价值"
        elif pe_rank < 0.5 and roe_rank > 0.7:
            style = "优质价值"
        elif growth_rank > 0.7 and pe_rank > 0.5:
            style = "高成长"
        elif mv_rank > 0.8:
            style = "大盘蓝筹"
        elif mv_rank < 0.2:
            style = "小盘股"
        elif roe_rank > 0.8:
            style = "高盈利"
        else:
            style = "均衡型"

        labels[cluster] = style

    return labels

# ===== 4. 基于量价特征的聚类 =====
def technical_clustering(tech_df, n_clusters=5):
    """基于量价特征的聚类"""
    print("\n=== 基于量价特征的聚类分析 ===")

    # 选择特征
    features = ['avg_return', 'volatility', 'avg_amount', 'total_return',
                'avg_turnover', 'turnover_std', 'avg_volume_ratio']

    # 准备数据
    df = tech_df[['ts_code'] + features].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 对数转换金额（消除量级差异）
    df['log_amount'] = np.log1p(df['avg_amount'])
    features_use = ['avg_return', 'volatility', 'log_amount', 'total_return',
                    'avg_turnover', 'turnover_std']

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features_use])

    # K-Means聚类
    print("执行K-Means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # 计算轮廓系数
    silhouette = silhouette_score(X, df['cluster'])
    print(f"  轮廓系数: {silhouette:.4f}")

    # 分析各聚类的特征
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'median'])

    # 聚类解释
    cluster_labels = interpret_technical_clusters(df, features)
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    return df, cluster_stats, {
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels
    }

def interpret_technical_clusters(df, features):
    """解释量价聚类的含义"""
    labels = {}
    cluster_means = df.groupby('cluster')[features].mean()

    for cluster in df['cluster'].unique():
        means = cluster_means.loc[cluster]

        # 判断特征
        vol_rank = (cluster_means['volatility'] <= means['volatility']).sum() / len(cluster_means)
        turnover_rank = (cluster_means['avg_turnover'] <= means['avg_turnover']).sum() / len(cluster_means)
        amount_rank = (cluster_means['avg_amount'] <= means['avg_amount']).sum() / len(cluster_means)
        return_rank = (cluster_means['total_return'] <= means['total_return']).sum() / len(cluster_means)

        # 分类逻辑
        if turnover_rank > 0.8 and vol_rank > 0.8:
            style = "高度活跃"
        elif turnover_rank < 0.2 and vol_rank < 0.3:
            style = "冷门低波"
        elif amount_rank > 0.8:
            style = "大资金关注"
        elif vol_rank > 0.7 and return_rank > 0.7:
            style = "强势高波"
        elif vol_rank > 0.7 and return_rank < 0.3:
            style = "弱势高波"
        elif turnover_rank > 0.6:
            style = "活跃交易"
        else:
            style = "常规流动"

        labels[cluster] = style

    return labels

# ===== 5. 多维度综合聚类 =====
def comprehensive_clustering(returns_df, fund_df, tech_df, n_clusters=8):
    """多维度综合聚类"""
    print("\n=== 多维度综合聚类分析 ===")

    # 找到共同股票
    common_stocks = set(returns_df.columns) & set(fund_df['ts_code']) & set(tech_df['ts_code'])
    print(f"  共同股票数: {len(common_stocks)}")

    # 准备收益率特征（动量、波动率等）
    returns_features = pd.DataFrame(index=list(common_stocks))
    for stock in common_stocks:
        if stock in returns_df.columns:
            r = returns_df[stock].values
            returns_features.loc[stock, 'momentum_1m'] = r[-20:].sum() if len(r) >= 20 else 0
            returns_features.loc[stock, 'momentum_3m'] = r[-60:].sum() if len(r) >= 60 else 0
            returns_features.loc[stock, 'volatility'] = r.std() if len(r) > 0 else 0
            returns_features.loc[stock, 'skewness'] = pd.Series(r).skew() if len(r) > 0 else 0

    returns_features = returns_features.reset_index().rename(columns={'index': 'ts_code'})

    # 合并所有特征
    fund_cols = ['ts_code', 'pe_ttm', 'pb', 'roe', 'total_mv', 'netprofit_yoy']
    tech_cols = ['ts_code', 'avg_turnover', 'avg_amount']

    combined = returns_features.merge(fund_df[fund_cols], on='ts_code', how='inner')
    combined = combined.merge(tech_df[tech_cols], on='ts_code', how='inner')

    # 清理数据
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()

    print(f"  综合数据股票数: {len(combined)}")

    # 特征列
    feature_cols = ['momentum_1m', 'momentum_3m', 'volatility', 'skewness',
                    'pe_ttm', 'pb', 'roe', 'total_mv', 'netprofit_yoy',
                    'avg_turnover', 'avg_amount']

    # 截断PE/PB极端值
    for col in ['pe_ttm', 'pb']:
        q01, q99 = combined[col].quantile([0.01, 0.99])
        combined[col] = combined[col].clip(lower=q01, upper=q99)

    # 对市值和成交额取对数
    combined['log_mv'] = np.log1p(combined['total_mv'])
    combined['log_amount'] = np.log1p(combined['avg_amount'])

    feature_cols_final = ['momentum_1m', 'momentum_3m', 'volatility', 'skewness',
                          'pe_ttm', 'pb', 'roe', 'netprofit_yoy',
                          'avg_turnover', 'log_mv', 'log_amount']

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(combined[feature_cols_final])

    # PCA降维
    print("执行PCA降维...")
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    print(f"  PCA解释方差比: {pca.explained_variance_ratio_}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

    # t-SNE降维（用于可视化）
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    # K-Means聚类
    print("执行K-Means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    combined['cluster'] = kmeans.fit_predict(X_pca)

    # 添加降维结果
    combined['pca_1'] = X_pca[:, 0]
    combined['pca_2'] = X_pca[:, 1]
    combined['tsne_1'] = X_tsne[:, 0]
    combined['tsne_2'] = X_tsne[:, 1]

    # 计算轮廓系数
    silhouette = silhouette_score(X_pca, combined['cluster'])
    print(f"  轮廓系数: {silhouette:.4f}")

    # PCA载荷矩阵
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols_final,
        columns=[f'PC{i+1}' for i in range(5)]
    )

    return combined, pca_loadings, {
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'pca_variance': pca.explained_variance_ratio_.tolist()
    }

# ===== 6. 与行业分类对比 =====
def compare_with_industry(cluster_result, industry_df, cluster_col='cluster'):
    """将聚类结果与申万行业分类对比"""
    print("\n=== 聚类与行业分类对比 ===")

    # 合并数据
    merged = cluster_result.merge(industry_df[['ts_code', 'l1_name', 'l2_name']],
                                   on='ts_code', how='inner')

    # 计算每个聚类中的行业分布
    cluster_industry = merged.groupby([cluster_col, 'l1_name']).size().unstack(fill_value=0)

    # 计算行业纯度
    cluster_purity = {}
    for cluster in merged[cluster_col].unique():
        cluster_data = merged[merged[cluster_col] == cluster]
        industry_counts = cluster_data['l1_name'].value_counts()
        purity = industry_counts.iloc[0] / len(cluster_data) if len(cluster_data) > 0 else 0
        top_industry = industry_counts.index[0] if len(industry_counts) > 0 else 'Unknown'
        cluster_purity[cluster] = {
            'purity': purity,
            'top_industry': top_industry,
            'count': len(cluster_data)
        }

    return cluster_industry, cluster_purity

# ===== 7. 聚类应用研究 =====
def find_pair_trading_candidates(returns_df, cluster_result, top_n=20):
    """寻找配对交易机会"""
    print("\n=== 配对交易机会分析 ===")

    # 计算相关性
    corr_matrix = returns_df.corr()

    pairs = []
    stocks = cluster_result['ts_code'].tolist()

    for i, stock1 in enumerate(stocks):
        if stock1 not in corr_matrix.columns:
            continue
        cluster1 = cluster_result[cluster_result['ts_code'] == stock1]['cluster'].values[0]

        for stock2 in stocks[i+1:]:
            if stock2 not in corr_matrix.columns:
                continue
            cluster2 = cluster_result[cluster_result['ts_code'] == stock2]['cluster'].values[0]

            # 同一聚类内的高相关股票对
            if cluster1 == cluster2:
                corr = corr_matrix.loc[stock1, stock2]
                if corr > 0.6:  # 高相关阈值
                    pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': corr,
                        'cluster': cluster1
                    })

    # 按相关性排序
    pairs = sorted(pairs, key=lambda x: x['correlation'], reverse=True)[:top_n]
    print(f"  找到 {len(pairs)} 个高相关配对")

    return pairs

def analyze_cluster_returns(returns_df, cluster_result, cluster_col='cluster'):
    """分析各聚类的收益特征"""
    print("\n=== 聚类收益特征分析 ===")

    # 按聚类计算平均收益
    cluster_returns = {}

    for cluster in cluster_result[cluster_col].unique():
        stocks = cluster_result[cluster_result[cluster_col] == cluster]['ts_code'].tolist()
        stocks_in_returns = [s for s in stocks if s in returns_df.columns]

        if len(stocks_in_returns) > 0:
            cluster_ret = returns_df[stocks_in_returns].mean(axis=1)
            cluster_returns[cluster] = {
                'total_return': cluster_ret.sum(),
                'avg_daily_return': cluster_ret.mean(),
                'volatility': cluster_ret.std(),
                'sharpe': cluster_ret.mean() / cluster_ret.std() * np.sqrt(252) if cluster_ret.std() > 0 else 0,
                'max_drawdown': (cluster_ret.cumsum() - cluster_ret.cumsum().cummax()).min(),
                'stock_count': len(stocks_in_returns)
            }

    return pd.DataFrame(cluster_returns).T

def portfolio_diversification_analysis(cluster_result, returns_df):
    """组合分散化建议"""
    print("\n=== 组合分散化建议 ===")

    # 计算聚类间的相关性
    cluster_means = {}
    for cluster in cluster_result['cluster'].unique():
        stocks = cluster_result[cluster_result['cluster'] == cluster]['ts_code'].tolist()
        stocks_in_returns = [s for s in stocks if s in returns_df.columns]
        if len(stocks_in_returns) > 0:
            cluster_means[cluster] = returns_df[stocks_in_returns].mean(axis=1)

    cluster_corr = pd.DataFrame(cluster_means).corr()

    # 找到低相关的聚类组合
    low_corr_pairs = []
    clusters = list(cluster_means.keys())
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            corr = cluster_corr.loc[c1, c2]
            if corr < 0.5:  # 低相关阈值
                low_corr_pairs.append({
                    'cluster1': c1,
                    'cluster2': c2,
                    'correlation': corr
                })

    low_corr_pairs = sorted(low_corr_pairs, key=lambda x: x['correlation'])

    return cluster_corr, low_corr_pairs

# ===== 8. 生成报告 =====
def generate_report(results):
    """生成分析报告"""
    print("\n=== 生成分析报告 ===")

    report = []
    report.append("# A股市场股票聚类分析报告\n")
    report.append(f"**分析时间范围**: {START_DATE} - {END_DATE}\n")
    report.append(f"**生成日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")

    # 数据概览
    report.append("\n## 1. 数据概览\n")
    report.append(f"- 收益率数据股票数: {results['data_stats']['return_stocks']}")
    report.append(f"- 基本面数据股票数: {results['data_stats']['fund_stocks']}")
    report.append(f"- 量价特征数据股票数: {results['data_stats']['tech_stocks']}")
    report.append(f"- 交易日数: {results['data_stats']['trading_days']}")

    # 基于收益率的聚类
    report.append("\n## 2. 基于收益率的聚类分析\n")
    report.append("### 方法说明")
    report.append("- 计算股票间的收益率相关性矩阵")
    report.append("- 使用层次聚类（Ward方法）和K-Means聚类")
    report.append(f"- 聚类数: {results['return_clustering']['metrics']['n_clusters']}")
    report.append(f"- 层次聚类轮廓系数: {results['return_clustering']['metrics']['silhouette_hier']:.4f}")
    report.append(f"- K-Means轮廓系数: {results['return_clustering']['metrics']['silhouette_kmeans']:.4f}")

    report.append("\n### 聚类与行业对比")
    report.append("各聚类的主导行业及纯度:")
    for cluster, info in results['return_clustering']['industry_purity'].items():
        report.append(f"- 聚类{cluster}: 主导行业={info['top_industry']}, 纯度={info['purity']:.2%}, 股票数={info['count']}")

    # 基于基本面的聚类
    report.append("\n## 3. 基于基本面的聚类分析\n")
    report.append("### 方法说明")
    report.append("- 使用特征: PE/PB/ROE/ROA/毛利率/净利率/资产负债率/市值/净利润增速/营收增速")
    report.append(f"- 聚类数: {results['fundamental_clustering']['metrics']['n_clusters']}")
    report.append(f"- 轮廓系数: {results['fundamental_clustering']['metrics']['silhouette']:.4f}")

    report.append("\n### 聚类风格解释")
    for cluster, label in results['fundamental_clustering']['metrics']['cluster_labels'].items():
        cluster_data = results['fundamental_clustering']['data']
        count = len(cluster_data[cluster_data['cluster'] == cluster])
        report.append(f"- 聚类{cluster} ({label}): {count}只股票")

    report.append("\n### 各聚类特征均值")
    fund_stats = results['fundamental_clustering']['stats']
    report.append("| 聚类 | 风格 | PE_TTM | PB | ROE(%) | 市值(亿) | 净利润增速(%) |")
    report.append("|------|------|--------|-----|--------|----------|---------------|")
    cluster_data = results['fundamental_clustering']['data']
    cluster_labels = results['fundamental_clustering']['metrics']['cluster_labels']
    for cluster in sorted(cluster_labels.keys()):
        c_data = cluster_data[cluster_data['cluster'] == cluster]
        pe = c_data['pe_ttm'].median()
        pb = c_data['pb'].median()
        roe = c_data['roe'].median()
        mv = c_data['total_mv'].median() / 10000 if c_data['total_mv'].median() > 0 else 0
        growth = c_data['netprofit_yoy'].median()
        label = cluster_labels[cluster]
        report.append(f"| {cluster} | {label} | {pe:.1f} | {pb:.2f} | {roe:.1f} | {mv:.1f} | {growth:.1f} |")

    # 基于量价特征的聚类
    report.append("\n## 4. 基于量价特征的聚类分析\n")
    report.append("### 方法说明")
    report.append("- 使用特征: 平均收益率/波动率/成交额/累计收益/换手率/换手率波动/量比")
    report.append(f"- 聚类数: {results['technical_clustering']['metrics']['n_clusters']}")
    report.append(f"- 轮廓系数: {results['technical_clustering']['metrics']['silhouette']:.4f}")

    report.append("\n### 聚类风格解释")
    for cluster, label in results['technical_clustering']['metrics']['cluster_labels'].items():
        tech_data = results['technical_clustering']['data']
        count = len(tech_data[tech_data['cluster'] == cluster])
        report.append(f"- 聚类{cluster} ({label}): {count}只股票")

    report.append("\n### 各聚类特征均值")
    report.append("| 聚类 | 风格 | 波动率(%) | 换手率(%) | 日均成交(万元) | 累计收益(%) |")
    report.append("|------|------|-----------|-----------|----------------|-------------|")
    tech_data = results['technical_clustering']['data']
    tech_labels = results['technical_clustering']['metrics']['cluster_labels']
    for cluster in sorted(tech_labels.keys()):
        c_data = tech_data[tech_data['cluster'] == cluster]
        vol = c_data['volatility'].median()
        turnover = c_data['avg_turnover'].median()
        amount = c_data['avg_amount'].median() / 10000 if c_data['avg_amount'].median() > 0 else 0
        total_ret = c_data['total_return'].median()
        label = tech_labels[cluster]
        report.append(f"| {cluster} | {label} | {vol:.2f} | {turnover:.2f} | {amount:.0f} | {total_ret:.1f} |")

    # 多维度综合聚类
    report.append("\n## 5. 多维度综合聚类分析\n")
    report.append("### 方法说明")
    report.append("- 综合收益率特征（动量、波动率、偏度）+ 基本面特征 + 量价特征")
    report.append("- 使用PCA降维后进行K-Means聚类")
    report.append(f"- 聚类数: {results['comprehensive_clustering']['metrics']['n_clusters']}")
    report.append(f"- 轮廓系数: {results['comprehensive_clustering']['metrics']['silhouette']:.4f}")

    report.append("\n### PCA解释方差")
    pca_var = results['comprehensive_clustering']['metrics']['pca_variance']
    cum_var = np.cumsum(pca_var)
    for i, (var, cum) in enumerate(zip(pca_var, cum_var)):
        report.append(f"- PC{i+1}: 解释方差 {var:.2%}, 累计 {cum:.2%}")

    report.append("\n### PCA主成分载荷（前3个主成分）")
    loadings = results['comprehensive_clustering']['pca_loadings']
    report.append("| 特征 | PC1 | PC2 | PC3 |")
    report.append("|------|-----|-----|-----|")
    for feature in loadings.index:
        report.append(f"| {feature} | {loadings.loc[feature, 'PC1']:.3f} | {loadings.loc[feature, 'PC2']:.3f} | {loadings.loc[feature, 'PC3']:.3f} |")

    report.append("\n### t-SNE可视化描述")
    report.append("将高维特征降至2维进行可视化，可观察到:")
    comp_data = results['comprehensive_clustering']['data']
    for cluster in sorted(comp_data['cluster'].unique()):
        c_data = comp_data[comp_data['cluster'] == cluster]
        tsne_center = (c_data['tsne_1'].mean(), c_data['tsne_2'].mean())
        report.append(f"- 聚类{cluster}: {len(c_data)}只股票，中心位置约({tsne_center[0]:.1f}, {tsne_center[1]:.1f})")

    # 聚类收益特征
    report.append("\n## 6. 聚类收益特征分析\n")
    cluster_returns = results['cluster_returns']
    report.append("| 聚类 | 累计收益(%) | 日均收益(%) | 波动率(%) | 夏普比率 | 最大回撤(%) | 股票数 |")
    report.append("|------|------------|------------|-----------|----------|------------|--------|")
    for cluster in cluster_returns.index:
        row = cluster_returns.loc[cluster]
        report.append(f"| {cluster} | {row['total_return']:.1f} | {row['avg_daily_return']:.3f} | {row['volatility']:.2f} | {row['sharpe']:.2f} | {row['max_drawdown']:.1f} | {int(row['stock_count'])} |")

    # 配对交易机会
    report.append("\n## 7. 配对交易机会\n")
    report.append("同一聚类内高相关性股票对（相关系数>0.6）:")
    pairs = results['pair_trading']
    if len(pairs) > 0:
        report.append("| 股票1 | 股票2 | 相关系数 | 聚类 |")
        report.append("|-------|-------|----------|------|")
        for pair in pairs[:15]:  # 显示前15对
            report.append(f"| {pair['stock1']} | {pair['stock2']} | {pair['correlation']:.4f} | {pair['cluster']} |")
    else:
        report.append("未找到符合条件的配对")

    # 组合分散化建议
    report.append("\n## 8. 组合分散化建议\n")
    report.append("### 聚类间相关性矩阵")
    cluster_corr = results['diversification']['cluster_corr']

    # 创建相关性矩阵表格
    headers = "| 聚类 | " + " | ".join([str(c) for c in cluster_corr.columns]) + " |"
    separator = "|------|" + "|".join(["-----" for _ in cluster_corr.columns]) + "|"
    report.append(headers)
    report.append(separator)
    for idx in cluster_corr.index:
        row_vals = " | ".join([f"{cluster_corr.loc[idx, c]:.2f}" for c in cluster_corr.columns])
        report.append(f"| {idx} | {row_vals} |")

    report.append("\n### 低相关聚类组合建议")
    report.append("选择相关性较低的聚类进行组合配置，可有效降低组合风险:")
    low_corr_pairs = results['diversification']['low_corr_pairs']
    if len(low_corr_pairs) > 0:
        for pair in low_corr_pairs[:10]:
            report.append(f"- 聚类{pair['cluster1']} + 聚类{pair['cluster2']}: 相关系数 {pair['correlation']:.3f}")
    else:
        report.append("聚类间相关性普遍较高，建议进一步细分聚类")

    # 投资应用建议
    report.append("\n## 9. 投资应用建议\n")

    report.append("### 9.1 风格配置建议")
    report.append("基于基本面聚类结果:")
    fund_labels = results['fundamental_clustering']['metrics']['cluster_labels']
    for cluster, label in fund_labels.items():
        c_data = results['fundamental_clustering']['data']
        c_subset = c_data[c_data['cluster'] == cluster]
        if label == "深度价值":
            report.append(f"- **{label}股票**（聚类{cluster}）: 适合价值投资者，在市场悲观时布局")
        elif label == "优质价值":
            report.append(f"- **{label}股票**（聚类{cluster}）: 兼具估值和质量，适合长期持有")
        elif label == "高成长":
            report.append(f"- **{label}股票**（聚类{cluster}）: 适合成长型投资者，需关注业绩兑现")
        elif label == "大盘蓝筹":
            report.append(f"- **{label}股票**（聚类{cluster}）: 流动性好，适合大资金配置")
        elif label == "小盘股":
            report.append(f"- **{label}股票**（聚类{cluster}）: 弹性大，适合风险偏好高的投资者")
        else:
            report.append(f"- **{label}股票**（聚类{cluster}）: 特征中等，可作为均衡配置")

    report.append("\n### 9.2 流动性配置建议")
    report.append("基于量价聚类结果:")
    tech_labels = results['technical_clustering']['metrics']['cluster_labels']
    for cluster, label in tech_labels.items():
        if "活跃" in label:
            report.append(f"- **{label}股票**（聚类{cluster}）: 适合短线交易，但需注意波动风险")
        elif "冷门" in label:
            report.append(f"- **{label}股票**（聚类{cluster}）: 流动性风险高，大资金需谨慎")
        elif "大资金" in label:
            report.append(f"- **{label}股票**（聚类{cluster}）: 机构关注度高，可跟踪主力动向")

    report.append("\n### 9.3 风格轮动策略建议")
    report.append("1. **市场上涨期**: 重点配置高成长、高弹性聚类")
    report.append("2. **市场震荡期**: 配置优质价值、低波动聚类")
    report.append("3. **市场下跌期**: 防守为主，配置大盘蓝筹、深度价值聚类")
    report.append("4. **监控指标**: 跟踪各聚类的相对强弱，进行动态调整")

    report.append("\n### 9.4 组合构建建议")
    report.append("1. **核心卫星策略**: 核心配置低相关性的多个聚类，卫星仓位配置高动量聚类")
    report.append("2. **等权配置**: 从每个主要聚类中选取代表性股票等权配置")
    report.append("3. **风险平价**: 根据各聚类波动率进行反向加权配置")

    report.append("\n## 10. 附录: 各聚类代表性股票\n")

    # 从综合聚类中选取每个聚类的代表性股票
    comp_data = results['comprehensive_clustering']['data']
    industry_df = results['industry_data']

    for cluster in sorted(comp_data['cluster'].unique()):
        c_data = comp_data[comp_data['cluster'] == cluster].head(10)
        c_data_with_name = c_data.merge(industry_df[['ts_code', 'name', 'l1_name']], on='ts_code', how='left')

        report.append(f"\n### 聚类{cluster}代表股票")
        report.append("| 股票代码 | 名称 | 行业 |")
        report.append("|----------|------|------|")
        for _, row in c_data_with_name.iterrows():
            name = row.get('name', 'N/A')
            industry = row.get('l1_name', 'N/A')
            report.append(f"| {row['ts_code']} | {name} | {industry} |")

    report.append("\n---")
    report.append("*报告由聚类分析系统自动生成*")

    # 写入文件
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"报告已保存至: {REPORT_PATH}")
    return '\n'.join(report)

# ===== 主函数 =====
def main():
    print("=" * 60)
    print("A股市场股票聚类分析")
    print("=" * 60)

    # 1. 数据准备
    returns_df = prepare_return_data()
    fund_df = prepare_fundamental_data()
    tech_df = prepare_technical_data()
    industry_df = get_industry_data()

    data_stats = {
        'return_stocks': len(returns_df.columns),
        'fund_stocks': len(fund_df),
        'tech_stocks': len(tech_df),
        'trading_days': len(returns_df)
    }

    # 2. 基于收益率的聚类
    return_cluster, corr_matrix, return_metrics = return_based_clustering(returns_df, n_clusters=10)
    return_cluster_industry, return_cluster_purity = compare_with_industry(return_cluster, industry_df, 'cluster_hier')

    # 3. 基于基本面的聚类
    fund_cluster, fund_stats, fund_metrics = fundamental_clustering(fund_df, n_clusters=6)

    # 4. 基于量价特征的聚类
    tech_cluster, tech_stats, tech_metrics = technical_clustering(tech_df, n_clusters=5)

    # 5. 多维度综合聚类
    comp_cluster, pca_loadings, comp_metrics = comprehensive_clustering(returns_df, fund_df, tech_df, n_clusters=8)

    # 6. 聚类收益分析
    cluster_returns = analyze_cluster_returns(returns_df, comp_cluster, 'cluster')

    # 7. 配对交易分析
    pairs = find_pair_trading_candidates(returns_df, comp_cluster, top_n=20)

    # 8. 组合分散化分析
    cluster_corr, low_corr_pairs = portfolio_diversification_analysis(comp_cluster, returns_df)

    # 汇总结果
    results = {
        'data_stats': data_stats,
        'return_clustering': {
            'data': return_cluster,
            'corr_matrix': corr_matrix,
            'metrics': return_metrics,
            'industry_distribution': return_cluster_industry,
            'industry_purity': return_cluster_purity
        },
        'fundamental_clustering': {
            'data': fund_cluster,
            'stats': fund_stats,
            'metrics': fund_metrics
        },
        'technical_clustering': {
            'data': tech_cluster,
            'stats': tech_stats,
            'metrics': tech_metrics
        },
        'comprehensive_clustering': {
            'data': comp_cluster,
            'pca_loadings': pca_loadings,
            'metrics': comp_metrics
        },
        'cluster_returns': cluster_returns,
        'pair_trading': pairs,
        'diversification': {
            'cluster_corr': cluster_corr,
            'low_corr_pairs': low_corr_pairs
        },
        'industry_data': industry_df
    }

    # 9. 生成报告
    report = generate_report(results)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
