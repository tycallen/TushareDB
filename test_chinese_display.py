#!/usr/bin/env python
"""
快速测试中文显示
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from market_width import setup_chinese_font

# 配置中文字体
setup_chinese_font(verbose=True)

# 创建测试数据
industries = ['软件服务', '半导体', '电气设备', '化工原料', '医疗保健', '银行', '航空', '食品']
dates = ['12-10', '12-11', '12-12', '12-13', '12-16']

np.random.seed(42)
data = np.random.randint(30, 70, size=(len(dates), len(industries)))
df = pd.DataFrame(data, index=dates, columns=industries)
df['总分'] = df.sum(axis=1)

print("\n测试数据:")
print(df)

# 创建热力图
fig = plt.figure(figsize=(14, 6))
grid = plt.GridSpec(1, 10)

# 左侧：行业宽度
cmap = sns.diverging_palette(200, 10, as_cmap=True)
ax1 = fig.add_subplot(grid[:, :-1])
ax1.xaxis.set_ticks_position('top')
sns.heatmap(
    df[df.columns[:-1]],
    vmin=0, vmax=100,
    annot=True, fmt="d",
    cmap=cmap,
    annot_kws={'size': 9},
    cbar=False,
    linewidths=0.5,
    linecolor='white'
)
ax1.set_ylabel('交易日', fontsize=12)

# 右侧：总分
ax2 = fig.add_subplot(grid[:, -1])
ax2.xaxis.set_ticks_position('top')
sns.heatmap(
    df[['总分']],
    vmin=0, vmax=len(industries) * 100,
    annot=True, fmt="d",
    cmap=cmap,
    annot_kws={'size': 10, 'weight': 'bold'},
    linewidths=0.5,
    linecolor='white'
)

plt.suptitle('市场宽度图 - 中文显示测试', fontsize=16, y=1.02, fontweight='bold')
plt.tight_layout()

# 保存图片
output_file = 'market_width_test.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ 测试图片已保存: {output_file}")
print("\n请检查图片中的中文是否显示正常：")
print("  1. 标题: '市场宽度图 - 中文显示测试'")
print("  2. 行业名称: 软件服务、半导体、电气设备等")
print("  3. 数字和符号应该正常显示")
print("\n如果看到方框□或乱码，说明字体配置有问题")
print("如果中文全部正常显示，说明配置成功！✓")
