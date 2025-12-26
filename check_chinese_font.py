#!/usr/bin/env python
"""
中文字体检查和测试工具
用于诊断和修复matplotlib中文显示问题
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def list_chinese_fonts():
    """列出系统中所有可用的中文字体"""
    print("=" * 70)
    print("系统中文字体检查")
    print("=" * 70)

    # 获取所有字体
    all_fonts = sorted([f.name for f in fm.fontManager.ttflist])

    # 中文字体关键词
    chinese_keywords = [
        'Hei', 'Song', 'Kai', 'Fang', 'Ming', 'Yuan',  # 中文字体名
        'PingFang', 'ST', 'Microsoft', 'SimHei', 'SimSun',  # 常见中文字体
        'WenQuanYi', 'Noto Sans CJK', 'Source Han',  # Linux/开源字体
        'KaiTi', 'FangSong', 'YouYuan', 'LiSu'  # 其他中文字体
    ]

    # 筛选可能的中文字体
    chinese_fonts = []
    for font in set(all_fonts):
        if any(keyword in font for keyword in chinese_keywords):
            chinese_fonts.append(font)

    chinese_fonts = sorted(set(chinese_fonts))

    if chinese_fonts:
        print(f"\n✓ 找到 {len(chinese_fonts)} 个可能的中文字体:")
        for i, font in enumerate(chinese_fonts, 1):
            print(f"  {i:2d}. {font}")
    else:
        print("\n✗ 未找到中文字体!")
        print("\n建议:")
        print("  - macOS: 系统应该自带 PingFang SC 或 Heiti TC")
        print("  - Windows: 安装微软雅黑或黑体")
        print("  - Linux: 安装 sudo apt-get install fonts-wqy-microhei")

    return chinese_fonts


def test_font(font_name):
    """测试指定字体是否能正确显示中文"""
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 6))

        # 测试文本
        test_text = f"""
字体测试: {font_name}

市场宽度图
中文显示测试

常用字: 股票 行业 涨跌 幅度
数字: 0123456789
符号: +-×÷%

如果上述文字全部显示正常，说明字体配置成功！
        """.strip()

        ax.text(0.5, 0.5, test_text, fontsize=14,
                ha='center', va='center',
                family=font_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        output_file = f'/tmp/font_test_{font_name.replace(" ", "_")}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()

        return True, output_file
    except Exception as e:
        return False, str(e)


def recommend_font():
    """推荐最佳中文字体"""
    print("\n" + "=" * 70)
    print("字体推荐")
    print("=" * 70)

    # 按优先级推荐
    recommended_fonts = [
        ('PingFang SC', 'macOS 苹方 - 简体中文（推荐）'),
        ('Heiti TC', 'macOS 黑体 - 繁体中文'),
        ('STHeiti', 'macOS 华文黑体'),
        ('Microsoft YaHei', 'Windows 微软雅黑（推荐）'),
        ('SimHei', 'Windows 黑体'),
        ('WenQuanYi Micro Hei', 'Linux 文泉驿微米黑（推荐）'),
        ('Noto Sans CJK SC', 'Linux Noto字体'),
    ]

    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    print("\n可用的推荐字体:")
    found_any = False
    for font, desc in recommended_fonts:
        if font in available_fonts:
            print(f"  ✓ {font:20s} - {desc}")
            found_any = True

    if not found_any:
        print("  ✗ 未找到推荐的中文字体")

    print("\n不可用的字体:")
    for font, desc in recommended_fonts:
        if font not in available_fonts:
            print(f"  ✗ {font:20s} - {desc}")


def generate_test_chart():
    """生成一个完整的市场宽度测试图"""
    from market_width import setup_chinese_font

    print("\n" + "=" * 70)
    print("生成测试图表")
    print("=" * 70)

    # 使用自动配置的字体
    setup_chinese_font(verbose=True)

    # 创建测试数据
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # 模拟市场宽度数据
    dates = ['11-15', '11-16', '11-17', '11-18', '11-19']
    industries = ['软件服务', '半导体', '电气设备', '化工原料', '医疗保健']

    data = np.random.randint(30, 70, size=(len(dates), len(industries)))
    df = pd.DataFrame(data, index=dates, columns=industries)
    df['总分'] = df.sum(axis=1)

    # 创建热力图
    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(1, 10)

    # 左侧热力图
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    heatmap1 = fig.add_subplot(grid[:, :-1])
    heatmap1.xaxis.set_ticks_position('top')
    sns.heatmap(
        df[df.columns[:-1]],
        vmin=0, vmax=100,
        annot=True, fmt="d",
        cmap=cmap,
        annot_kws={'size': 10},
        cbar=False,
        linewidths=0.5,
        linecolor='white'
    )

    # 右侧总分
    heatmap2 = fig.add_subplot(grid[:, -1])
    heatmap2.xaxis.set_ticks_position('top')
    sns.heatmap(
        df[['总分']],
        vmin=0, vmax=500,
        annot=True, fmt="d",
        cmap=cmap,
        annot_kws={'size': 10},
        linewidths=0.5,
        linecolor='white'
    )

    plt.suptitle('市场宽度图 - 中文测试', fontsize=16, y=1.02)
    plt.tight_layout()

    output_file = '/tmp/market_width_chinese_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 测试图表已生成: {output_file}")
    print("\n请打开图片检查:")
    print("  1. 标题和轴标签的中文是否正常显示")
    print("  2. 行业名称是否正确显示")
    print("  3. 是否有方框或乱码")

    return output_file


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("matplotlib 中文字体检查工具")
    print("=" * 70)

    # 1. 列出可用字体
    chinese_fonts = list_chinese_fonts()

    # 2. 推荐字体
    recommend_font()

    # 3. 生成测试图表
    try:
        output_file = generate_test_chart()
        print("\n" + "=" * 70)
        print("检查完成!")
        print("=" * 70)
        print(f"\n请查看测试图片: {output_file}")
        print("\n如果中文显示正常，配置成功！")
        print("如果仍有问题，请尝试:")
        print("  1. 清除matplotlib缓存: rm -rf ~/.matplotlib")
        print("  2. 重新安装字体")
        print("  3. 重启Python内核")

    except Exception as e:
        print(f"\n✗ 生成测试图表失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
