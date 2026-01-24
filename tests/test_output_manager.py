"""
测试OutputManager
"""

import pytest
import pandas as pd
import os
import shutil
from tushare_db.sector_analysis import OutputManager


@pytest.fixture
def output_manager():
    """创建测试用的OutputManager"""
    test_dir = 'output/test_output'
    om = OutputManager(output_dir=test_dir)
    yield om
    # 清理测试目录
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_output_manager_init(output_manager):
    """测试OutputManager初始化"""
    assert os.path.exists(output_manager.output_dir)
    assert os.path.exists(output_manager.data_dir)
    assert os.path.exists(output_manager.plots_dir)
    print(f"✓ 输出目录已创建: {output_manager.output_dir}")


def test_save_dataframe_csv(output_manager):
    """测试保存CSV"""
    df = pd.DataFrame({
        'sector_code': ['801010.SI', '801030.SI'],
        'sector_name': ['农林牧渔', '基础化工'],
        'return': [1.5, -0.8]
    })

    file_path = output_manager.save_dataframe(df, 'test_data', format='csv')

    assert os.path.exists(file_path)
    assert file_path.endswith('.csv')

    # 读取验证
    df_read = pd.read_csv(file_path)
    assert len(df_read) == 2
    assert list(df_read.columns) == ['sector_code', 'sector_name', 'return']

    print(f"✓ CSV文件已保存: {file_path}")


def test_save_dataframe_excel(output_manager):
    """测试保存Excel"""
    df = pd.DataFrame({
        'sector_code': ['801010.SI', '801030.SI'],
        'return': [1.5, -0.8]
    })

    file_path = output_manager.save_dataframe(df, 'test_data', format='excel')

    assert os.path.exists(file_path)
    assert file_path.endswith('.xlsx')

    # 读取验证
    df_read = pd.read_excel(file_path)
    assert len(df_read) == 2

    print(f"✓ Excel文件已保存: {file_path}")


def test_save_plot(output_manager):
    """测试保存图表"""
    import matplotlib.pyplot as plt

    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Test Plot')

    file_path = output_manager.save_plot(fig, 'test_plot')
    plt.close(fig)

    assert os.path.exists(file_path)
    assert file_path.endswith('.png')

    print(f"✓ 图表已保存: {file_path}")


def test_generate_report_basic(output_manager):
    """测试生成基础报告"""
    results = {
        'returns': pd.DataFrame({
            'sector_code': ['801010.SI', '801030.SI', '801010.SI', '801030.SI'],
            'sector_name': ['农林牧渔', '基础化工', '农林牧渔', '基础化工'],
            'trade_date': ['20240102', '20240102', '20240103', '20240103'],
            'return': [1.5, -0.8, 0.5, 1.2]
        })
    }

    metadata = {
        'start_date': '20240101',
        'end_date': '20240131',
        'level': 'L1',
        'period': 'daily',
        'method': 'equal'
    }

    report_path = output_manager.generate_report(results, metadata)

    assert os.path.exists(report_path)
    assert report_path.endswith('.md')

    # 读取报告内容
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert '申万行业板块关系分析报告' in content
    assert '分析参数' in content
    assert '板块涨跌幅统计' in content

    print(f"✓ 报告已生成: {report_path}")
    print(f"✓ 报告长度: {len(content)} 字符")


def test_generate_report_full(output_manager):
    """测试生成完整报告"""
    # 准备完整的分析结果
    returns = pd.DataFrame({
        'sector_code': ['801010.SI', '801030.SI'] * 5,
        'trade_date': ['20240102'] * 2 + ['20240103'] * 2 + ['20240104'] * 2 +
                      ['20240105'] * 2 + ['20240108'] * 2,
        'return': [1.5, -0.8, 0.5, 1.2, -0.3, 0.9, 0.7, -0.5, 1.1, 0.4]
    })

    correlation = pd.DataFrame(
        [[1.0, 0.85], [0.85, 1.0]],
        index=['801010.SI', '801030.SI'],
        columns=['801010.SI', '801030.SI']
    )

    lead_lag = pd.DataFrame({
        'sector_lead': ['801010.SI'],
        'sector_lag': ['801030.SI'],
        'lag_days': [2],
        'correlation': [0.75],
        'p_value': [0.001]
    })

    linkage = pd.DataFrame({
        'sector_a': ['801010.SI'],
        'sector_b': ['801030.SI'],
        'beta': [0.95],
        'r_squared': [0.88],
        'p_value': [0.0001]
    })

    results = {
        'returns': returns,
        'correlation': correlation,
        'lead_lag': lead_lag,
        'linkage': linkage
    }

    metadata = {
        'start_date': '20240101',
        'end_date': '20240131',
        'level': 'L1',
        'period': 'daily',
        'method': 'equal'
    }

    report_path = output_manager.generate_report(results, metadata)

    # 验证报告内容
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert '相关性分析' in content
    assert '传导关系分析' in content
    assert '联动强度分析' in content

    print(f"✓ 完整报告已生成")
    print(f"  - 包含板块涨跌幅统计")
    print(f"  - 包含相关性分析")
    print(f"  - 包含传导关系分析")
    print(f"  - 包含联动强度分析")


def test_generate_report_with_saved_data(output_manager):
    """测试生成报告时列出已保存的数据文件"""
    # 先保存一些数据文件
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    output_manager.save_dataframe(df1, 'test_data1', format='csv')
    output_manager.save_dataframe(df1, 'test_data2', format='excel')

    # 生成报告
    results = {
        'returns': pd.DataFrame({
            'sector_code': ['801010.SI'],
            'trade_date': ['20240102'],
            'return': [1.5]
        })
    }

    report_path = output_manager.generate_report(results)

    # 验证报告中列出了数据文件
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'test_data1.csv' in content
    assert 'test_data2.xlsx' in content

    print("✓ 报告中已列出保存的数据文件")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
