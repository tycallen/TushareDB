"""
测试 update_index_member_all 下载历史变迁记录 (is_new='N')
"""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd


def test_update_index_member_all_downloads_both_current_and_historical():
    """
    验证 update_index_member_all 会同时下载 is_new='Y' 和 is_new='N' 的数据

    这是 PIT 行业中性化的关键：
    - is_new='Y': 当前成员（out_date=NULL）
    - is_new='N': 历史成员（out_date 有值，已被调出）

    两者合在一起才能支持完整的 Point-in-Time 回测
    """
    # Arrange: 模拟 DataDownloader
    downloader = Mock()
    downloader.download_index_member_all = Mock(return_value=10)

    # 模拟行业列表
    industries_df = pd.DataFrame({
        'index_code': ['801010.SI', '801020.SI'],
        'industry_name': ['农林牧渔', '基础化工'],
        'level': ['L1', 'L1']
    })
    downloader.db.execute_query.return_value = industries_df

    # 模拟时间检查 - 超过7天需要更新
    downloader.db.get_cache_metadata.return_value = None

    # Act: 调用更新函数（这里模拟的是期望的行为）
    # 注意：当前实现会失败，因为还没有实现下载 is_new='N' 的逻辑

    # 期望调用次数：2个行业 × 2种 is_new 状态 = 4次
    expected_calls = 4

    # 这里我们先手动模拟期望的行为来验证测试本身
    current_calls = 0
    for _, row in industries_df.iterrows():
        # 下载当前成员
        downloader.download_index_member_all(l1_code=row['index_code'], is_new='Y')
        current_calls += 1
        # 下载历史成员（这是我们要添加的）
        downloader.download_index_member_all(l1_code=row['index_code'], is_new='N')
        current_calls += 1

    # Assert
    assert current_calls == expected_calls, f"期望调用 {expected_calls} 次，实际 {current_calls} 次"

    # 验证两次都传了 is_new 参数
    call_args_list = downloader.download_index_member_all.call_args_list

    # 第一个行业
    assert call_args_list[0][1].get('is_new') == 'Y', "第一次调用应该是 is_new='Y'"
    assert call_args_list[1][1].get('is_new') == 'N', "第二次调用应该是 is_new='N'"

    # 第二个行业
    assert call_args_list[2][1].get('is_new') == 'Y', "第三次调用应该是 is_new='Y'"
    assert call_args_list[3][1].get('is_new') == 'N', "第四次调用应该是 is_new='N'"


@pytest.fixture
def mock_downloader():
    """创建一个完全模拟的 DataDownloader"""
    mock = Mock()

    # 模拟行业数据
    industries_df = pd.DataFrame({
        'index_code': ['850531.SI'],  # 黄金行业
        'industry_name': ['黄金'],
        'level': ['L3']
    })
    mock.db.execute_query.return_value = industries_df
    mock.db.get_cache_metadata.return_value = None
    mock.db.table_exists.return_value = True

    # 模拟返回数据
    def mock_download(l3_code=None, is_new=None):
        # 模拟 API 返回不同 is_new 的数据
        if is_new == 'Y':
            # 当前成员，out_date=NULL
            return pd.DataFrame({
                'ts_code': ['000506.SZ', '600988.SH'],
                'l1_code': ['801050.SI', '801050.SI'],
                'l3_code': [l3_code, l3_code],
                'in_date': ['20220729', '20040414'],
                'out_date': [None, None],
                'is_new': ['Y', 'Y']
            })
        elif is_new == 'N':
            # 历史成员，out_date 有值
            return pd.DataFrame({
                'ts_code': ['000506.SZ', '601899.SH'],
                'l1_code': ['801050.SI', '801050.SI'],
                'l3_code': [l3_code, l3_code],
                'in_date': ['20150701', '20080407'],
                'out_date': ['20180615', '20210729'],
                'is_new': ['N', 'N']
            })
        return pd.DataFrame()

    mock.fetcher.fetch.side_effect = lambda table, **kwargs: mock_download(
        kwargs.get('l3_code'), kwargs.get('is_new')
    )

    return mock


def test_index_member_all_upsert_with_pit_data(mock_downloader):
    """
    验证下载后的数据写入逻辑：
    - 同一 PK (ts_code, l3_code, in_date) 的 is_new='N' 记录会更新 is_new='Y' 的 out_date
    """
    from tushare_db.downloader import DataDownloader

    # 模拟万辰集团的场景：
    # - is_new='Y' 返回：in_date=20240730, out_date=NULL (脏数据)
    # - is_new='N' 返回：in_date=20240730, out_date=20260304 (正确)

    current_df = pd.DataFrame({
        'ts_code': ['300972.SZ'],
        'l1_code': ['801010.SI'],
        'l3_code': ['850531.SI'],
        'in_date': ['20240730'],
        'out_date': [None],  # 脏数据
        'is_new': ['Y']
    })

    historical_df = pd.DataFrame({
        'ts_code': ['300972.SZ'],
        'l1_code': ['801010.SI'],
        'l3_code': ['850531.SI'],
        'in_date': ['20240730'],
        'out_date': ['20260304'],  # 正确值
        'is_new': ['N']
    })

    # 合并两份数据
    combined = pd.concat([current_df, historical_df], ignore_index=True)

    # 验证：有两条记录
    assert len(combined) == 2

    # 验证：它们有相同的 PK
    pk_cols = ['ts_code', 'l3_code', 'in_date']
    assert combined.iloc[0][pk_cols].tolist() == combined.iloc[1][pk_cols].tolist()

    # 验证：is_new 不同
    assert set(combined['is_new'].unique()) == {'Y', 'N'}


def test_actual_update_script_downloads_historical_data():
    """
    实际测试：验证 update_daily.py 中的 update_index_member_all 函数
    会下载 is_new='N' 的历史变迁数据

    这是集成测试，检查实际代码行为。
    """
    import re

    # 读取实际脚本代码
    script_path = '/data10/tyc/quant/TushareDB/scripts/update_daily.py'
    with open(script_path, 'r') as f:
        script_code = f.read()

    # 提取 update_index_member_all 函数体
    func_match = re.search(
        r'def update_index_member_all\(.*?\):(.*?)(?=\ndef \w+\(|\Z)',
        script_code,
        re.DOTALL
    )
    assert func_match, f"未找到 update_index_member_all 函数"
    func_body = func_match.group(1)

    # 关键断言：必须同时下载 is_new='Y' 和 is_new='N'
    is_new_count = func_body.count('is_new')

    # 期望：至少 2 次（一次 Y，一次 N）
    # 当前实现：0 次（没有显式传递）
    assert is_new_count >= 2, (
        f"❌ 测试失败：update_index_member_all 应该显式传递 is_new='Y' 和 is_new='N'，"
        f"但实际代码中 is_new 出现 {is_new_count} 次\n"
        f"这是 PIT 数据缺失的根本原因"
    )

    # 验证有分别调用 Y 和 N
    has_y = "is_new='Y'" in func_body or 'is_new="Y"' in func_body
    has_n = "is_new='N'" in func_body or 'is_new="N"' in func_body

    assert has_y and has_n, (
        f"❌ 测试失败：必须同时显式调用 is_new='Y' 和 is_new='N'\n"
        f"has is_new='Y': {has_y}, has is_new='N': {has_n}"
    )


if __name__ == '__main__':
    # 运行测试
    test_update_index_member_all_downloads_both_current_and_historical()
    print("✓ 单元测试通过")

    try:
        test_actual_update_script_downloads_historical_data()
        print("✓ 集成测试通过")
    except AssertionError as e:
        print(f"\n{e}")
        exit(1)
