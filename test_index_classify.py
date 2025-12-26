#!/usr/bin/env python
"""
申万行业分类（index_classify）接口测试脚本
"""
from tushare_db import DataDownloader, DataReader
import pandas as pd


def test_download_index_classify():
    """测试下载申万行业分类数据"""
    print("=" * 70)
    print("测试下载申万行业分类数据")
    print("=" * 70)

    downloader = DataDownloader()

    try:
        # 测试1：下载申万2021版L1一级行业
        print("\n【测试1】下载申万2021版L1一级行业")
        print("-" * 70)
        rows = downloader.download_index_classify(level='L1', src='SW2021')
        print(f"✓ 成功下载 {rows} 条数据")

        # 测试2：下载申万2021版L2二级行业
        print("\n【测试2】下载申万2021版L2二级行业")
        print("-" * 70)
        rows = downloader.download_index_classify(level='L2', src='SW2021')
        print(f"✓ 成功下载 {rows} 条数据")

        # 测试3：下载申万2021版L3三级行业
        print("\n【测试3】下载申万2021版L3三级行业")
        print("-" * 70)
        rows = downloader.download_index_classify(level='L3', src='SW2021')
        print(f"✓ 成功下载 {rows} 条数据")

        # 测试4：下载申万2014版L1一级行业
        print("\n【测试4】下载申万2014版L1一级行业")
        print("-" * 70)
        rows = downloader.download_index_classify(level='L1', src='SW2014')
        print(f"✓ 成功下载 {rows} 条数据")

        print("\n" + "=" * 70)
        print("下载测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_query_index_classify():
    """测试查询申万行业分类数据"""
    print("\n" * 2)
    print("=" * 70)
    print("测试查询申万行业分类数据")
    print("=" * 70)

    reader = DataReader()

    try:
        # 测试1：查询所有L1一级行业
        print("\n【测试1】查询申万2021版L1一级行业")
        print("-" * 70)
        df = reader.get_index_classify(level='L1', src='SW2021')

        if not df.empty:
            print(f"查询结果: {len(df)} 条数据\n")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(df.to_string(index=False))
        else:
            print("✗ 未查询到数据")

        # 测试2：查询所有L2二级行业
        print("\n【测试2】查询申万2021版L2二级行业（显示前10条）")
        print("-" * 70)
        df = reader.get_index_classify(level='L2', src='SW2021')

        if not df.empty:
            print(f"查询结果: {len(df)} 条数据\n")
            print(df.head(10).to_string(index=False))
            print(f"\n... 共 {len(df)} 条数据")
        else:
            print("✗ 未查询到数据")

        # 测试3：查询特定一级行业下的二级行业
        print("\n【测试3】查询特定L1行业下的L2二级行业")
        print("-" * 70)

        # 先获取一个L1行业代码
        df_l1 = reader.get_index_classify(level='L1', src='SW2021')
        if not df_l1.empty:
            parent_code = df_l1.iloc[0]['industry_code']
            parent_name = df_l1.iloc[0]['industry_name']

            print(f"父级行业: {parent_code} - {parent_name}")

            df = reader.get_index_classify(level='L2', parent_code=parent_code, src='SW2021')

            if not df.empty:
                print(f"\n查询结果: {len(df)} 条二级行业\n")
                print(df[['industry_code', 'industry_name', 'parent_code', 'level']].to_string(index=False))
            else:
                print("✗ 该一级行业下无二级行业")
        else:
            print("✗ 未找到L1行业数据")

        # 测试4：比较SW2014和SW2021版本的差异
        print("\n【测试4】比较SW2014和SW2021版本")
        print("-" * 70)
        df_2014 = reader.get_index_classify(level='L1', src='SW2014')
        df_2021 = reader.get_index_classify(level='L1', src='SW2021')

        print(f"SW2014版 L1一级行业数量: {len(df_2014)}")
        print(f"SW2021版 L1一级行业数量: {len(df_2021)}")

        if not df_2014.empty and not df_2021.empty:
            print("\nSW2014版 L1行业列表:")
            print(df_2014[['industry_code', 'industry_name']].to_string(index=False))
            print("\nSW2021版 L1行业列表:")
            print(df_2021[['industry_code', 'industry_name']].to_string(index=False))

        # 测试5：数据字段完整性检查
        print("\n【测试5】数据字段完整性检查")
        print("-" * 70)

        expected_columns = [
            'index_code', 'industry_name', 'parent_code',
            'level', 'industry_code', 'is_pub', 'src'
        ]

        df_sample = reader.get_index_classify(level='L1', src='SW2021')
        if not df_sample.empty:
            print(f"数据库中的字段:")
            for i, col in enumerate(df_sample.columns, 1):
                status = "✓" if col in expected_columns else "?"
                print(f"  {status} {i:2d}. {col}")

            missing_cols = set(expected_columns) - set(df_sample.columns)
            if missing_cols:
                print(f"\n缺少的字段: {', '.join(missing_cols)}")
            else:
                print(f"\n✓ 所有预期字段都存在")

        print("\n" + "=" * 70)
        print("查询测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


def main():
    """主函数"""
    print("\n")
    print("#" * 70)
    print("# 申万行业分类（index_classify）接口测试")
    print("#" * 70)

    # 测试下载功能
    test_download_index_classify()

    # 测试查询功能
    test_query_index_classify()

    print("\n")
    print("#" * 70)
    print("# 测试完成！")
    print("#" * 70)
    print("\n")


if __name__ == '__main__':
    main()
