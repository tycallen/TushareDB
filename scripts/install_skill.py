#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code Skill 安装脚本

将本项目的 tushare-duckdb skill 安装到 Claude Code 的 skills 目录

安装位置选项：
  1. 全局安装：~/.claude/skills/tushare-duckdb/
  2. 项目安装：.claude/skills/tushare-duckdb/

使用方法：
    python scripts/install_skill.py [--global | --project]

默认行为：
    - 如果未指定选项，会提示选择安装位置
    - 如果目标目录已存在，会提示是否覆盖

作者：Tushare-DuckDB
日期：2026-01-31
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def get_skill_source_dir() -> Path:
    """获取 skill 源目录"""
    return get_project_root() / "docs" / "skills" / "tushare-duckdb"


def get_global_skill_dir() -> Path:
    """获取全局 skill 安装目录"""
    return Path.home() / ".claude" / "skills" / "tushare-duckdb"


def get_project_skill_dir() -> Path:
    """获取项目 skill 安装目录"""
    return get_project_root() / ".claude" / "skills" / "tushare-duckdb"


def install_skill(target_dir: Path, force: bool = False) -> bool:
    """
    安装 skill 到目标目录

    Args:
        target_dir: 目标安装目录
        force: 是否强制覆盖

    Returns:
        是否安装成功
    """
    source_dir = get_skill_source_dir()

    # 检查源目录是否存在
    if not source_dir.exists():
        print(f"错误：skill 源目录不存在: {source_dir}")
        return False

    # 检查必要文件
    skill_file = source_dir / "SKILL.md"
    if not skill_file.exists():
        print(f"错误：skill 文件不存在: {skill_file}")
        return False

    # 检查目标目录是否存在
    if target_dir.exists():
        if not force:
            response = input(f"目标目录已存在: {target_dir}\n是否覆盖？[y/N] ").strip().lower()
            if response != 'y':
                print("取消安装")
                return False

        # 删除旧目录
        shutil.rmtree(target_dir)
        print(f"已删除旧目录: {target_dir}")

    # 创建父目录
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # 复制整个目录
    shutil.copytree(source_dir, target_dir)

    print(f"✓ Skill 已安装到: {target_dir}")
    return True


def uninstall_skill(target_dir: Path) -> bool:
    """
    卸载 skill

    Args:
        target_dir: 目标目录

    Returns:
        是否卸载成功
    """
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return False

    response = input(f"确认删除 skill 目录？{target_dir}\n[y/N] ").strip().lower()
    if response != 'y':
        print("取消卸载")
        return False

    shutil.rmtree(target_dir)
    print(f"✓ Skill 已卸载: {target_dir}")
    return True


def show_status():
    """显示当前安装状态"""
    source_dir = get_skill_source_dir()
    global_dir = get_global_skill_dir()
    project_dir = get_project_skill_dir()

    print("=" * 60)
    print("Tushare-DuckDB Skill 安装状态")
    print("=" * 60)

    print(f"\n源目录: {source_dir}")
    if source_dir.exists():
        skill_file = source_dir / "SKILL.md"
        print(f"  状态: ✓ 存在")
        print(f"  SKILL.md: {'✓ 存在' if skill_file.exists() else '✗ 不存在'}")
    else:
        print(f"  状态: ✗ 不存在")

    print(f"\n全局安装: {global_dir}")
    if global_dir.exists():
        print(f"  状态: ✓ 已安装")
    else:
        print(f"  状态: ✗ 未安装")

    print(f"\n项目安装: {project_dir}")
    if project_dir.exists():
        print(f"  状态: ✓ 已安装")
    else:
        print(f"  状态: ✗ 未安装")

    print("\n" + "=" * 60)


def interactive_install():
    """交互式安装"""
    print("=" * 60)
    print("Tushare-DuckDB Skill 安装向导")
    print("=" * 60)

    print("\n选择安装位置：")
    print("  1. 全局安装 (~/.claude/skills/)")
    print("     - 所有项目都可使用")
    print("     - 推荐日常使用")
    print()
    print("  2. 项目安装 (.claude/skills/)")
    print("     - 仅当前项目可使用")
    print("     - 适合团队协作（可提交到 Git）")
    print()
    print("  3. 查看当前状态")
    print()
    print("  0. 退出")

    while True:
        choice = input("\n请选择 [1/2/3/0]: ").strip()

        if choice == '1':
            if install_skill(get_global_skill_dir()):
                print("\n安装完成！")
                print("现在可以在 Claude Code 中使用 /tushare-duckdb 命令")
            break

        elif choice == '2':
            if install_skill(get_project_skill_dir()):
                print("\n安装完成！")
                print("Skill 已安装到项目目录，可以提交到 Git")
            break

        elif choice == '3':
            show_status()

        elif choice == '0':
            print("退出")
            break

        else:
            print("无效选择，请重新输入")


def main():
    parser = argparse.ArgumentParser(
        description="安装/卸载 Tushare-DuckDB Claude Code Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/install_skill.py              # 交互式安装
  python scripts/install_skill.py --global     # 全局安装
  python scripts/install_skill.py --project    # 项目安装
  python scripts/install_skill.py --status     # 查看状态
  python scripts/install_skill.py --uninstall --global  # 卸载全局安装
        """
    )

    # 安装位置选项（互斥）
    location_group = parser.add_mutually_exclusive_group()
    location_group.add_argument(
        "--global", "-g",
        dest="global_install",
        action="store_true",
        help="全局安装到 ~/.claude/skills/"
    )
    location_group.add_argument(
        "--project", "-p",
        dest="project_install",
        action="store_true",
        help="项目安装到 .claude/skills/"
    )

    # 其他选项
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制覆盖已存在的安装"
    )
    parser.add_argument(
        "--uninstall", "-u",
        action="store_true",
        help="卸载 skill"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="显示安装状态"
    )

    args = parser.parse_args()

    # 显示状态
    if args.status:
        show_status()
        return 0

    # 确定目标目录
    if args.global_install:
        target_dir = get_global_skill_dir()
    elif args.project_install:
        target_dir = get_project_skill_dir()
    else:
        # 没有指定位置，进入交互模式
        interactive_install()
        return 0

    # 执行安装/卸载
    if args.uninstall:
        success = uninstall_skill(target_dir)
    else:
        success = install_skill(target_dir, force=args.force)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
