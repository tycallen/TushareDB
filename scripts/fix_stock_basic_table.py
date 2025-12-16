#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 stock_basic 表结构

问题：旧版本创建的 stock_basic 表只有 4 列，缺少很多字段（如 symbol, area, industry 等）
解决：删除旧表，让新架构重新创建完整的表结构

使用方法：
    python scripts/fix_stock_basic_table.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataReader, get_logger

logger = get_logger(__name__)

def main():
    db_path = str(project_root / "tushare.db")
    
    logger.info("=" * 60)
    logger.info("开始修复 stock_basic 表结构")
    logger.info(f"数据库: {db_path}")
    logger.info("=" * 60)
    
    reader = DataReader(db_path=db_path)
    
    try:
        # 1. 检查当前表结构
        if reader.db.table_exists('stock_basic'):
            current_columns = reader.db.get_table_columns('stock_basic')
            logger.info(f"当前表结构（{len(current_columns)} 列）: {current_columns}")
            
            # 2. 备份数据（可选）
            logger.info("正在备份现有数据到 stock_basic_backup...")
            reader.db.con.execute("DROP TABLE IF EXISTS stock_basic_backup")
            reader.db.con.execute("CREATE TABLE stock_basic_backup AS SELECT * FROM stock_basic")
            backup_count = reader.db.con.execute("SELECT COUNT(*) FROM stock_basic_backup").fetchone()[0]
            logger.info(f"✓ 已备份 {backup_count} 条记录到 stock_basic_backup 表")
            
            # 3. 删除旧表
            logger.info("正在删除旧的 stock_basic 表...")
            reader.db.con.execute("DROP TABLE stock_basic")
            logger.info("✓ 旧表已删除")
        else:
            logger.info("stock_basic 表不存在，无需修复")
            return
        
        logger.info("=" * 60)
        logger.info("✓ 修复完成！")
        logger.info("下一步：运行 python scripts/update_daily.py 重新下载股票列表")
        logger.info("=" * 60)
        
    finally:
        reader.close()

if __name__ == "__main__":
    main()
