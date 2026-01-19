#!/usr/bin/env python3
"""
Validates that docs/skills/tushare-duckdb.md stays in sync with TABLE_PRIMARY_KEYS.

Run in CI or as pre-commit hook:
    python scripts/validate_skill_sync.py

Exit codes:
    0 - Skill file is in sync
    1 - Skill file is out of sync (missing or extra tables)
    2 - Skill file not found
"""
import sys
import re
from pathlib import Path

# Add src to path for importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tushare_db.duckdb_manager import TABLE_PRIMARY_KEYS

SKILL_PATH = PROJECT_ROOT / "docs/skills/tushare-duckdb/SKILL.md"


def extract_tables_from_skill(content: str) -> set:
    """Extract table names from markdown table in skill file.

    Matches pattern: | `table_name` | ... in the Implemented Tables section.
    """
    pattern = r'\| `(\w+)` \|'
    return set(re.findall(pattern, content))


def validate() -> int:
    """Validate skill file is in sync with TABLE_PRIMARY_KEYS.

    Returns:
        0 if in sync, 1 if out of sync, 2 if skill file not found.
    """
    if not SKILL_PATH.exists():
        print(f"Error: Skill file not found at {SKILL_PATH}")
        return 2

    skill_content = SKILL_PATH.read_text(encoding='utf-8')
    skill_tables = extract_tables_from_skill(skill_content)
    code_tables = set(TABLE_PRIMARY_KEYS.keys())

    missing_in_skill = code_tables - skill_tables
    extra_in_skill = skill_tables - code_tables

    if missing_in_skill or extra_in_skill:
        print("Skill file out of sync with TABLE_PRIMARY_KEYS!")
        print()
        if missing_in_skill:
            print(f"  Tables in code but missing in skill ({len(missing_in_skill)}):")
            for table in sorted(missing_in_skill):
                pk = TABLE_PRIMARY_KEYS[table]
                print(f"    - {table} (PK: {', '.join(pk)})")
        if extra_in_skill:
            print(f"  Tables in skill but not in code ({len(extra_in_skill)}):")
            for table in sorted(extra_in_skill):
                print(f"    - {table}")
        print()
        print("Please update docs/skills/tushare-duckdb.md to match TABLE_PRIMARY_KEYS.")
        return 1

    print(f"Skill file in sync with TABLE_PRIMARY_KEYS ({len(code_tables)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
