# Gemini Agent - 项目备忘录

本文档记录了在使用 Gemini Agent 开发和维护此项目过程中遇到的一些关键问题、解决方案和重要决策。

## 1. 动态数据表创建

- **问题**: 在项目初始化时，执行 `scripts/init_data.py` 脚本失败，日志显示 `duckdb.duckdb.CatalogException: Table with name ... does not exist!`。
- **分析**: `tushare_db` 库的 `duckdb_manager.py` 模块中，`write_dataframe` 方法在以 `append` 模式写入数据时，没有处理数据表不存在的情况，错误地假设了 `INSERT INTO` 语句会自动建表。
- **解决方案**: 修改了 `src/tushare_db/duckdb_manager.py` 中的 `write_dataframe` 方法。在 `append` 模式下，首先调用 `table_exists()` 检查表是否存在。如果不存在，则先通过 `CREATE TABLE ... AS SELECT ...` 的方式创建该表，然后再插入数据，从而解决了该 bug。

## 2. API 参数误用作 SQL 查询条件

- **问题**: 修复上一个问题后，再次运行脚本，出现 `duckdb.duckdb.BinderException: Referenced column "fields" not found` 错误。
- **分析**: `cache_policies.py` 中的 `_build_where_clause` 方法在构建 SQL `WHERE` 子句时，将所有传入的 API 参数（例如用于控制返回字段的 `fields` 参数）都错误地当作了数据库表的列名进行查询，导致了 SQL 错误。
- **解决方案**: 重写了 `src/tushare_db/cache_policies.py` 中的 `_build_where_clause` 方法。新的实现会先通过 `duckdb_manager.get_table_columns()` 获取表的实际列名列表，然后只将那些在表中真实存在的参数构建到 `WHERE` 子句中，忽略了如 `fields` 这类的 API 控制参数。

## 3. `pro_bar` 接口的特殊处理

- **问题**: `pro_bar` 接口在一次性获取多只股票数据时，Tushare 官方推荐的做法是循环单只股票进行查询，而项目之前的 `fetch` 方法无法处理这种情况。
- **分析**: `tushare_fetcher.py` 中的 `fetch` 方法是一个通用方法，它将 `ts_code` 列表或逗号分隔的字符串直接传递给 Tushare API，这对于 `pro_bar` 接口是无效的。
- **解决方案**: 修改了 `src/tushare_db/tushare_fetcher.py` 中的 `fetch` 方法。增加了特殊逻辑判断：如果 `api_name` 是 `pro_bar` 且 `ts_code` 是多值，则自动将其拆分，并循环调用 Tushare API，最后将所有结果合并为一个 DataFrame。这使得该接口的调用对上层用户保持透明。

## 4. 抑制第三方库的 `FutureWarning`

- **问题**: 运行脚本时，控制台输出 `FutureWarning: Series.fillna with 'method' is deprecated...`。
- **分析**: 此警告来自于 `tushare` 库的 `data_pro.py` 文件，是由于 `tushare` 内部使用了 `pandas` 中一个即将被废弃的 `fillna` 方法。这是一个第三方库的问题，我们无法直接修改其源代码。
- **解决方案**: 为了保持控制台输出的整洁，在我们的入口脚本 (`scripts/init_data.py` 和 `scripts/update_daily.py`) 的顶部，使用 Python 的 `warnings` 模块来显式地忽略这个特定的 `FutureWarning`。代码如下：
  ```python
  import warnings
  warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")
  ```
  这可以在不影响功能的情况下，提供更好的用户体验。
