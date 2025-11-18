#!/usr/bin/env python3
"""
MCP Server for Tushare-DuckDB
让 Claude Desktop 直接访问股票数据

配置方法：
1. 在 Claude Desktop 设置中添加此 MCP Server
2. 配置文件路径（通常在 ~/Library/Application Support/Claude/claude_desktop_config.json）
3. 添加以下配置：
{
  "mcpServers": {
    "tushare-db": {
      "command": "python3",
      "args": ["/Users/allen/workspace/python/stock/Tushare-DuckDB/mcp_server.py"]
    }
  }
}
"""

import json
import sys
from typing import Any
from tushare_db import DataReader

# MCP Server 基础实现
class TushareDBMCPServer:
    def __init__(self):
        self.reader = DataReader()

    def handle_request(self, request: dict) -> dict:
        """处理MCP请求"""
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "query_stock_daily":
                df = self.reader.get_stock_daily(
                    params["ts_code"],
                    params["start_date"],
                    params["end_date"],
                    adj=params.get("adj")
                )
                return {"result": df.to_dict('records')}

            elif method == "query_stock_basic":
                df = self.reader.get_stock_basic(
                    ts_code=params.get("ts_code"),
                    list_status=params.get("list_status", 'L')
                )
                return {"result": df.to_dict('records')}

            elif method == "query_trade_calendar":
                df = self.reader.get_trade_calendar(
                    start_date=params.get("start_date"),
                    end_date=params.get("end_date"),
                    is_open=params.get("is_open")
                )
                return {"result": df.to_dict('records')}

            elif method == "custom_query":
                df = self.reader.query(
                    params["sql"],
                    params.get("params")
                )
                return {"result": df.to_dict('records')}

            elif method == "list_tables":
                tables = [
                    "stock_basic", "pro_bar", "adj_factor", "daily_basic",
                    "trade_cal", "stock_company", "cyq_perf", "stk_factor_pro"
                ]
                return {"result": tables}

            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e)}

    def run(self):
        """运行MCP Server"""
        for line in sys.stdin:
            request = json.loads(line)
            response = self.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()

if __name__ == "__main__":
    server = TushareDBMCPServer()
    server.run()
