# Tushare-DuckDB 快速启动指南

## 🚀 一键查看股票K线图

### 方式1: 使用股票代码（推荐）

```bash
./quick_view.sh 000001.SZ   # 查看平安银行
./quick_view.sh 600000.SH   # 查看浦发银行
./quick_view.sh 600519.SH   # 查看贵州茅台
```

### 方式2: 使用股票名称（更简单）

```bash
./view_stock.sh 平安银行
./view_stock.sh 招商银行
./view_stock.sh 贵州茅台
```

### 功能说明

脚本会自动完成：
1. ✅ 启动后端 API 服务器（端口 8000）
2. ✅ 启动前端页面服务器（端口 5173）
3. ✅ 在浏览器中打开指定股票的K线图页面

### 停止服务器

```bash
./stop_servers.sh
```

---

## 📝 支持的快捷名称

当前支持的股票名称（可编辑 `view_stock.sh` 添加更多）：

| 名称 | 代码 | 类型 |
|------|------|------|
| 平安银行 | 000001.SZ | 深圳 |
| 万科A | 000002.SZ | 深圳 |
| 五粮液 | 000858.SZ | 深圳 |
| 招商银行 | 600036.SH | 上海 |
| 浦发银行 | 600000.SH | 上海 |
| 贵州茅台 | 600519.SH | 上海 |
| 工商银行 | 601398.SH | 上海 |
| 建设银行 | 601939.SH | 上海 |
| 中国平安 | 601318.SH | 上海 |
| 中国石油 | 601857.SH | 上海 |
| 中国石化 | 600028.SH | 上海 |

---

## 🔧 首次使用

**1. 确保已安装依赖：**
```bash
# Python 依赖（如果还没安装）
pip install -r requirements.txt

# 前端依赖（脚本会自动检查并安装）
cd frontend && npm install
```

**2. 运行脚本：**
```bash
./quick_view.sh 000001.SZ
```

首次运行会：
- 自动安装前端依赖（如果需要）
- 启动后端和前端服务器
- 在浏览器中打开股票详情页

---

## 📖 使用示例

### 示例 1: 快速查看单只股票
```bash
$ ./view_stock.sh 贵州茅台

[INFO] 准备查看股票: 600519.SH
[SUCCESS] 后端服务器已启动
[SUCCESS] 前端服务器已启动
[INFO] 在浏览器中打开: http://localhost:5173/stock/600519.SH
```

### 示例 2: 切换查看不同股票
```bash
# 服务器保持运行，直接查看另一只股票
./quick_view.sh 600036.SH   # 查看招商银行
```

浏览器会自动打开新页面，服务器无需重启。

### 示例 3: 完成后停止服务器
```bash
./stop_servers.sh

[INFO] 停止后端服务器 (PID: 12345)...
[SUCCESS] 后端服务器已停止
[INFO] 停止前端服务器 (PID: 12346)...
[SUCCESS] 前端服务器已停止
```

---

## 🌐 手动访问

如果服务器已在运行，也可以直接在浏览器中访问：

- **首页**: http://localhost:5173/
- **股票详情**: http://localhost:5173/stock/000001.SZ
- **API文档**: http://localhost:8000/docs
- **后端健康检查**: http://localhost:8000/

---

## 🛠️ 高级用法

### 查看日志

```bash
# 后端日志
tail -f backend.log

# 前端日志
tail -f frontend/frontend.log
```

### 手动启动服务器

```bash
# 启动后端
uvicorn src.tushare_db.web_server:app --host 0.0.0.0 --port 8000

# 启动前端（另一个终端）
cd frontend && npm run dev
```

### 修改端口

编辑 `quick_view.sh` 文件中的配置：
```bash
BACKEND_PORT=8000   # 修改为其他端口
FRONTEND_PORT=5173  # 修改为其他端口
```

---

## ❓ 常见问题

### Q: 端口已被占用怎么办？
A: 脚本会自动检测，如果端口已被占用会复用现有服务器。如果需要重启，先运行 `./stop_servers.sh`

### Q: 如何添加更多快捷股票名称？
A: 编辑 `view_stock.sh`，在 `STOCKS` 数组中添加：
```bash
["新股票名"]="000000.XX"
```

### Q: 浏览器没有自动打开？
A: 手动访问 http://localhost:5173/stock/<股票代码>

### Q: 前端显示数据加载失败？
A: 检查后端是否正常运行：
```bash
curl http://localhost:8000/api/stock_basic?list_status=L
```

---

## 📚 相关文档

- [API 参考文档](API_REFERENCE_FOR_LLM.md) - DataReader/DataDownloader 详细用法
- [AI 快速参考](README_FOR_AI.md) - 给 AI 助手阅读的简明文档
- [迁移指南](MIGRATION_GUIDE.md) - 从旧架构迁移到新架构
- [工作总结](WORK_SUMMARY.md) - 项目重构工作记录

---

**Enjoy! 🎉**
