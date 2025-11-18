#!/bin/bash
# Tushare-DuckDB 快速查看工具
# 功能：启动服务器并在浏览器中打开指定股票的K线图页面
#
# 用法:
#   ./quick_view.sh 000001.SZ  # 查看平安银行
#   ./quick_view.sh 600000.SH  # 查看浦发银行

set -e

# 配置
BACKEND_PORT=8000
FRONTEND_PORT=5173
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ -z "$1" ]; then
    log_error "请提供股票代码！"
    echo ""
    echo "用法示例:"
    echo "  $0 000001.SZ  # 深圳股票"
    echo "  $0 600000.SH  # 上海股票"
    echo ""
    exit 1
fi

STOCK_CODE="$1"
log_info "准备查看股票: $STOCK_CODE"

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 启动后端服务器
start_backend() {
    if check_port $BACKEND_PORT; then
        log_warn "后端服务器已在运行 (端口 $BACKEND_PORT)"
    else
        log_info "启动后端服务器..."
        cd "$BACKEND_DIR"
        nohup uvicorn src.tushare_db.web_server:app --host 0.0.0.0 --port $BACKEND_PORT > backend.log 2>&1 &
        echo $! > backend.pid
        log_success "后端服务器已启动 (PID: $(cat backend.pid))"

        # 等待后端启动
        log_info "等待后端服务器就绪..."
        for i in {1..30}; do
            if curl -s http://localhost:$BACKEND_PORT/ > /dev/null 2>&1; then
                log_success "后端服务器就绪！"
                break
            fi
            sleep 1
            if [ $i -eq 30 ]; then
                log_error "后端服务器启动超时"
                exit 1
            fi
        done
    fi
}

# 启动前端服务器
start_frontend() {
    if check_port $FRONTEND_PORT; then
        log_warn "前端服务器已在运行 (端口 $FRONTEND_PORT)"
    else
        log_info "启动前端服务器..."
        cd "$FRONTEND_DIR"

        # 检查 node_modules
        if [ ! -d "node_modules" ]; then
            log_info "首次运行，安装前端依赖..."
            npm install
        fi

        nohup npm run dev > frontend.log 2>&1 &
        echo $! > frontend.pid
        log_success "前端服务器已启动 (PID: $(cat frontend.pid))"

        # 等待前端启动
        log_info "等待前端服务器就绪..."
        for i in {1..30}; do
            if curl -s http://localhost:$FRONTEND_PORT/ > /dev/null 2>&1; then
                log_success "前端服务器就绪！"
                break
            fi
            sleep 1
            if [ $i -eq 30 ]; then
                log_error "前端服务器启动超时"
                exit 1
            fi
        done
    fi
}

# 在浏览器中打开页面
open_browser() {
    local url="http://localhost:$FRONTEND_PORT/stock/$STOCK_CODE"
    log_info "在浏览器中打开: $url"

    # 根据操作系统选择打开方式
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$url"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "$url" 2>/dev/null || firefox "$url" 2>/dev/null || google-chrome "$url" 2>/dev/null
    else
        log_warn "无法自动打开浏览器，请手动访问: $url"
    fi

    log_success "完成！"
}

# 显示停止服务器的提示
show_stop_hint() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}服务器运行中${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "• 后端API:  http://localhost:$BACKEND_PORT"
    echo "• 前端页面: http://localhost:$FRONTEND_PORT"
    echo "• 当前股票: http://localhost:$FRONTEND_PORT/stock/$STOCK_CODE"
    echo ""
    echo "停止服务器："
    echo "  ./stop_servers.sh"
    echo ""
    echo "查看其他股票："
    echo "  ./quick_view.sh <股票代码>"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# 主流程
main() {
    log_info "Tushare-DuckDB 快速查看工具"
    echo ""

    start_backend
    start_frontend

    sleep 2  # 给服务器一点额外时间完全就绪
    open_browser

    show_stop_hint
}

# 执行主流程
main
