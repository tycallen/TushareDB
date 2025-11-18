#!/bin/bash
# 停止 Tushare-DuckDB 服务器
# 功能：停止后端和前端服务器

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 停止后端
stop_backend() {
    if [ -f "$BACKEND_DIR/backend.pid" ]; then
        local pid=$(cat "$BACKEND_DIR/backend.pid")
        if kill -0 $pid 2>/dev/null; then
            log_info "停止后端服务器 (PID: $pid)..."
            kill $pid
            rm "$BACKEND_DIR/backend.pid"
            log_success "后端服务器已停止"
        else
            log_warn "后端服务器进程不存在 (PID: $pid)"
            rm "$BACKEND_DIR/backend.pid"
        fi
    else
        log_warn "未找到后端服务器 PID 文件"
    fi
}

# 停止前端
stop_frontend() {
    if [ -f "$FRONTEND_DIR/frontend.pid" ]; then
        local pid=$(cat "$FRONTEND_DIR/frontend.pid")
        if kill -0 $pid 2>/dev/null; then
            log_info "停止前端服务器 (PID: $pid)..."
            kill $pid
            rm "$FRONTEND_DIR/frontend.pid"
            log_success "前端服务器已停止"
        else
            log_warn "前端服务器进程不存在 (PID: $pid)"
            rm "$FRONTEND_DIR/frontend.pid"
        fi
    else
        log_warn "未找到前端服务器 PID 文件"
    fi
}

# 备用方案：通过端口强制停止
force_stop_by_port() {
    log_info "尝试通过端口强制停止服务器..."

    # 停止 8000 端口的进程（后端）
    local backend_pids=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$backend_pids" ]; then
        log_info "发现后端进程: $backend_pids"
        kill $backend_pids 2>/dev/null || true
        log_success "后端服务器已停止"
    fi

    # 停止 5173 端口的进程（前端）
    local frontend_pids=$(lsof -ti:5173 2>/dev/null || true)
    if [ -n "$frontend_pids" ]; then
        log_info "发现前端进程: $frontend_pids"
        kill $frontend_pids 2>/dev/null || true
        log_success "前端服务器已停止"
    fi

    if [ -z "$backend_pids" ] && [ -z "$frontend_pids" ]; then
        log_warn "未发现运行中的服务器"
    fi
}

# 主流程
main() {
    log_info "停止 Tushare-DuckDB 服务器..."
    echo ""

    stop_backend
    stop_frontend

    # 如果 PID 方式失败，尝试通过端口强制停止
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 || lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "部分服务器仍在运行，使用强制停止..."
        force_stop_by_port
    fi

    echo ""
    log_success "所有服务器已停止"
}

main
