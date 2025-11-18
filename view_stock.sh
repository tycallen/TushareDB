#!/bin/bash
# 快捷查看常用股票
# 无需记忆股票代码，使用简单名称即可

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 常用股票映射
declare -A STOCKS=(
    # 银行
    ["平安银行"]="000001.SZ"
    ["万科A"]="000002.SZ"
    ["中国平安"]="601318.SH"
    ["招商银行"]="600036.SH"
    ["浦发银行"]="600000.SH"
    ["工商银行"]="601398.SH"
    ["建设银行"]="601939.SH"

    # 科技
    ["贵州茅台"]="600519.SH"
    ["五粮液"]="000858.SZ"
    ["中国石油"]="601857.SH"
    ["中国石化"]="600028.SH"

    # 指数代码示例
    ["沪深300"]="000300.SH"
    ["上证指数"]="000001.SH"
)

# 显示帮助
show_help() {
    echo "用法:"
    echo "  $0 <股票名称或代码>"
    echo ""
    echo "常用股票快捷名称:"
    for name in "${!STOCKS[@]}"; do
        echo "  $name → ${STOCKS[$name]}"
    done | sort
    echo ""
    echo "也可以直接使用股票代码:"
    echo "  $0 000001.SZ"
    echo "  $0 600000.SH"
}

# 主逻辑
if [ -z "$1" ]; then
    show_help
    exit 1
fi

INPUT="$1"

# 检查是否是快捷名称
if [ -n "${STOCKS[$INPUT]}" ]; then
    CODE="${STOCKS[$INPUT]}"
    echo "查看: $INPUT ($CODE)"
else
    # 否则认为输入的就是股票代码
    CODE="$INPUT"
fi

# 调用主脚本
exec "$SCRIPT_DIR/quick_view.sh" "$CODE"
