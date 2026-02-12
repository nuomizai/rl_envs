#!/bin/bash
set -e

# 获取脚本所在绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "\033[34m部署命令行工具...\033[0m"
mkdir -p ~/.local/bin
cp "$SCRIPT_DIR/xhumanoid-xtele" ~/.local/bin/
chmod +x ~/.local/bin/xhumanoid-xtele

echo -e "\n\033[32m安装成功！\033[0m"
