#!/bin/bash
set -e

# 获取脚本所在绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "\033[34m卸载旧版xtele...\033[0m"
pip uninstall -y xtele || true

echo -e "\033[34m安装当前版本xtele...\033[0m"
cd "$PROJECT_ROOT"
cd ..
pip install -e .

echo -e "\n\033[32m安装成功！\033[0m"
