#!/bin/bash
set -e

# 获取脚本所在绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
cd scripts
bash ./install_cmd_tool.sh

cd "$PROJECT_ROOT"
cd scripts
bash ./install_pip_pkg.sh

cd "$PROJECT_ROOT"
cd scripts
bash ./install_config.sh ur_single
