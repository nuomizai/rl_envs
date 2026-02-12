#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查参数是否存在
if [ $# -eq 0 ]; then
    echo -e "\033[31m错误：请提供配置文件名参数（如 'ur_single'）\033[0m"
    echo -e "\033[33m可用的配置文件：\033[0m"
    # 自动列出 examples 目录下所有 .toml 文件
    for file in "$PROJECT_ROOT"/examples/*.toml; do
        filename=$(basename "$file" .toml)
        echo -e "  \033[36m$filename\033[0m"
    done
    exit 1
fi

CONFIG_NAME="$1"

# 源文件路径
SOURCE_FILE="$PROJECT_ROOT/examples/${CONFIG_NAME}.toml"

# 目标目录路径
TARGET_DIR="$HOME/.config/xhumanoid/xtele"
TARGET_FILE="$TARGET_DIR/default.toml"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "\033[31m错误：配置文件 '${CONFIG_NAME}' 不存在\033[0m"
    echo -e "\033[33m可用的配置文件：\033[0m"

    # 获取并列出所有可用的配置文件
    mapfile -t config_files < <(find "$PROJECT_ROOT/examples" -maxdepth 1 -name '*.toml' -exec basename {} .toml \;)

    # 突出显示推荐的配置文件
    for config in "${config_files[@]}"; do
        if [[ "$config" == "ur_single" ]]; then
            echo -e "  \033[1;32m$config\033[0m (推荐)"
        else
            echo -e "  \033[36m$config\033[0m"
        fi
    done

    exit 1
fi

# 执行复制操作
echo -e "\033[34m复制配置文件: ${CONFIG_NAME}.toml -> default.toml\033[0m"
mkdir -p "$TARGET_DIR"
cp -f "$SOURCE_FILE" "$TARGET_FILE"

# 检查复制结果
if [ $? -eq 0 ]; then
    echo -e "\033[32m配置文件已更新: $TARGET_FILE\033[0m"
else
    echo -e "\033[31m错误：复制文件失败\033[0m"
    exit 1
fi
