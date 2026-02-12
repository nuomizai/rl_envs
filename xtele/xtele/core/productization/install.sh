#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 安装xtele.wkl包
echo -e "\033[34m安装xtele.wkl包...\033[0m"
pip install --user "$SCRIPT_DIR/xtele.wkl" || {
    echo -e "\033[31m安装失败，请检查包是否存在\033[0m"
    exit 1
}

# 设置环境变量
echo -e "\033[34m配置环境变量...\033[0m"
TARGET_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA083239-if00"
ENV_FILE="$HOME/.bashrc"

# 检查是否已存在配置
if grep -q "DYNAMIXEL_PORT=" "$ENV_FILE"; then
    # 更新现有配置
    sed -i "s|export DYNAMIXEL_PORT=.*|export DYNAMIXEL_PORT=\"$TARGET_PORT\"|" "$ENV_FILE"
else
    # 追加新配置
    echo "export DYNAMIXEL_PORT=\"$TARGET_PORT\"" >> "$ENV_FILE"
fi

# 部署命令行工具
echo -e "\033[34m部署命令行工具...\033[0m"
mkdir -p ~/.local/bin
cp "$SCRIPT_DIR/xhumanoid-xtele" ~/.local/bin/
chmod +x ~/.local/bin/xhumanoid-xtele

echo -e "\033[32m安装成功！\033[0m"
echo -e "请执行以下命令使配置生效："
echo -e "\033[33msource ~/.bashrc\033[0m"
echo -e "或重新打开终端"
