#!/bin/bash
# 双击运行：首次自动安装依赖（需联网一次），之后完全本地运行。
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "首次运行，正在初始化（约 1-3 分钟，需要联网下载依赖）..."
    python3 -m venv .venv || { echo "创建虚拟环境失败，请确认已安装 Xcode 命令行工具：xcode-select --install"; read -p "按回车退出"; exit 1; }
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q || { echo "依赖安装失败，请检查网络后重试"; read -p "按回车退出"; exit 1; }
    echo "初始化完成。"
fi

exec .venv/bin/python -m ptt.gui
