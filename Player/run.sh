#!/usr/bin/env bash
# 参赛入口：bash run.sh port（判题器标准启动方式，内部调 Demo 同款 main3.py）
set -euo pipefail
cd "$(dirname "$0")"

# 优先用能真正运行的解释器（Windows 下 python3 可能是商店占位 stub）
for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version >/dev/null 2>&1; then
        exec "$candidate" main3.py "$@"
    fi
done
echo "no usable python interpreter found" >&2
exit 1
