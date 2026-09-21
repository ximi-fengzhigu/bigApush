#!/bin/bash
# bigApush 收盘选股脚本
# 运行环境: Linux LoongArch64
# 时间: 工作日 CST 07:10

WORKDIR="/home/thtf/projects/bigApush"
PYTHON="/usr/local/bin/python3.10"
LOGFILE="$WORKDIR/logs/cron_$(date +%Y%m%d_%H%M%S).log"

cd "$WORKDIR" || exit 1

# 创建日志目录
mkdir -p logs

# 运行选股
echo "=== $(date '+%Y-%m-%d %H:%M:%S') 开始运行选股 ===" >> "$LOGFILE"
$PYTHON main.py run >> "$LOGFILE" 2>&1
EXIT_CODE=$?

echo "=== 退出码: $EXIT_CODE ===" >> "$LOGFILE"
exit $EXIT_CODE