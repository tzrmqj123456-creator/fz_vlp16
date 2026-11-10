#!/bin/bash
# 切换到原版 LeGO-LOAM 基线 (分支 main = 提交 740b139, RS-only 回环, 无 SC) 并编译
# 用法: bash ~/fz_vlp16/tools/build_original.sh
set -e
cd /home/trq/fz_vlp16

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo ">>> 错误: 工作区有未提交改动, 先 git commit 或 git stash 再切换" >&2
    exit 1
fi

git checkout main
source /opt/ros/noetic/setup.bash
source /home/trq/fz_vlp16/catkin_ws/devel/setup.bash
cd /home/trq/fz_vlp16/catkin_ws && catkin_make 2>&1 | tail -3
echo ">>> 编译完成: 原版 LeGO-LOAM (RS-only), 当前分支 main"
