#!/bin/bash
# 切换到 SC-LeGO-LOAM 基线 (分支 sc-lego-loam: SC 集成 + 双时间门槛 + 0.13/0.5/冷却3s 调参) 并编译
# 用法: bash ~/fz_vlp16/tools/build_sc.sh
set -e
cd /home/trq/fz_vlp16

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo ">>> 错误: 工作区有未提交改动, 先 git commit 或 git stash 再切换" >&2
    exit 1
fi

git checkout sc-lego-loam
source /opt/ros/noetic/setup.bash
source /home/trq/fz_vlp16/catkin_ws/devel/setup.bash
cd /home/trq/fz_vlp16/catkin_ws && catkin_make 2>&1 | tail -3
echo ">>> 编译完成: SC-LeGO-LOAM, 当前分支 sc-lego-loam"
