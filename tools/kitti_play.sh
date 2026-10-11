#!/bin/bash
# KITTI 序列完整播放(实时) + 记录 /aft_mapped_to_init 供 evo 评估
# 用法: bash kitti_play.sh [序列号, 默认 05]
set -u
SEQ="${1:-05}"
source /opt/ros/noetic/setup.bash
source /home/trq/fz_vlp16/catkin_ws/devel/setup.bash

# 确保 pcd 保存目录存在(节点退出时自动保存地图到该目录)
mkdir -p /home/trq/fz_vlp16/results/pcd

rostopic echo -p /aft_mapped_to_init > /tmp/aft_${SEQ}.csv 2>&1 &
REC=$!

start=$(date +%s)
stdbuf -oL rosrun kitti_player kitti_player _data_path:=/home/trq/fz_vlp16/dataset _sequence:=$SEQ 2>&1 | tail -3
sleep 8   # 让最后几帧走完管线
kill $REC 2>/dev/null

end=$(date +%s)
echo "RUN_DONE elapsed=$((end-start))s"
echo "=== /aft_mapped_to_init 记录行数 ==="
grep -vc '^%' /tmp/aft_${SEQ}.csv
