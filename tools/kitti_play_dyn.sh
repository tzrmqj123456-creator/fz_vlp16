#!/bin/bash
# 动态滤除管线: 同步播放器(图像+点云同时间戳) + 记录 /aft_mapped_to_init 供 evo 评估
# 用法: bash kitti_play_dyn.sh [序列号, 默认 05]
# 前提: 终端1 已 roslaunch lego_loam_dyn dyn_run.launch
set -u
SEQ="${1:-05}"
source /opt/ros/noetic/setup.bash
source /home/trq/fz_vlp16/catkin_ws/devel/setup.bash

mkdir -p /home/trq/fz_vlp16/results/pcd

rostopic echo -p /aft_mapped_to_init > /tmp/aft_${SEQ}.csv 2>&1 &
REC=$!

start=$(date +%s)
stdbuf -oL rosrun lego_loam_dyn kitti_dyn_player.py \
  _img_dir:=/home/trq/fz_vlp16/05_image/image_2 \
  _velodyne_dir:=/home/trq/fz_vlp16/dataset/sequences/${SEQ}/velodyne 2>&1 | tail -3
sleep 8   # 让最后几帧走完管线
kill $REC 2>/dev/null

end=$(date +%s)
echo "RUN_DONE elapsed=$((end-start))s"
echo "=== /aft_mapped_to_init 记录行数 ==="
grep -vc '^%' /tmp/aft_${SEQ}.csv
