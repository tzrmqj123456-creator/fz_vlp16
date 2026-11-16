#!/bin/bash
# RPE 评估(轨迹层): 平移/旋转漂移, 100m 与 800m 窗口, 归档到结果目录
# 用法: bash ~/fz_vlp16/tools/eval_rpe.sh [序列号, 默认05] [结果目录名, 默认 run_月日_时分]
# 前提: 刚跑完播放脚本(生成 /tmp/aft_<seq>.csv), 建议在 run_eval.sh 之后紧接着执行
set -e
SEQ="${1:-05}"
RUN="${2:-run_$(date +%m%d_%H%M)}"
ROOT=/home/trq/fz_vlp16
cd "$ROOT"

[ -f /tmp/aft_${SEQ}.csv ] || { echo "错误: 未找到 /tmp/aft_${SEQ}.csv (请先跑 kitti_play.sh / kitti_play_dyn.sh)"; exit 1; }
python3 tools/csv2tum.py /tmp/aft_${SEQ}.csv /tmp/aft_${SEQ}_fixed.tum
[ -f /tmp/gt_${SEQ}.tum ] || python3 tools/kitti2tum_gt.py data_odometry_poses/dataset/poses/${SEQ}.txt /tmp/gt_${SEQ}.tum

OUT="$ROOT/results/$RUN/evo_axes/rpe"
mkdir -p "$OUT"
cp /tmp/aft_${SEQ}_fixed.tum "$OUT/est.tum"
cp /tmp/gt_${SEQ}.tum "$OUT/gt.tum"
echo "tum 已归档到 $OUT/"

echo "===== RPE 平移漂移 (100m 窗口) ====="
evo_rpe tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r trans_part --delta 100 --delta_unit m -p \
  --save_plot "$OUT/rpe_trans_100m.png" --no_warnings 2>&1 | tee "$OUT/rpe_trans_100m.txt"
echo "===== RPE 旋转漂移 (100m 窗口) ====="
evo_rpe tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r rot_part --delta 100 --delta_unit m -p \
  --save_plot "$OUT/rpe_rot_100m.png" --no_warnings 2>&1 | tee "$OUT/rpe_rot_100m.txt"
echo "===== RPE 平移漂移 (800m 窗口, KITTI 论文常用) ====="
evo_rpe tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r trans_part --delta 800 --delta_unit m -p \
  --save_plot "$OUT/rpe_trans_800m.png" --no_warnings 2>&1 | tee "$OUT/rpe_trans_800m.txt"
echo "===== RPE 旋转漂移 (800m 窗口) ====="
evo_rpe tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r rot_part --delta 800 --delta_unit m -p \
  --save_plot "$OUT/rpe_rot_800m.png" --no_warnings 2>&1 | tee "$OUT/rpe_rot_800m.txt"

echo "===== 完成: $OUT/ ====="
