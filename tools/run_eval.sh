#!/bin/bash
# 一键评估脚本: 位姿转换 + 建目录 + 官方 evo 评估(3D + xz 地平面) + pcd 归档
# 用法: bash ~/fz_vlp16/tools/run_eval.sh [序列号, 默认 05] [结果目录名, 默认 run_月日_时分]
# 前提: 已用 kitti_play.sh 跑完并记录了 /tmp/aft_<seq>.csv
set -e
SEQ="${1:-05}"
RUN="${2:-run_$(date +%m%d_%H%M)}"
ROOT=/home/trq/fz_vlp16
cd "$ROOT"

echo "===== 1/4 位姿转换 ====="
[ -f /tmp/aft_${SEQ}.csv ] || { echo "错误: 未找到 /tmp/aft_${SEQ}.csv (请先运行 kitti_play.sh)"; exit 1; }
python3 tools/csv2tum.py /tmp/aft_${SEQ}.csv /tmp/aft_${SEQ}_fixed.tum
python3 tools/kitti2tum_gt.py data_odometry_poses/dataset/poses/${SEQ}.txt /tmp/gt_${SEQ}.tum

echo "===== 2/4 建立结果目录 ====="
OUT="$ROOT/results/$RUN"
mkdir -p "$OUT/evo_axes/xyz" "$OUT/evo_axes/xz"
echo "$OUT"

echo "===== 3/4 官方 evo 评估 ====="
echo "--- [XYZ 三轴] ---"
evo_ape tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r full --t_max_diff 0.03 \
  --save_plot "$OUT/evo_axes/xyz/plot.png" --save_results "$OUT/evo_axes/xyz/xyz.zip" \
  --no_warnings 2>&1 | tee "$OUT/evo_axes/xyz/stats.txt"
echo "--- [xz 地平面投影] ---"
evo_ape tum /tmp/gt_${SEQ}.tum /tmp/aft_${SEQ}_fixed.tum -a -r full --t_max_diff 0.03 \
  --project_to_plane xz --plot_mode xz \
  --save_plot "$OUT/evo_axes/xz/plot.png" --save_results "$OUT/evo_axes/xz/xz.zip" \
  --no_warnings 2>&1 | tee "$OUT/evo_axes/xz/stats.txt"

echo "===== 4/4 pcd 归档 ====="
if ls results/pcd/*.pcd >/dev/null 2>&1; then
    mkdir -p "$OUT/pcd"
    cp results/pcd/*.pcd "$OUT/pcd/"
    echo "已归档: $(ls $OUT/pcd | tr '\n' ' ')"
else
    echo "无 pcd(节点退出时未保存——下次 Ctrl+C 关闭 launch 后等几秒)"
fi

# 汇总报告
{ echo "KITTI seq$SEQ LeGO-LOAM 评估汇总(官方 evo_ape CLI 输出原文)"; echo; \
  echo "### XYZ 三轴"; cat "$OUT/evo_axes/xyz/stats.txt"; echo; \
  echo "### xz 地平面投影"; cat "$OUT/evo_axes/xz/stats.txt"; } > "$OUT/eval_report.txt"
echo "===== 完成: $OUT ====="
