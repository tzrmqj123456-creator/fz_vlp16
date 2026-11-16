#!/bin/bash
# 第2层指标: 回环检测质量统计 + 因子 GT 真伪验证
# 用法: bash ~/fz_vlp16/tools/eval_loop.sh <launch日志> [结果目录名]
# 例: bash ~/fz_vlp16/tools/eval_loop.sh /tmp/dyn_run_launch5.log run_1115_1444
# 输出: 检测/添加/拒绝/跳过计数, 每个添加因子的 GT 距离真伪表(定性, 行号偏移±14s)
set -e
LOG="${1:?用法: eval_loop.sh <launch日志> [结果目录名]}"
RUN="${2:-run_$(date +%m%d_%H%M)}"
ROOT=/home/trq/fz_vlp16
cd "$ROOT"

OUT="$ROOT/results/$RUN"
mkdir -p "$OUT"

echo "===== 回环统计 ($LOG) =====" | tee "$OUT/loop_stats.txt"
for pat in "loop factor is added" "Reject this loop" "Skip this loop" "Loop found" "Not loop"; do
    n=$(grep -c "$pat" "$LOG" || true)
    echo "$pat : $n" | tee -a "$OUT/loop_stats.txt"
done
echo "--- 添加的因子列表 ---" | tee -a "$OUT/loop_stats.txt"
grep "loop factor is added" "$LOG" | tee -a "$OUT/loop_stats.txt"

echo "" | tee -a "$OUT/loop_stats.txt"
echo "===== 因子 GT 真伪验证 =====" | tee -a "$OUT/loop_stats.txt"
[ -f /tmp/gt_05.tum ] || python3 tools/kitti2tum_gt.py data_odometry_poses/dataset/poses/05.txt /tmp/gt_05.tum
[ -f /tmp/aft_05_fixed.tum ] || { echo "注意: 无 /tmp/aft_05_fixed.tum, 只做统计不做真伪验证" | tee -a "$OUT/loop_stats.txt"; exit 0; }

grep -oE "Current \[ *[0-9]+ \] and (SC|RS) nearest \[ *[0-9]+ \]" "$LOG" | \
  grep -oE "[0-9]+" | paste - - - - | \
while read c h _ _; do
    python3 - "$c" "$h" << 'EOF' | tee -a "$OUT/loop_stats.txt"
import sys
import numpy as np
c, h = int(sys.argv[1]), int(sys.argv[2])
def load(path):
    ts, xyz = [], []
    for line in open(path):
        p = line.split()
        if len(p) >= 8:
            ts.append(float(p[0])); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz)
gt_ts, gt_xyz = load('/tmp/gt_05.tum')
est_ts, _ = load('/tmp/aft_05_fixed.tum')
if c >= len(est_ts) or h >= len(est_ts):
    print(f'  (超出范围: curr {c}, hist {h})'); sys.exit(0)
p1 = gt_xyz[np.argmin(np.abs(gt_ts - est_ts[c]))]
p2 = gt_xyz[np.argmin(np.abs(gt_ts - est_ts[h]))]
d = np.linalg.norm(p1 - p2)
verdict = '真' if d < 15 else ('近' if d < 40 else '假')
print(f'  因子 ({c:>4}, {h:>4}): GT 距离 {d:6.1f} m -> {verdict} (粗判, kf↔行号偏移±14s)')
EOF
done

echo "===== 完成: $OUT/loop_stats.txt ====="
