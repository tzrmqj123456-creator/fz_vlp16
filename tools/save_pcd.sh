#!/bin/bash
# 归档运行后的 pcd 文件到 fz_vlp16/results/<时间戳>/pcd/
# 用法: 运行结束后(Ctrl+C 关闭 launch, mapOptmization 退出时会自动保存 pcd 到 /tmp), 再执行本脚本
# 说明: cornerMap.pcd + surfaceMap.pcd = 完整全局地图(角点+面点), trajectory.pcd = 关键帧轨迹
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/trq/fz_vlp16/results/$TS/pcd
mkdir -p "$OUT"
N=0
for f in finalCloud.pcd cornerMap.pcd surfaceMap.pcd trajectory.pcd; do
    if [ -f /tmp/$f ]; then
        cp /tmp/$f "$OUT/"
        echo "已归档: $f ($(du -h /tmp/$f | cut -f1))"
        N=$((N+1))
    fi
done
[ $N -eq 0 ] && echo "未找到 pcd 文件——请先 Ctrl+C 结束 launch, 等 mapOptmization 退出保存完成后再执行本脚本"
echo "输出目录: $OUT"
