#!/usr/bin/env python3
"""KITTI 真值 poses/05.txt -> TUM 格式(时间戳按 10Hz 合成: t=i*0.1)
用法: python3 kitti2tum_gt.py poses/05.txt gt_05.tum
KITTI GT 是相机系约定(x右/y下/z前), 与 LeGO-LOAM 的 camera_init 系同约定, 无需换轴;
残留的 velodyne->cam0 固定外参由 evo_ape -a 对齐吸收。
"""
import sys
import numpy as np

rows = []
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        T = np.array([float(v) for v in line.split()]).reshape(3, 4)
        R, t = T[:, :3], T[:, 3]
        qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        if qw < 1e-12:  # 退化保护
            qx = qy = qz = qw = 0.0
        else:
            qx = (R[2, 1] - R[1, 2]) / (4 * qw)
            qy = (R[0, 2] - R[2, 0]) / (4 * qw)
            qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        rows.append((i * 0.1, t[0], t[1], t[2], qx, qy, qz, qw))

with open(sys.argv[2], 'w') as f:
    for r in rows:
        f.write(f'{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} '
                f'{r[4]:.9f} {r[5]:.9f} {r[6]:.9f} {r[7]:.9f}\n')
print(f'输出 {len(rows)} 个位姿')
