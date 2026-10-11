#!/usr/bin/env python3
"""aft_05.csv -> TUM 格式(相对时间戳, 起始归零)
用法: python3 csv2tum.py /tmp/aft_05.csv /tmp/aft_05.tum
输出行: t tx ty tz qx qy qz qw (t 为相对首帧的秒数)
注意: rostopic echo -p 的表头第一行本身以 %time 开头(需保留作表头), 时间为纳秒整数
"""
import csv
import sys

rows = []
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append((
            float(r['field.header.stamp']) / 1e9,  # ros 时间(纳秒) -> 秒
            float(r['field.pose.pose.position.x']),
            float(r['field.pose.pose.position.y']),
            float(r['field.pose.pose.position.z']),
            float(r['field.pose.pose.orientation.x']),
            float(r['field.pose.pose.orientation.y']),
            float(r['field.pose.pose.orientation.z']),
            float(r['field.pose.pose.orientation.w'])))

t0 = rows[0][0]
with open(sys.argv[2], 'w') as f:
    for r in rows:
        t = r[0] - t0
        # LeGO-LOAM 的 camera_init 系(x=左,y=上,z=前)与 KITTI 相机系(x=右,y=下,z=前)
        # 相差绕 z 轴 180°: 位置 (x,y,z)->(-x,-y,z), 四元数乘 q_z180=(0,0,1,0)
        x, y, z = -r[1], -r[2], r[3]
        qx, qy, qz, qw = -r[5], r[4], r[7], -r[6]  # p⊗q, p=(0,0,1,0): (-qy, qx, qw, -qz)
        f.write(f'{t:.6f} {x:.6f} {y:.6f} {z:.6f} '
                f'{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n')
print(f'输出 {len(rows)} 个位姿(已绕z旋转180°对齐KITTI相机系), 时间跨度 {rows[-1][0] - t0:.1f}s')
