#!/usr/bin/env python3
"""rostopic echo -p /aft_mapped_to_init 记录的 CSV 转 evo 可读的 kitti 格式(3x4 矩阵每行)
用法: python3 csv2kitti.py aft_05.csv aft_05.kitti
LeGO-LOAM 的 camera_init 系与 KITTI 真值同为相机约定(x右/y下/z前), 四元数直接转 R;
残留的 velodyne->cam0 固定外参由 evo_ape 的 -a (umeyama 对齐) 吸收。
"""
import csv
import sys
import numpy as np


def quat2R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)]])


rows = []
with open(sys.argv[1]) as f:
    reader = csv.DictReader(line for line in f if not line.startswith('%'))
    for r in reader:
        x, y, z = (float(r['field.pose.pose.position.x']),
                   float(r['field.pose.pose.position.y']),
                   float(r['field.pose.pose.position.z']))
        qx, qy, qz, qw = (float(r['field.pose.pose.orientation.x']),
                          float(r['field.pose.pose.orientation.y']),
                          float(r['field.pose.pose.orientation.z']),
                          float(r['field.pose.pose.orientation.w']))
        T = np.hstack([quat2R(qx, qy, qz, qw), np.array([[x], [y], [z]])])
        rows.append(T)

print(f'转换位姿数: {len(rows)}')
with open(sys.argv[2], 'w') as f:
    for T in rows:
        f.write(' '.join(f'{v:.9e}' for v in T.flatten()) + '\n')
