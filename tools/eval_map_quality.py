#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3层指标 ②: 建图质量 (残影率 + 静态结构精度)
输入: 一轮运行的归档结果目录 (含 pcd/cornerMap.pcd, pcd/surfaceMap.pcd)
      + 该轮的 est.tum (由 eval_rpe.sh 归档) 或 /tmp/aft_05_fixed.tum
输出:
  Ghost Rate (残影率)  : 地图中落在"动态物经过区域"(SemanticKITTI moving 点集) 0.5m 内的点数占比
  Static NN (静态精度) : 地图点到静态 GT 点集的最近邻距离 mean/median (越小越好)
  Static Coverage      : 地图点在静态 GT 1.0m 内的占比 (地图完整性)
用法: python3 tools/eval_map_quality.py <结果目录名>
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/trq/fz_vlp16'
RUN = sys.argv[1] if len(sys.argv) > 1 else None
if RUN is None:
    print('用法: python3 tools/eval_map_quality.py <结果目录名, 如 run_1115_1444>')
    sys.exit(1)

VEL_DIR = f'{ROOT}/dataset/sequences/05/velodyne'
LBL_DIR = f'{ROOT}/semkitti_labels/dataset/sequences/05/labels'
CALIB = f'{ROOT}/data_odometry_calib/dataset/sequences/05/calib.txt'
MOVING_IDS = set(range(251, 258))
STATIC_EXCLUDE = MOVING_IDS | {0, 1}
R_MOV, R_STA = 0.5, 1.0          # 残影/覆盖判定半径
F_STEP, P_STEP = 5, 5            # 每5帧、每5点抽样


def load_tum(path):
    ts, xyz = [], []
    for line in open(path):
        p = line.split()
        if len(p) >= 8:
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz)


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    Pp, Qq = P - mp, Q - mq
    cov = Pp.T @ Qq / len(P)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / ((Pp ** 2).sum() / len(P))
    return s, R, mp, mq


def read_pcd(path):
    with open(path, 'rb') as f:
        d = f.read()
    h = d.find(b'DATA binary')
    h = d.find(b'\n', h) + 1
    n = (len(d) - h) // 16
    arr = np.frombuffer(d[h:], np.float32, count=n * 4).reshape(n, 4)
    return arr[:, :3]


def load_calib():
    data = {}
    for line in open(CALIB):
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    return np.vstack([data['Tr'].reshape(3, 4), [0, 0, 0, 1]])  # velo -> cam0 (4x4)


def main():
    out = f'{ROOT}/results/{RUN}'
    est_path = f'{out}/evo_axes/rpe/est.tum'
    if not os.path.exists(est_path):
        est_path = '/tmp/aft_05_fixed.tum'
    gt_path = f'{out}/evo_axes/rpe/gt.tum'
    if not os.path.exists(gt_path):
        gt_path = '/tmp/gt_05.tum'
        if not os.path.exists(gt_path):
            os.system(f'python3 {ROOT}/tools/kitti2tum_gt.py {ROOT}/data_odometry_poses/dataset/poses/05.txt {gt_path}')
    est_ts, est_p = load_tum(est_path)
    gt_ts, gt_p = load_tum(gt_path)
    print(f'est {len(est_p)} 行, gt {len(gt_p)} 行')

    # 地图 -> GT 世界: pcd(camera_init) 绕z转180 -> TUM系 -> umeyama(est->gt)
    Tr_v2c = load_calib()
    map_pts = np.vstack([read_pcd(f'{out}/pcd/cornerMap.pcd'), read_pcd(f'{out}/pcd/surfaceMap.pcd')])
    map_pts = map_pts[:, [2, 0, 1]]          # (ty,tz,tx) -> (tx,ty,tz)
    map_pts[:, :2] = -map_pts[:, :2]          # camera_init -> KITTI 相机系(绕z 180)
    s, R, mp, mq = umeyama(est_p, gt_p)
    map_world = s * (R @ map_pts.T).T + (mq - s * R @ mp)
    print(f'地图点数 {len(map_world)} (umeyama: s={s:.4f})')

    # 累积 moving/static GT 点集 (帧内 velo -> cam0 -> GT 位姿 -> 世界)
    files = sorted(f for f in os.listdir(LBL_DIR) if f.endswith('.label'))
    mov_list, sta_list = [], []
    t0 = time.time()
    for k in range(0, len(files), F_STEP):
        name = files[k][:-6]
        bin_path = f'{VEL_DIR}/{name}.bin'
        if not os.path.exists(bin_path):
            continue
        pts = np.fromfile(bin_path, np.float32).reshape(-1, 4)[::P_STEP]
        lbl = np.fromfile(f'{LBL_DIR}/{files[k]}', np.uint32)[::P_STEP]
        sem = lbl & 0xFFFF
        is_mov = np.array([x in MOVING_IDS for x in sem])
        is_sta = np.array([x not in STATIC_EXCLUDE for x in sem])
        if not is_mov.any() and not is_sta.any():
            continue
        # velo -> cam0
        xyz1 = np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T
        cam = (Tr_v2c @ xyz1)[:3].T
        # cam0 -> 世界 (GT 位姿)
        t = 0.1 * k
        gi = int(np.argmin(np.abs(gt_ts - t)))
        from scipy.spatial.transform import Rotation as Rrot
        # gt tum 只存了平移; 位姿旋转从 KITTI poses 读
        pose_path = f'{ROOT}/data_odometry_poses/dataset/poses/05.txt'
        line = open(pose_path).readlines()[gi]
        Tgt = np.array([float(x) for x in line.split()]).reshape(3, 4)
        cam_world = (Tgt[:3, :3] @ cam.T).T + Tgt[:3, 3]
        if is_mov.any():
            mov_list.append(cam_world[is_mov])
        if is_sta.any():
            sta_list.append(cam_world[is_sta])
        if k % 500 == 0:
            print(f'  累积 GT 点集 {k}/{len(files)} ({time.time()-t0:.0f}s)')
    mov_pts = np.vstack(mov_list) if mov_list else np.zeros((0, 3))
    sta_pts = np.vstack(sta_list) if sta_list else np.zeros((0, 3))
    print(f'moving GT 点 {len(mov_pts)}, 静态 GT 点 {len(sta_pts)}')

    # 残影率: 地图点 0.5m 内有 moving GT 点的占比
    if len(mov_pts):
        tree_m = cKDTree(mov_pts)
        d_m, _ = tree_m.query(map_world, k=1, workers=-1)
        ghost = float((d_m < R_MOV).mean())
    else:
        ghost = 0.0
    # 静态精度/覆盖: 地图点 -> 静态 GT 最近邻
    tree_s = cKDTree(sta_pts)
    d_s, _ = tree_s.query(map_world, k=1, workers=-1)
    coverage = float((d_s < R_STA).mean())

    res_path = f'{out}/map_quality.txt'
    with open(res_path, 'w') as f:
        f.write(f'Ghost Rate (残影率, <{R_MOV}m 判定): {ghost:.6f}\n')
        f.write(f'Static NN mean (静态精度): {d_s.mean():.3f} m\n')
        f.write(f'Static NN median: {np.median(d_s):.3f} m\n')
        f.write(f'Static Coverage (<{R_STA}m): {coverage:.4f}\n')
    print('===== 建图质量 =====')
    print(f'Ghost Rate (残影率, <{R_MOV}m): {ghost:.6f}  (越低越好)')
    print(f'Static NN mean (静态精度): {d_s.mean():.3f} m  (越低越好)')
    print(f'Static NN median: {np.median(d_s):.3f} m')
    print(f'Static Coverage (<{R_STA}m): {coverage:.4f}  (越高越好, 地图完整性)')
    print(f'结果已存: {res_path}')


if __name__ == '__main__':
    main()
