#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流动态判定离线验证: Farneback 光流 + 自车运动拟合 + 实例残差, 对照 SemanticKITTI moving 标注
输出: parked/moving 实例的残差光流分布 + 阈值扫描 + ego 速度合理性检查
用法: python3 tools/eval_flow_offline.py [结果目录名, 默认 run_月日_时分]
"""
import os
import sys
import time
import numpy as np
import cv2

ROOT = '/home/trq/fz_vlp16'
RUN = sys.argv[1] if len(sys.argv) > 1 else 'run_' + time.strftime('%m%d_%H%M')
IMG_DIR = f'{ROOT}/05_image/image_2_jpg'
VEL_DIR = f'{ROOT}/dataset/sequences/05/velodyne'
LBL_DIR = f'{ROOT}/semkitti_labels/dataset/sequences/05/labels'
CALIB = f'{ROOT}/data_odometry_calib/dataset/sequences/05/calib.txt'
MODEL = f'{ROOT}/lego_loam_dyn/models/yolov8s-seg.pt'
MOVING_IDS = set(range(251, 258))
STEP = 1   # 逐帧(0.1s间隔): GN 初值按单帧位移设定, 多帧间隔会差整数倍导致拟合失败
CONF, DILATE = 0.35, 3

sys.path.insert(0, f'{ROOT}/lego_loam_dyn/scripts')
from dyn_flow import compute_flow, egomotion_fit, instance_residual, foreground_cluster, FLOW_W, FLOW_H, RaftFlow
from ultralytics import YOLO

BACKEND = sys.argv[2] if len(sys.argv) > 2 else 'raft'   # raft | farneback


def load_calib():
    data = {}
    for line in open(CALIB):
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = data['P2'].reshape(3, 4)
    Tr = np.vstack([data['Tr'].reshape(3, 4), [0, 0, 0, 1]])
    return P2 @ Tr, P2[:3, :3]


def main():
    P, K = load_calib()
    Ks = K.copy()
    Ks[0, 0] *= FLOW_W / 1242.0
    Ks[0, 2] *= FLOW_W / 1242.0
    Ks[1, 1] *= FLOW_H / 375.0
    Ks[1, 2] *= FLOW_H / 375.0
    model = YOLO(MODEL)
    raft = RaftFlow() if BACKEND == 'raft' else None

    res_park, res_mov = [], []
    ego_speeds = []
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
    t_flow = []
    for i in range(1, len(files), STEP):
        if BACKEND == 'raft':
            img1c = cv2.imread(f'{IMG_DIR}/{files[i]}')
            if img1c is None:
                continue
            t0 = time.time()
            flow = raft(img1c)
            t_flow.append((time.time() - t0) * 1000)
            if flow is None:
                continue
        else:
            img0 = cv2.imread(f'{IMG_DIR}/{files[i-1]}', cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(f'{IMG_DIR}/{files[i]}', cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                continue
            t0 = time.time()
            flow = compute_flow(img0, img1)      # (FLOW_H, FLOW_W, 2) 降采样空间
            t_flow.append((time.time() - t0) * 1000)
        name = files[i][:-4]
        pts = np.fromfile(f'{VEL_DIR}/{name}.bin', np.float32).reshape(-1, 4)
        lbl = np.fromfile(f'{LBL_DIR}/{name}.label', np.uint32)
        if len(pts) != len(lbl):
            continue
        sem = lbl & 0xFFFF

        # 深度点投影(原始分辨率) -> 降采样空间坐标 + 光流采样(最近邻)
        xyz1 = np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T
        X = P @ xyz1
        Z = X[2]
        valid = Z > 0.3
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < 1242) & (v >= 0) & (v < 375)
        pos_in = np.full(len(pts), -1, np.int64)
        pos_in[in_img] = np.arange(in_img.sum())
        Z_in = Z[in_img]
        us = (u[in_img] * (FLOW_W / 1242.0)).astype(np.int64)
        vs = (v[in_img] * (FLOW_H / 375.0)).astype(np.int64)
        us = np.clip(us, 0, FLOW_W - 1)
        vs = np.clip(vs, 0, FLOW_H - 1)
        fu = flow[vs, us, 0]
        fv_ = flow[vs, us, 1]

        # 自车运动拟合(离散迭代精化; 不依赖 odom)
        ret = egomotion_fit(us.astype(float), vs.astype(float), Z_in, flow, Ks)
        if ret is None:
            continue
        sol, R, t, ratio = ret
        ego_speeds.append(float(np.linalg.norm(sol[:3])))

        # 实例残差
        res = model(cv2.imread(f'{IMG_DIR}/{files[i]}'), conf=CONF,
                    classes=[0, 1, 2, 3, 5, 6, 7], verbose=False)[0]
        if res.masks is None or len(res.masks.data) == 0:
            continue
        masks = res.masks.data.cpu().numpy().astype(bool)
        masks = np.stack([cv2.resize(m.astype(np.uint8), (1242, 375),
                                     interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
        masks = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks])
        Zc = np.where(Z > 0, Z, np.inf)
        for m in masks:
            mem = np.zeros(len(pts), bool)
            mem[in_img] = m[v[in_img], u[in_img]]
            full_idx = np.where(mem)[0]
            if len(full_idx) < 10:
                continue
            # 前景簇: 只取物体本体点(排除掩码边缘背景, 否则期望流错误抬高噪声底)
            clu = foreground_cluster(pts[full_idx, :3], Zc[full_idx])
            if len(clu) < 10:
                continue
            fidx = full_idx[clu]
            idx = pos_in[fidx]
            idx = idx[idx >= 0]
            if len(idx) < 10:
                continue
            resid = instance_residual(us[idx].astype(float), vs[idx].astype(float),
                                      Z_in[idx], flow, R, t, Ks)
            frac = (np.array([s in MOVING_IDS for s in sem[fidx]]).sum()) / len(fidx)
            (res_mov if frac > 0.5 else res_park).append(resid)

    res_park, res_mov = np.array(res_park), np.array(res_mov)
    print(f'===== 光流残差分布 (评估 {len(files)//STEP} 帧对, RAFT {FLOW_W}x{FLOW_H}) =====')
    print(f'ego 速度拟合: 中位 {np.median(ego_speeds)*10:.1f} m/s (GT 起始段约 7-8 m/s, 合理性检查)')
    print(f'parked 实例残差: n={len(res_park)}, 中位 {np.median(res_park):.3f} m/帧, 90分位 {np.percentile(res_park,90):.3f} m/帧')
    print(f'moving 实例残差: n={len(res_mov)}, 中位 {np.median(res_mov):.3f} m/帧, 10分位 {np.percentile(res_mov,10):.3f} m/帧')
    print(f'光流计算耗时: mean {np.mean(t_flow):.0f} ms/帧')
    print('阈值扫描 (残差 > th 判 moving, 单位 m/帧):')
    for th in [0.05, 0.1, 0.15, 0.2, 0.3]:
        tp = (res_mov > th).sum(); fp = (res_park > th).sum()
        fn = (res_mov <= th).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f'  th={th:.1f}px: Precision {prec:.3f} | Recall {rec:.3f} | 误删parked {fp}/{len(res_park)}')


if __name__ == '__main__':
    main()
