#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3层指标 ①: 动态滤除质量离线评估 (不需要跑 ROS 管线)
对抽样帧: 图像 -> YOLOv8-seg 掩膜(与 dyn_filter 相同参数) -> 点云投影
          -> 深度连续前景簇 -> 多帧实例跟踪(世界系速度) -> 确认 moving 删前景簇
          (与 dyn_filter 相同逻辑, 里程计来自 /tmp/aft_05.csv 插值)
对照 SemanticKITTI moving 标注计算:
  Dynamic Recall / Precision / F1 / Static Retention + 推理耗时
用法: python3 tools/eval_filter_offline.py [结果目录名, 默认 run_月日_时分]
"""
import csv
import os
import sys
import time
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = '/home/trq/fz_vlp16'
RUN = sys.argv[1] if len(sys.argv) > 1 else 'run_' + time.strftime('%m%d_%H%M')
IMG_DIR = f'{ROOT}/05_image/image_2_jpg'
VEL_DIR = f'{ROOT}/dataset/sequences/05/velodyne'
LBL_DIR = f'{ROOT}/semkitti_labels/dataset/sequences/05/labels'
CALIB = f'{ROOT}/data_odometry_calib/dataset/sequences/05/calib.txt'
MODEL = f'{ROOT}/lego_loam_dyn/models/yolov8s-seg.pt'
ODY_CSV = '/tmp/aft_05.csv'
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]
MOVING_IDS = set(range(251, 258))
STATIC_EXCLUDE = MOVING_IDS | {0, 1}
CONF, DILATE = 0.35, 3
IMG_W, IMG_H = 1242, 375
STEP = 2          # 每 2 帧评估 1 帧(0.2s 间隔, 与在线跟踪节奏一致)
CLUSTER_EPS, MIN_CLUSTER_PTS = 0.8, 10
NN_THRESH = 0.4

sys.path.insert(0, f'{ROOT}/lego_loam_dyn/scripts')
from dyn_track import quat_rotate, nlerp, foreground_cluster, StaticMapResidual


def load_calib():
    data = {}
    for line in open(CALIB):
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = data['P2'].reshape(3, 4)
    Tr = np.vstack([data['Tr'].reshape(3, 4), [0, 0, 0, 1]])
    return P2 @ Tr


def load_odom():
    """aft csv -> (时间戳, 四元数, 平移) 数组"""
    if not os.path.exists(ODY_CSV):
        return None, None, None
    ts, qs, txyz = [], [], []
    with open(ODY_CSV) as f:
        for r in csv.DictReader(f):
            ts.append(float(r['field.header.stamp']) / 1e9)
            qs.append([float(r['field.pose.pose.orientation.x']),
                       float(r['field.pose.pose.orientation.y']),
                       float(r['field.pose.pose.orientation.z']),
                       float(r['field.pose.pose.orientation.w'])])
            txyz.append([float(r['field.pose.pose.position.x']),
                         float(r['field.pose.pose.position.y']),
                         float(r['field.pose.pose.position.z'])])
    return np.array(ts), np.array(qs), np.array(txyz)


def odom_at(ts, qs, txyz, t):
    """插值到 t 的位姿"""
    i = int(np.searchsorted(ts, t, side='right') - 1)
    i = min(max(i, 0), len(ts) - 1)
    if i == len(ts) - 1 or t <= ts[0]:
        return qs[i], txyz[i]
    a = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9)
    return nlerp(qs[i], qs[i + 1], a), (1 - a) * txyz[i] + a * txyz[i + 1]


def main():
    P = load_calib()
    model = YOLO(MODEL)
    odom_ts, odom_q, odom_t = load_odom()
    track_on = odom_ts is not None
    if not track_on:
        print('警告: 无 /tmp/aft_05.csv, 多帧跟踪关闭(纯外观法结果)')
    residual = StaticMapResidual(max_scans=30, voxel=1.0, nn_thresh=NN_THRESH)

    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
    tp = fp = fn = 0
    static_kept = static_total = 0
    n_saved = n_removed = 0  # 实例级: 保留(parked)/剔除(moving确认)
    times = []
    rows = []
    for i in range(0, len(files), STEP):
        name = files[i][:-4]
        img = cv2.imread(f'{IMG_DIR}/{files[i]}')
        if (img.shape[1], img.shape[0]) != (IMG_W, IMG_H):
            img = cv2.resize(img, (IMG_W, IMG_H))
        pts = np.fromfile(f'{VEL_DIR}/{name}.bin', np.float32).reshape(-1, 4)
        lbl = np.fromfile(f'{LBL_DIR}/{name}.label', np.uint32)
        if len(pts) != len(lbl):
            continue
        sem = lbl & 0xFFFF

        # 掩膜 (与 dyn_filter 相同)
        t0 = time.time()
        res = model(img, conf=CONF, classes=DYNAMIC_CLASSES, verbose=False)[0]
        times.append((time.time() - t0) * 1000)
        mask = np.zeros((IMG_H, IMG_W), bool)
        masks = None
        if res.masks is not None and len(res.masks.data) > 0:
            masks = res.masks.data.cpu().numpy().astype(bool)
            masks = np.stack([cv2.resize(m.astype(np.uint8), (IMG_W, IMG_H),
                                         interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
            masks = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks])
            mask = masks.any(0)

        # 投影
        N = len(pts)
        xyz1 = np.hstack([pts[:, :3], np.ones((N, 1))]).T
        X = P @ xyz1
        Z = X[2]
        valid = Z > 0
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        Zc = np.where(Z > 0, Z, np.inf)

        # 实例级判定 (与 dyn_filter 相同: 前景簇 + 静态地图残差)
        drop = np.zeros(N, bool)
        t_frame = 0.1 * i
        if masks is not None and track_on:
            q, t = odom_at(odom_ts, odom_q, odom_t, t_frame)
            world = quat_rotate(q, pts[:, :3]) + t
            valid_idx = np.where(in_img)[0]
            rr, cc = v[valid_idx], u[valid_idx]
            for m in masks:
                idx = valid_idx[m[rr, cc]]
                if len(idx) == 0:
                    continue
                clu = foreground_cluster(pts[idx, :3], Zc[idx], eps=CLUSTER_EPS)
                if len(clu) < MIN_CLUSTER_PTS:
                    continue  # 簇太小: 保留(静止先验)
                if residual.is_moving(world[idx[clu]]):
                    drop[idx[clu]] = True
                    n_removed += 1
                else:
                    n_saved += 1
            residual.add_scan(world)
            residual.maybe_rebuild(t_frame)
        else:
            drop[in_img] = mask[v[in_img], u[in_img]]

        gt_mov = np.array([s in MOVING_IDS for s in sem])
        gt_sta = np.array([s not in STATIC_EXCLUDE for s in sem])
        tp += int((drop & gt_mov).sum())
        fp += int((drop & ~gt_mov).sum())
        fn += int((~drop & gt_mov).sum())
        static_kept += int((~drop & gt_sta).sum())
        static_total += int(gt_sta.sum())
        rows.append((name, int(gt_mov.sum()), int((drop & gt_mov).sum())))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    retention = static_kept / max(static_total, 1)

    out_dir = f'{ROOT}/results/{RUN}'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'{out_dir}/filter_metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('frame,gt_moving,removed_moving\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]}\n')
    print(f'===== 动态滤除质量 (评估 {len(rows)} 帧, {os.path.basename(MODEL)}, 多帧跟踪: {"开" if track_on else "关(纯外观)"}) =====')
    print(f'Dynamic Recall   : {recall:.3f}  (tp={tp}, fn={fn})')
    print(f'Dynamic Precision: {precision:.3f}  (fp={fp})')
    print(f'F1               : {f1:.3f}')
    print(f'Static Retention : {retention:.4f}  (保留 {static_kept}/{static_total})')
    print(f'实例判定: 确认moving剔除 {n_removed} 个, 保留(含parked/未确认) {n_saved} 个')
    print(f'YOLO 推理耗时    : mean {np.mean(times):.1f} ms/帧')
    print(f'结果已存: {csv_path}')


if __name__ == '__main__':
    main()
