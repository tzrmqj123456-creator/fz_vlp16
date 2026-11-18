#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3层指标 ①: 动态滤除质量离线评估 (不需要跑 ROS 管线)
对抽样帧: 图像 -> YOLOv8-seg 掩膜(与 dyn_filter 相同参数) -> 点云投影滤除,
对照 SemanticKITTI moving 标注计算:
  Dynamic Recall    = 剔除的真动态点 / GT 动态点
  Dynamic Precision = 剔除的点中真动态占比
  Static Retention  = 保留的静态点 / GT 静态点 (误删度量, 关键)
  F1
另测每帧 YOLO 推理耗时(系统层指标)。
用法: python3 tools/eval_filter_offline.py [结果目录名, 默认 run_月日_时分]
SemanticKITTI moving 语义 id: 251-257 (moving-car/bicyclist/person/motorcyclist/other-vehicle/truck/bus)
"""
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
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]   # COCO: person..truck (与 dyn_filter 一致)
MOVING_IDS = set(range(251, 258))          # SemanticKITTI moving 段
STATIC_EXCLUDE = MOVING_IDS | {0, 1}       # 静态 = 非 moving 且非 unlabeled/outlier
CONF, DILATE = 0.35, 3
IMG_W, IMG_H = 1242, 375
STEP = 10  # 每 10 帧评估 1 帧


def load_calib():
    data = {}
    for line in open(CALIB):
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = data['P2'].reshape(3, 4)
    Tr = np.vstack([data['Tr'].reshape(3, 4), [0, 0, 0, 1]])
    return P2 @ Tr  # 3x4 velodyne -> rectified 左目像素


def main():
    P = load_calib()
    model = YOLO(MODEL)
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
    tot = tp = fp = fn = 0
    static_kept = static_total = 0
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

        # 掩膜 (与 dyn_filter 相同逻辑)
        t0 = time.time()
        res = model(img, conf=CONF, classes=DYNAMIC_CLASSES, verbose=False)[0]
        times.append((time.time() - t0) * 1000)
        mask = np.zeros((IMG_H, IMG_W), bool)
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)
            masks = np.stack([cv2.resize(m.astype(np.uint8), (IMG_W, IMG_H),
                                         interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
            masks = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks])
            mask = masks.any(0)

        # 投影 + 剔除判定
        xyz1 = np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T
        X = P @ xyz1
        Z = X[2]
        valid = Z > 0
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        dropped = np.zeros(len(pts), bool)
        dropped[in_img] = mask[v[in_img], u[in_img]]

        gt_mov = np.array([s in MOVING_IDS for s in sem])
        gt_sta = np.array([s not in STATIC_EXCLUDE for s in sem])
        tp += int((dropped & gt_mov).sum())
        fp += int((dropped & ~gt_mov).sum())
        fn += int((~dropped & gt_mov).sum())
        static_kept += int((~dropped & gt_sta).sum())
        static_total += int(gt_sta.sum())
        rows.append((name, int(gt_mov.sum()), int((dropped & gt_mov).sum())))

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
    print(f'===== 动态滤除质量 (评估 {len(rows)} 帧, {os.path.basename(MODEL)}) =====')
    print(f'Dynamic Recall   : {recall:.3f}  (tp={tp}, fn={fn})')
    print(f'Dynamic Precision: {precision:.3f}  (fp={fp})')
    print(f'F1               : {f1:.3f}')
    print(f'Static Retention : {retention:.4f}  (保留 {static_kept}/{static_total})')
    print(f'YOLO 推理耗时    : mean {np.mean(times):.1f} ms/帧')
    print(f'结果已存: {csv_path}')


if __name__ == '__main__':
    main()
