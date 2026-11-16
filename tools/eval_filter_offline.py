#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3层指标 ①: 动态滤除质量离线评估 (不需要跑 ROS 管线)
对抽样帧: 图像 -> YOLOv8-seg 掩膜(与 dyn_filter 相同参数) -> 点云投影
          -> 实例级剔除判定【外观掩膜 + 几何二次确认】(与 dyn_filter 相同逻辑)
对照 SemanticKITTI moving 标注计算:
  Dynamic Recall    = 剔除的真动态点 / GT 动态点
  Dynamic Precision = 剔除的点中真动态占比
  Static Retention  = 保留的静态点 / GT 静态点 (误删度量, 关键)
  F1
几何二次确认的里程计来源: /tmp/aft_05.csv (最近一次运行的 /aft_mapped_to_init 记录);
  不存在时自动回退为纯外观法(结果会带警告)。
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
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]   # COCO: person..truck (与 dyn_filter 一致)
MOVING_IDS = set(range(251, 258))          # SemanticKITTI moving 段
STATIC_EXCLUDE = MOVING_IDS | {0, 1}
CONF, DILATE = 0.35, 3
IMG_W, IMG_H = 1242, 375
STEP = 10  # 每 10 帧评估 1 帧
MIN_INST_PTS = 10

sys.path.insert(0, f'{ROOT}/lego_loam_dyn/scripts')
from geo_confirm import GeoConfirm, quat_rotate, largest_cluster


def load_calib():
    data = {}
    for line in open(CALIB):
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = data['P2'].reshape(3, 4)
    Tr = np.vstack([data['Tr'].reshape(3, 4), [0, 0, 0, 1]])
    return P2 @ Tr  # 3x4 velodyne -> rectified 左目像素


def load_odom():
    """从 rostopic echo -p 的 aft csv 读里程计: 时间戳 + 四元数 + 平移"""
    if not os.path.exists(ODY_CSV):
        return None, None, None
    ts, qs, ts_xyz = [], [], []
    with open(ODY_CSV) as f:
        for r in csv.DictReader(f):
            ts.append(float(r['field.header.stamp']) / 1e9)
            qs.append([float(r['field.pose.pose.orientation.x']),
                       float(r['field.pose.pose.orientation.y']),
                       float(r['field.pose.pose.orientation.z']),
                       float(r['field.pose.pose.orientation.w'])])
            ts_xyz.append([float(r['field.pose.pose.position.x']),
                           float(r['field.pose.pose.position.y']),
                           float(r['field.pose.pose.position.z'])])
    return np.array(ts), np.array(qs), np.array(ts_xyz)


def main():
    P = load_calib()
    model = YOLO(MODEL)
    odom_ts, odom_q, odom_t = load_odom()
    geo_on = odom_ts is not None
    if not geo_on:
        print('警告: 无 /tmp/aft_05.csv, 几何二次确认关闭(纯外观法结果)')
    gc = GeoConfirm(max_scans=20, voxel=1.0, nn_dist=1.0, moving_ratio=0.5)

    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
    tot = tp = fp = fn = 0
    static_kept = static_total = 0
    n_static_saved = n_moving_removed = 0  # 实例级统计
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
        if res.masks is not None:
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

        # 里程计 + 世界系
        t_frame = 0.1 * i
        world = None
        if geo_on:
            oi = int(np.argmin(np.abs(odom_ts - t_frame)))
            world = quat_rotate(odom_q[oi], pts[:, :3]) + odom_t[oi]

        # 实例级判定 (与 dyn_filter 相同)
        drop = np.zeros(N, bool)
        if masks is not None and geo_on:
            valid_idx = np.where(in_img)[0]
            rr, cc = v[valid_idx], u[valid_idx]  # 掩膜索引: m[行=v, 列=u]
            for m in masks:
                idx = valid_idx[m[rr, cc]]
                if len(idx) == 0:
                    continue
                # 3D 聚类摘出物体本体, 只对本体簇做判定 (与 dyn_filter 一致)
                clu = largest_cluster(pts[idx, :3])
                if len(clu) < MIN_INST_PTS or gc.is_moving(world[idx[clu]]):
                    drop[idx] = True
                else:
                    n_static_saved += 1
            gc.add_scan(world)
            gc.maybe_rebuild(t_frame)
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
    print(f'===== 动态滤除质量 (评估 {len(rows)} 帧, {os.path.basename(MODEL)}, 几何确认: {"开" if geo_on else "关(纯外观)"}) =====')
    print(f'Dynamic Recall   : {recall:.3f}  (tp={tp}, fn={fn})')
    print(f'Dynamic Precision: {precision:.3f}  (fp={fp})')
    print(f'F1               : {f1:.3f}')
    print(f'Static Retention : {retention:.4f}  (保留 {static_kept}/{static_total})')
    print(f'实例判定: 判静态救回 {n_static_saved} 个实例')
    print(f'YOLO 推理耗时    : mean {np.mean(times):.1f} ms/帧')
    print(f'结果已存: {csv_path}')


if __name__ == '__main__':
    main()
