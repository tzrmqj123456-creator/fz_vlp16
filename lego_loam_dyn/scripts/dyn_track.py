#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多帧运动确认 (dyn_track): 深度连续前景簇 + 静态地图残差判定
管线: 实例掩码取点 -> 3D欧式聚类取最小深度簇(前景, 排除掩码混入的背景)
      -> 簇点 vs 近N帧【原始】点云局部地图的中位NN -> parked(贴图≈0.05m)/moving(悬空≈0.5m+)判定
设计要点:
  - 只删前景簇: 掩码内背景点自动保留
  - 地图用原始(未滤除)点云构建: parked 车首帧就入图, 不自强化误删; 当前帧先判定后入图
  - 每帧独立判定, 无实例身份关联(身份/锚点噪声是跟踪路线的失败根源)
局限(论文可讨论): 与自车共速的目标(相对速度≈0, 轨迹与自身重合)原理性漏检
"""
import numpy as np
from collections import deque
from scipy.spatial import cKDTree


def quat_rotate(q, pts):
    """四元数 (x,y,z,w) 旋转点集 (N,3) -> (N,3)"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return pts @ R.T


def nlerp(q0, q1, a):
    """四元数线性插值+归一化"""
    q = (1 - a) * np.array(q0) + a * np.array(q1)
    return q / (np.linalg.norm(q) + 1e-12)


def foreground_cluster(pts_lidar, z_cam, eps=0.8, min_pts=3):
    """3D 欧式聚类, 返回【平均相机深度最小且深度方差小】的簇索引
    (前景物体离自车最近; 方差约束排除前景地面条带——地面深度跨度大而物体紧凑)
    pts_lidar: (N,3) 雷达系; z_cam: (N,) 相机系深度(仅用于选簇)"""
    n = len(pts_lidar)
    if n == 0:
        return np.array([], np.int64)
    tree = cKDTree(pts_lidar)
    visited = np.zeros(n, bool)
    best_idx, best_score = np.array([], np.int64), np.inf
    for seed in range(n):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        clu = []
        while stack:
            i = stack.pop()
            clu.append(i)
            for j in tree.query_ball_point(pts_lidar[i], eps):
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        if len(clu) >= min_pts:
            z_mean = float(z_cam[clu].mean())
            z_std = float(z_cam[clu].std())
            if z_std > 8.0:
                continue  # 深度跨度大 -> 地面条带等背景, 跳过
            score = z_mean + 0.3 * z_std
            if score < best_score:
                best_score = score
                best_idx = np.array(clu, np.int64)
    return best_idx


class StaticMapResidual:
    """局部静态地图残差判定: 前景簇 vs 近N帧原始点云地图的中位NN
    parked: 簇点贴着自己的地图副本(NN≈0.05m) -> 保留
    moving: 当前位置地图只有路面/背景(NN≈0.5~2m, 物体悬在路面上) -> 剔除
    慢速/共速目标: 轨迹与自身重合 -> 原理性漏检(已知局限)"""

    def __init__(self, max_scans=30, voxel=1.0, nn_thresh=0.4,
                 rebuild_every=1.0, map_subsample=2):
        self.max_scans = max_scans
        self.voxel = voxel
        self.nn_thresh = nn_thresh
        self.rebuild_every = rebuild_every
        self.map_subsample = map_subsample
        self.buffer = deque()
        self.tree = None
        self.last_build = -1e9

    def add_scan(self, pts_world):
        """入图: 原始(未滤除)点云, 降采样+体素降采样后入环形缓冲
        性能: 体素格打包成 int64 一维后 unique(比三维 unique 快数倍)"""
        pts = pts_world[::self.map_subsample]
        q = (pts / self.voxel).round().astype(np.int32)
        packed = q[:, 0].astype(np.int64) * 16777216 + q[:, 1].astype(np.int64) * 4096 + q[:, 2].astype(np.int64)
        _, idx = np.unique(packed, return_index=True)
        self.buffer.append(pts[idx])
        while len(self.buffer) > self.max_scans:
            self.buffer.popleft()

    def maybe_rebuild(self, now):
        """节流重建 KD 树"""
        if now - self.last_build >= self.rebuild_every and self.buffer:
            self.tree = cKDTree(np.vstack(list(self.buffer)))
            self.last_build = now

    def nn_dist(self, pts_world):
        """簇点到地图的中位最近邻距离; 无地图返回 inf(保守: 视为 moving)"""
        if self.tree is None or len(pts_world) == 0:
            return np.inf
        return float(np.median(self.tree.query(pts_world, k=1)[0]))

    def is_moving(self, pts_world):
        return self.nn_dist(pts_world) > self.nn_thresh
