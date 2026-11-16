#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何二次确认 (GeoConfirm): 里程计补偿的局部静态地图 + 近邻残差, 区分 parked(静态) / moving(动态)
原理(Dynablox 风格): 候选点变换到世界系后, 与"近 N 帧原始点云构建的局部地图"做最近邻:
  parked 车: 地图里同位置有它自己(前几帧看到过) -> 残差小 -> 判静态 -> 保留
  moving 车: 当前位置地图里是空路面/背景 -> 残差大 -> 判动态 -> 剔除
关键设计:
  1. 地图用【未滤除】的原始点云构建——若用滤除后的点, parked 车在首帧被外观法误删后将永不入图,
     自强化误删(鸡生蛋问题)
  2. 当前帧【先检查后入图】——避免候选点与自己重合导致恒判静态
局限(论文可讨论): 每帧位移 < nn_dist 的慢速目标(约 <2.5 m/s)会漏过
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


class GeoConfirm:
    def __init__(self, max_scans=25, voxel=1.0, nn_dist=0.5, static_ratio=0.7, rebuild_every=1.5):
        self.max_scans = max_scans
        self.voxel = voxel
        self.nn_dist = nn_dist
        self.static_ratio = static_ratio
        self.rebuild_every = rebuild_every
        self.buffer = deque()
        self.tree = None
        self.last_build = -1e9

    def add_scan(self, pts_world):
        """入图: 原始(未滤除)点云, 体素降采样后入环形缓冲"""
        q = (pts_world / self.voxel).round().astype(np.int32)
        _, idx = np.unique(q, axis=0, return_index=True)
        self.buffer.append(pts_world[idx])
        while len(self.buffer) > self.max_scans:
            self.buffer.popleft()

    def maybe_rebuild(self, now):
        """节流重建 KD 树"""
        if now - self.last_build >= self.rebuild_every and self.buffer:
            self.tree = cKDTree(np.vstack(list(self.buffer)))
            self.last_build = now

    def static_fraction(self, pts_world):
        """候选点与静态地图的重合比例; 无地图返回 0(保守: 视为动态)"""
        if self.tree is None or len(pts_world) == 0:
            return 0.0
        d, _ = self.tree.query(pts_world, k=1)
        return float((d < self.nn_dist).mean())

    def is_static(self, pts_world):
        return self.static_fraction(pts_world) >= self.static_ratio
