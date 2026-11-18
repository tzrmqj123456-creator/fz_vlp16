#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流动态判定 (dyn_flow): Farneback/RAFT-small 光流 + 雷达深度 -> 自车运动估计 -> 实例残差
关键: 离散形式(MODT 风格)——0.1s 帧间隔下近处静态点流达 20~50px, 连续时间线性近似
在"当前点位置"评估流会有几十像素的系统误差; 正确做法是:
  1. 用当前点云 (u, Z) 反投影 3D, 用 (R,t) 预测其上一帧位置 u_prev
  2. 在 u_prev 采样观测光流, 与期望位移 (u - u_prev) 比较
  3. 残差 < 阈值 的点视为静态内点, 迭代精化 (R,t)
自车运动从光流本身拟合, 完全不依赖里程计。
"""
import numpy as np
import cv2
import torch

FLOW_W, FLOW_H = 640, 192   # 在线用(35ms/帧); 离线验证可改 1240x376 降噪声


def compute_flow(g0, g1):
    """相邻帧灰度图(原尺寸) -> 光流场 (FLOW_H, FLOW_W, 2), 降采样空间 (Farneback CPU)"""
    s0 = cv2.resize(g0, (FLOW_W, FLOW_H))
    s1 = cv2.resize(g1, (FLOW_W, FLOW_H))
    return cv2.calcOpticalFlowFarneback(s0, s1, None, 0.5, 3, 21, 3, 5, 1.2, 0)


class RaftFlow:
    """RAFT-small (torchvision, fp32 GPU): 噪声底 ~0.3-0.5px, 远优于 Farneback ~3px"""

    def __init__(self):
        import torchvision.models.optical_flow as ofm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # fp32: torchvision RAFT 内部 grid_sample 与 half 不兼容
        self.model = ofm.raft_small(weights=ofm.Raft_Small_Weights.DEFAULT).to(self.device).eval()
        self.prev = None

    def __call__(self, img_rgb):
        """img_rgb: HxWx3 uint8 -> 光流场 (FLOW_H, FLOW_W, 2) 或 None(第一帧)"""
        im = cv2.resize(img_rgb, (FLOW_W, FLOW_H))
        t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        if self.prev is None:
            self.prev = t
            return None
        with torch.no_grad():
            flow = self.model(self.prev, t)[-1]   # (1, 2, H, W)
        self.prev = t
        # 注意: torchvision RAFT 返回的是 img2->img1 反向流(光度检验实证: warp(-flow) 误差 11 vs +flow 36),
        # 取负得到标准前向约定(位移: img1->img2)
        return -flow[0].permute(1, 2, 0).cpu().numpy()


def _backproject(u, v, Z, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u - cx) / fx * Z, (v - cy) / fy * Z, Z], axis=1)


def _project(X, K):
    return K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]


def _linear_fit(u, v, Z, fu, fv, K):
    """连续时间线性模型(仅作初值): Bruss-Horn 式, 6 未知 (vx,vy,vz,wx,wy,wz)"""
    n = len(u)
    if n < 200:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu = (u - cx).astype(np.float64)
    vv = (v - cy).astype(np.float64)
    zz = Z.astype(np.float64)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    A[0::2, 0] = -fx / zz
    A[0::2, 2] = uu / zz
    A[0::2, 3] = (uu * vv / fx) / zz
    A[0::2, 4] = -(fx + uu * uu / fx) / zz
    A[0::2, 5] = vv / zz
    b[0::2] = fu
    A[1::2, 1] = -fy / zz
    A[1::2, 2] = vv / zz
    A[1::2, 3] = (fy + vv * vv / fy) / zz
    A[1::2, 4] = -(uu * vv / fy) / zz
    A[1::2, 5] = -uu / zz
    b[1::2] = fv
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def _sol2se3(sol):
    v = sol[:3]
    w = sol[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        R = np.eye(3)
    else:
        k = w / theta
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
    return R, v


def _sample_flow(flow, u, v):
    """双线性采样光流场 (FLOW_H, FLOW_W, 2)"""
    u = np.clip(u, 0, FLOW_W - 1.001)
    v = np.clip(v, 0, FLOW_H - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    f00 = flow[v0, u0]
    f10 = flow[v0, np.minimum(u0 + 1, FLOW_W - 1)]
    f01 = flow[np.minimum(v0 + 1, FLOW_H - 1), u0]
    f11 = flow[np.minimum(v0 + 1, FLOW_H - 1), np.minimum(u0 + 1, FLOW_W - 1)]
    return (f00 * (1 - du) * (1 - dv) + f10 * du * (1 - dv) + f01 * (1 - du) * dv + f11 * du * dv)


def egomotion_fit(u, v, Z, flow, K, iters=6, thresh=2.0):
    """高斯-牛顿迭代拟合自车运动 (vx,vy,vz,wx,wy,wz):
    初值 = 典型车速(前向 0.8 m/帧), 代价 = 离散 warp 残差(观测流 - 期望位移),
    数值雅可比 + 内点加权。返回 (state6, R, t, 内点率); 失败返回 None"""
    n = len(u)
    if n < 200:
        return None
    uu, vv, ZZ = u.astype(np.float64), v.astype(np.float64), Z.astype(np.float64)
    X = _backproject(uu, vv, ZZ, K)
    state = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    eps = 1e-4
    for _ in range(iters):
        R, t = _sol2se3(state)
        Xp = (R.T @ (X - t).T).T
        up, vp = _project(Xp, K)
        fu_obs = _sample_flow(flow, up, vp)[:, 0]
        fv_obs = _sample_flow(flow, up, vp)[:, 1]
        r_u = fu_obs - (uu - up)
        r_v = fv_obs - (vv - vp)
        w = (np.hypot(r_u, r_v) < thresh).astype(np.float64)
        if w.sum() < 200:
            return None
        # 数值雅可比: 6 列扰动
        J = np.zeros((n * 2, 6))
        for j in range(6):
            st = state.copy()
            st[j] += eps
            Rj, tj = _sol2se3(st)
            Xpj = (Rj.T @ (X - tj).T).T
            upj, vpj = _project(Xpj, K)
            J[0::2, j] = ((uu - upj) - (uu - up)) / eps
            J[1::2, j] = ((vv - vpj) - (vv - vp)) / eps
        r = np.concatenate([r_u, r_v])
        W = np.repeat(w, 2)
        A = J[W > 0]
        b = r[W > 0]
        # 车辆运动先验(岭正则): 地面车辆横向速度≈0, 抑制 vx/vy/ω 与平动-转动耦合的病态解
        lam = 30.0
        P = np.sqrt(lam) * np.diag([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])   # vz 不约束
        A = np.vstack([A, P])
        b = np.concatenate([b, np.zeros(6)])
        delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        state = state + delta
    R, t = _sol2se3(state)
    Xp = (R.T @ (X - t).T).T
    up, vp = _project(Xp, K)
    fu_obs = _sample_flow(flow, up, vp)[:, 0]
    fv_obs = _sample_flow(flow, up, vp)[:, 1]
    resid = np.hypot(fu_obs - (uu - up), fv_obs - (vv - vp))
    return state, R, t, float((resid < thresh).mean())


def instance_residual(u, v, Z, flow, R, t, K):
    """实例的【度量归一化】残差 (m/帧): 残差流 × Z/fx = 横向运动速度
    像素残差随深度放大(残差 ∝ 1/Z, 近处 parked 车残差天然大), 乘深度归一化后
    parked 车 ≈ 0.02-0.05 m/帧(任意距离), moving 车(相对 1-5m/s = 0.1-0.5 m/帧)可干净分离
    注意: 只应传入物体本体点(前景簇)——掩码边缘混入的背景点深度不连续, 期望流错误"""
    uu, vv, ZZ = u.astype(np.float64), v.astype(np.float64), Z.astype(np.float64)
    X = _backproject(uu, vv, ZZ, K)
    Xp = (R.T @ (X - t).T).T
    up, vp = _project(Xp, K)
    fu_obs = _sample_flow(flow, up, vp)[:, 0]
    fv_obs = _sample_flow(flow, up, vp)[:, 1]
    resid_px = np.hypot(fu_obs - (uu - up), fv_obs - (vv - vp))
    resid_m = resid_px * ZZ / K[0, 0]   # 米/帧
    return float(np.median(resid_m))


def foreground_cluster(pts_lidar, z_cam, eps=0.8, min_pts=3, min_height=0.5):
    """3D 欧式聚类, 返回【平均相机深度最小且深度方差小、垂直高度≥0.5m】的簇索引
    (深度最小=物体比背景近; 方差约束排除地面条带; 高度约束排除路面斑块)
    pts_lidar: (N,3) 雷达系; z_cam: (N,) 相机系深度(仅用于选簇)"""
    from scipy.spatial import cKDTree
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
                continue
            height = float(pts_lidar[clu, 2].max() - pts_lidar[clu, 2].min())
            if height < min_height:
                continue
            score = z_mean + 0.3 * z_std
            if score < best_score:
                best_score = score
                best_idx = np.array(clu, np.int64)
    return best_idx
