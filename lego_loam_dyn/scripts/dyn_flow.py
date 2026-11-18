#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最轻量光流动态判定 (dyn_flow): 零新依赖、CPU 推理、不占显存
  1. cv2 Farneback 光流(降采样 640x192, 12 线程 CPU ~15-25ms)
  2. 雷达投影深度 + 光流 -> 最小二乘自车运动拟合 (v, omega), 迭代加权去动态点
  3. 残差光流 = |观测 - 期望| -> 实例级运动判定
关键设计: 自车运动从光流本身拟合, 完全不依赖里程计 ——
  绕开"里程计噪声吞没运动信号"的死穴(残差法失败的根因);
  静止场景的光流被 ego-motion 完全解释(残差≈0), 移动物体残差突出。
线性式(每点 2 方程, 6 未知): u'=(1/Z)(-fx·vx+u·vz+(uv/fx)wx-(fx+u²/fx)wy+v·wz)
                           v'=(1/Z)(-fy·vy+v·vz+(fy+v²/fy)wx-(uv/fy)wy-u·wz)
"""
import numpy as np
import cv2

FLOW_W, FLOW_H = 640, 192


def compute_flow(g0, g1):
    """相邻帧灰度图(原尺寸) -> 光流场 (FLOW_H, FLOW_W, 2), 降采样空间"""
    s0 = cv2.resize(g0, (FLOW_W, FLOW_H))
    s1 = cv2.resize(g1, (FLOW_W, FLOW_H))
    return cv2.calcOpticalFlowFarneback(s0, s1, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def fit_egomotion(u, v, Z, fu, fv, K, thresh=1.5):
    """由(像素,深度,光流)对应点拟合自车运动 (vx,vy,vz,wx,wy,wz)
    两轮迭代最小二乘: 第一轮全点, 第二轮只留残差<thresh 的内点(动态点被自动剔除)
    u,v: 像素坐标(原始分辨率); Z: 相机系深度; fu,fv: 该点光流(原始分辨率像素单位)
    K: 3x3 内参(原始分辨率); 返回 (sol6, 内点率) 或 (None, 0)"""
    n = len(u)
    if n < 200:
        return None, 0.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu = (u - cx).astype(np.float64)
    vv = (v - cy).astype(np.float64)
    zz = Z.astype(np.float64)
    fu = fu.astype(np.float64)
    fv = fv.astype(np.float64)
    inl_mask = np.ones(n, bool)
    sol = None
    for _ in range(2):
        m = inl_mask.sum()
        A = np.zeros((2 * m, 6))
        b = np.zeros(2 * m)
        uu_, vv_, zz_ = uu[inl_mask], vv[inl_mask], zz[inl_mask]
        fu_, fv_ = fu[inl_mask], fv[inl_mask]
        A[0::2, 0] = -fx / zz_
        A[0::2, 2] = uu_ / zz_
        A[0::2, 3] = (uu_ * vv_ / fx) / zz_
        A[0::2, 4] = -(fx + uu_ * uu_ / fx) / zz_
        A[0::2, 5] = vv_ / zz_
        b[0::2] = fu_
        A[1::2, 1] = -fy / zz_
        A[1::2, 2] = vv_ / zz_
        A[1::2, 3] = (fy + vv_ * vv_ / fy) / zz_
        A[1::2, 4] = -(uu_ * vv_ / fy) / zz_
        A[1::2, 5] = -uu_ / zz_
        b[1::2] = fv_
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = np.abs(A @ sol - b)
        inl_new = np.zeros(n, bool)
        inl_new[inl_mask] = (res[0::2] < thresh) & (res[1::2] < thresh)
        if inl_new.sum() < 0.5 * m or inl_new.sum() == m:
            inl_mask = inl_new
            break
        inl_mask = inl_new
    return sol, float(inl_mask.mean())


def expected_flow(u, v, Z, sol, K):
    """由 ego-motion 计算各点的期望光流(原始分辨率像素单位)"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu = (u - cx).astype(np.float64)
    vv = (v - cy).astype(np.float64)
    zz = Z.astype(np.float64)
    vx, vy, vz, wx, wy, wz = sol
    eu = (-fx * vx + uu * vz + (uu * vv / fx) * wx - (fx + uu * uu / fx) * wy + vv * wz) / zz
    ev = (-fy * vy + vv * vz + (fy + vv * vv / fy) * wx - (uu * vv / fy) * wy - uu * wz) / zz
    return eu, ev
