#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-seg 视觉辅助动态点滤除节点 (外观候选 + 光流确认)
  相机图 -> YOLOv8-seg 实例掩膜(动态类别) -> 掩膜膨胀 -> 雷达点投影到像素平面
  -> RAFT-small 光流 + 自车运动自拟合 -> 实例前景簇的光流残差(度量归一化 m/帧)
  -> 残差 > 阈值(0.4 m/帧, parked 噪声底之上)才剔除, parked 车保留
KITTI 投影链: X_cam = P2 @ Tr @ [x,y,z,1]
评估 GT: SemanticKITTI moving 标注(251-257)
"""
import os
import sys

import numpy as np
import cv2
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from message_filters import TimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from dyn_flow import (RaftFlow, egomotion_fit, instance_residual, foreground_cluster,
                      FLOW_W, FLOW_H)

# COCO 动态类别白名单: person, bicycle, car, motorcycle, bus, train, truck
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]


class DynFilter:
    def __init__(self):
        calib_path = rospy.get_param('~calib_path')
        model_name = rospy.get_param('~model', 'yolov8s-seg.pt')
        self.conf = rospy.get_param('~conf_thres', 0.35)
        self.dilate_px = rospy.get_param('~dilate_px', 3)
        self.pub_debug = rospy.get_param('~pub_debug', True)

        self.P = self.load_calib(calib_path)  # 3x4: velodyne -> 左目(rectified)像素
        rospy.loginfo('[dyn_filter] 投影矩阵 P = %s', np.array2string(self.P, precision=3))
        self.model = YOLO(model_name)
        self.bridge = CvBridge()

        # 光流确认: RAFT-small(640x192, ~35ms) + 自车运动自拟合 + 实例残差
        self.flow_check = rospy.get_param('~flow_check', True)
        self.flow_thresh = rospy.get_param('~flow_thresh', 0.4)      # m/帧, parked 噪声底之上
        self.fit_subsample = rospy.get_param('~fit_subsample', 8)    # ego 拟合点降采样
        self.raft = RaftFlow() if self.flow_check else None
        self.Ks = None
        if self.flow_check:
            self.Ks = self.load_K()
        rospy.loginfo('[dyn_filter] 光流确认: %s (thresh=%.2f m/帧, RAFT %dx%d)',
                      '开' if self.flow_check else '关', self.flow_thresh, FLOW_W, FLOW_H)

        img_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        pc_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
        # 图像与点云由 kitti_dyn_player 同进程同时间戳发布 -> 精确同步
        ts = TimeSynchronizer([img_sub, pc_sub], 10)
        ts.registerCallback(self.cb)

        self.pub = rospy.Publisher('/points_dyn_filtered', PointCloud2, queue_size=2)
        self.pub_mask_img = rospy.Publisher('/dyn_filter/mask_debug', Image, queue_size=2) if self.pub_debug else None
        self.pub_proj_img = rospy.Publisher('/dyn_filter/projection_debug', Image, queue_size=2) if self.pub_debug else None
        self.proj_step = rospy.get_param('~proj_step', 5)  # 投影图每 N 个点画 1 个
        rospy.loginfo('[dyn_filter] 就绪: model=%s conf=%.2f dilate=%dpx', model_name, self.conf, self.dilate_px)

    def load_K(self):
        """内参(缩放至光流分辨率)"""
        data = {}
        with open(rospy.get_param('~calib_path')) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    data[k.strip()] = np.array([float(x) for x in v.split()])
        K = data['P2'].reshape(3, 4)[:3, :3].copy()
        K[0, 0] *= FLOW_W / 1242.0
        K[0, 2] *= FLOW_W / 1242.0
        K[1, 1] *= FLOW_H / 375.0
        K[1, 2] *= FLOW_H / 375.0
        return K

    @staticmethod
    def load_calib(path):
        """解析 KITTI 标定, 返回 3x4 投影矩阵 P (velodyne -> rectified 左目像素)
        支持两种格式:
          raw 格式:      P = P2 @ R0_rect @ Tr_velo_to_cam (calib.txt 只有 P0-P3 时缺这两项,
                         它们在 calib_cam_to_cam.txt / calib_velo_to_cam.txt 里, 勿用)
          odometry 格式: P = P2 @ Tr (Tr 已折叠 rectification, 推荐用 data_odometry_calib 的 calib.txt)"""
        data = {}
        with open(path) as f:
            for line in f:
                if ':' not in line:
                    continue
                k, v = line.split(':', 1)
                data[k.strip()] = np.array([float(x) for x in v.split()])
        P2 = data['P2'].reshape(3, 4)

        def to_4x4(tr):
            return np.vstack([tr.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])

        if 'R0_rect' in data and 'Tr_velo_to_cam' in data:
            R0 = np.eye(4)
            R0[:3, :3] = data['R0_rect'].reshape(3, 3)
            return P2 @ R0 @ to_4x4(data['Tr_velo_to_cam'])
        if 'Tr' in data:
            return P2 @ to_4x4(data['Tr'])
        raise KeyError('标定文件缺 R0_rect/Tr_velo_to_cam/Tr: %s' % path)

    def cb(self, img_msg, pc_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        H, W = img.shape[:2]

        # 1. YOLOv8-seg 推理, 取动态类别实例掩膜 (推理尺寸 -> 重采样回原图尺寸)
        res = self.model(img, conf=self.conf, classes=DYNAMIC_CLASSES, verbose=False)[0]
        mask = np.zeros((H, W), bool)
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)  # (n, hm, wm) 推理分辨率
            masks = np.stack([cv2.resize(m.astype(np.uint8), (W, H),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                              for m in masks])
            if self.dilate_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * self.dilate_px + 1, 2 * self.dilate_px + 1))
                masks = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks])
            mask = masks.any(0)

        # 2. 点云读取 (XYZI 快速路径 + 通用回退)
        step = pc_msg.point_step // 4
        names = [f.name for f in pc_msg.fields]
        if names[:4] == ['x', 'y', 'z', 'intensity'] and step == 4:
            pts = np.frombuffer(pc_msg.data, np.float32).reshape(-1, 4)
        else:
            pts = np.array(list(pc2.read_points(pc_msg, skip_nans=True)), np.float32)

        # 3. 投影 + 实例级判定 (外观候选 + 光流确认)
        N = len(pts)
        xyz1 = np.hstack([pts[:, :3], np.ones((N, 1))]).T
        X = self.P @ xyz1
        Z = X[2]
        valid = Z > 0
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        drop = np.zeros(N, bool)

        flow = self.raft(img) if self.flow_check else None
        flow_ret = None
        if flow is not None:
            # 自车运动自拟合(降采样点, 不依赖里程计)
            idx_fit = np.where(in_img)[0][::self.fit_subsample]
            us_f = (u[idx_fit] * (FLOW_W / 1242.0)).astype(float)
            vs_f = (v[idx_fit] * (FLOW_H / 375.0)).astype(float)
            flow_ret = egomotion_fit(us_f, vs_f, Z[idx_fit], flow, self.Ks)

        if res.masks is not None and flow_ret is not None:
            R, t = flow_ret[1], flow_ret[2]
            valid_idx = np.where(in_img)[0]
            rr, cc = v[valid_idx], u[valid_idx]
            Zc = np.where(Z > 0, Z, np.inf)
            for m in masks:
                idx = valid_idx[m[rr, cc]]
                if len(idx) == 0:
                    continue
                # 前景簇: 只取物体本体点算残差(掩码边缘背景点会抬高噪声底)
                clu = foreground_cluster(pts[idx, :3], Zc[idx])
                if len(clu) < 10:
                    continue
                fidx = idx[clu]
                resid = instance_residual((u[fidx] * (FLOW_W / 1242.0)).astype(float),
                                          (v[fidx] * (FLOW_H / 375.0)).astype(float),
                                          Z[fidx], flow, R, t, self.Ks)
                if resid > self.flow_thresh:
                    drop[idx] = True    # 光流确认 moving: 删整掩码(parked 车保留)
        else:
            # 光流不可用(首帧/拟合失败): 外观法回退(整掩码删除)
            drop[in_img] = mask[v[in_img], u[in_img]]
        keep = ~drop

        # 4. 发布静态点云 (保留原字段结构)
        out = pc2.create_cloud(pc_msg.header, pc_msg.fields, pts[keep])
        self.pub.publish(out)
        if self.pub_mask_img is not None:
            dbg = img.copy()
            dbg[mask] = (dbg[mask] * 0.4 + np.array([0, 0, 180], np.uint8) * 0.6).astype(np.uint8)
            self.pub_mask_img.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))

        if self.pub_proj_img is not None:
            # 点云投影图: 图像与点云同一坐标系(像素平面), 保留点按深度着色, 剔除的动态点标红
            depth_img = np.zeros((H, W), np.uint8)
            ik = np.where(keep)[0][::self.proj_step]
            uk, vk = u[ik], v[ik]
            vk_valid = (uk >= 0) & (uk < W) & (vk >= 0) & (vk < H)
            uk, vk = uk[vk_valid], vk[vk_valid]
            if len(uk) > 0:
                dep = np.linalg.norm(pts[ik][vk_valid, :3], axis=1)
                dnorm = np.clip((dep - dep.min()) / (dep.max() - dep.min() + 1e-6), 0, 1)
                depth_img[vk, uk] = (dnorm * 255).astype(np.uint8)
            colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            vis = img.copy()
            # 只给有点的像素着色, 背景保持原图(否则无点区域被 JET 的 0=深蓝 染蓝)
            pmask = depth_img > 0
            vis[pmask] = (img[pmask] * 0.25 + colored[pmask] * 0.75).astype(np.uint8)
            id_ = np.where(drop)[0][::self.proj_step]
            ud, vd = u[id_], v[id_]
            vd_valid = (ud >= 0) & (ud < W) & (vd >= 0) & (vd < H)
            vis[vd[vd_valid], ud[vd_valid]] = (0, 0, 255)
            self.pub_proj_img.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        rospy.loginfo_throttle(5, '[dyn_filter] 帧 %s: %d/%d 点剔除, 掩膜实例 %d',
                               pc_msg.header.stamp, int(drop.sum()), N, int(mask.sum() > 0))


if __name__ == '__main__':
    rospy.init_node('dyn_filter')
    DynFilter()
    rospy.spin()
