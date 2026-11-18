#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-seg 视觉辅助动态点滤除节点 (几何确认在线版)
  相机图 -> YOLOv8-seg 实例掩膜(动态类别) -> 掩膜膨胀 -> 雷达点投影到像素平面
  -> 深度连续前景簇 -> 静态地图残差判定(parked 贴图保留 / moving 悬空剔除)
  -> 发布静态点云 /points_dyn_filtered
KITTI 投影链: X_cam = P2 @ Tr @ [x,y,z,1]
在线里程计滞后解法:
  - 查询用【预测位姿】(最近两条 odom 匀速外推, 车辆加速度误差≈厘米级)
  - 入图用【精确位姿】(帧暂存, 等该帧 odom 到达后变换入图) -> 地图无滞后抖动
  - 输出不延迟, 管线吞吐不受影响
评估 GT: SemanticKITTI moving 标注(251-257)
"""
import os
import sys
import queue
import threading
from collections import deque

import numpy as np
import cv2
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from message_filters import TimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from dyn_track import (quat_rotate, nlerp, predict_pose, foreground_cluster,
                       StaticMapResidual)

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

        img_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        pc_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
        # 图像与点云由 kitti_dyn_player 同进程同时间戳发布 -> 精确同步
        ts = TimeSynchronizer([img_sub, pc_sub], 10)
        ts.registerCallback(self.cb)

        # 几何确认: 里程计缓冲 + 静态地图残差 + 延迟入图队列
        # 重活(树重建等)放独立工作线程: 若在 rospy 回调线程里周期性停顿, 同步队列积压
        # 会造成给 SLAM 的扫描流时密时疏, 早期里程计轮间随机崩坏(2026-11-18 实测定位)
        self.track_check = rospy.get_param('~track_check', True)
        self.odom_buf = deque(maxlen=30)     # [(t, q, xyz)] 工作线程写/回调线程读, 用锁
        self.odom_lock = threading.Lock()
        self.pending = deque()               # [(t, pts(N,3) float32)] 待精确位姿入图
        rospy.Subscriber('/aft_mapped_to_init', Odometry, self.odom_cb)
        self.residual = StaticMapResidual(max_scans=rospy.get_param('~map_scans', 15),
                                          voxel=rospy.get_param('~map_voxel', 1.0),
                                          nn_thresh=rospy.get_param('~nn_thresh', 0.4),
                                          rebuild_every=3.0,
                                          map_subsample=3)
        self.residual_q = queue.Queue()
        self.worker = threading.Thread(target=self._residual_worker, daemon=True)
        self.worker.start()
        self.cluster_eps = rospy.get_param('~cluster_eps', 0.8)
        self.min_cluster_pts = rospy.get_param('~min_cluster_pts', 10)
        rospy.loginfo('[dyn_filter] 几何确认: %s (nn_thresh=%.1fm map=%d帧 预测查询+延迟精确入图, 维护线程化)',
                      '开' if self.track_check else '关',
                      self.residual.nn_thresh, self.residual.max_scans)

        self.pub = rospy.Publisher('/points_dyn_filtered', PointCloud2, queue_size=2)
        self.pub_mask_img = rospy.Publisher('/dyn_filter/mask_debug', Image, queue_size=2) if self.pub_debug else None
        self.pub_proj_img = rospy.Publisher('/dyn_filter/projection_debug', Image, queue_size=2) if self.pub_debug else None
        self.proj_step = rospy.get_param('~proj_step', 5)  # 投影图每 N 个点画 1 个
        rospy.loginfo('[dyn_filter] 就绪: model=%s conf=%.2f dilate=%dpx', model_name, self.conf, self.dilate_px)

    def odom_cb(self, msg):
        # 只入队, 重活在工作线程做, 绝不在回调线程阻塞
        self.residual_q.put(('odom', msg))

    def _residual_worker(self):
        while not rospy.is_shutdown():
            item = self.residual_q.get()
            kind = item[0]
            if kind == 'odom':
                msg = item[1]
                o = msg.pose.pose.orientation
                p = msg.pose.pose.position
                with self.odom_lock:
                    self.odom_buf.append((msg.header.stamp.to_sec(),
                                          (o.x, o.y, o.z, o.w), (p.x, p.y, p.z)))
                self.flush_pending()
            elif kind == 'scan':
                _, t, pts = item
                self.pending.append((t, pts))
                self.flush_pending()

    def odom_at(self, ts):
        """插值到 ts 的位姿 (q, xyz); 无数据返回 None"""
        with self.odom_lock:
            buf = list(self.odom_buf)
        if not buf:
            return None
        if ts <= buf[0][0]:
            return buf[0][1], buf[0][2]
        if ts >= buf[-1][0]:
            return buf[-1][1], buf[-1][2]
        for i in range(len(buf) - 1):
            t0, q0, p0 = buf[i]
            t1, q1, p1 = buf[i + 1]
            if t0 <= ts < t1:
                a = (ts - t0) / max(t1 - t0, 1e-9)
                return nlerp(q0, q1, a), (1 - a) * np.array(p0) + a * np.array(p1)
        return buf[-1][1], buf[-1][2]

    def odom_at_pred(self, ts):
        """查询用位姿: 有足够 odom 时外推预测(消除查询侧滞后); 不足时插值/截断"""
        with self.odom_lock:
            buf = list(self.odom_buf)
        if not buf:
            return None
        if len(buf) >= 2 and ts > buf[-1][0]:
            t0, q0, p0 = buf[-2]
            t1, q1, p1 = buf[-1]
            return predict_pose(q0, p0, t0, q1, p1, t1, ts)
        return self.odom_at(ts)

    def flush_pending(self):
        """暂存帧等其精确 odom 到达后入图(地图无滞后抖动); 仅工作线程调用"""
        with self.odom_lock:
            buf = list(self.odom_buf)
        if not buf:
            return
        newest = buf[-1][0]
        while self.pending and self.pending[0][0] <= newest:
            t, pts = self.pending.popleft()
            q, p = self.odom_at(t)
            self.residual.add_scan(quat_rotate(q, pts) + p)
            self.residual.maybe_rebuild(t)

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
        if res.masks is not None and len(res.masks.data) > 0:
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

        # 3. 投影 + 实例级剔除判定 (前景簇 + 静态地图残差, 预测位姿查询)
        N = len(pts)
        xyz1 = np.hstack([pts[:, :3], np.ones((N, 1))]).T
        X = self.P @ xyz1
        Z = X[2]
        valid = Z > 0
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        Zc = np.where(Z > 0, Z, np.inf)  # 相机系深度(选前景簇用)

        drop = np.zeros(N, bool)
        frame_t = img_msg.header.stamp.to_sec()
        odom = self.odom_at_pred(frame_t) if self.track_check else None
        if res.masks is not None and odom is not None:
            q, t = odom
            world = quat_rotate(q, pts[:, :3]) + t
            valid_idx = np.where(in_img)[0]
            rr, cc = v[valid_idx], u[valid_idx]
            for m in masks:
                idx = valid_idx[m[rr, cc]]
                if len(idx) == 0:
                    continue
                # 深度连续前景簇: 掩码内点可能同时来自车身与车后背景, 取最小深度簇=物体本体
                clu = foreground_cluster(pts[idx, :3], Zc[idx], eps=self.cluster_eps)
                if len(clu) < self.min_cluster_pts:
                    continue  # 簇太小无法可靠判定: 保留(静止先验, 宁漏删不误删 parked)
                # 静态地图残差: parked 贴图保留, moving 悬空剔除 (只删前景簇, 背景点保留)
                if self.residual.is_moving(world[idx[clu]]):
                    drop[idx[clu]] = True
            # 本帧暂存, 等精确 odom 到达后入图 (工作线程处理, 不阻塞回调)
            self.residual_q.put(('scan', frame_t, np.ascontiguousarray(pts[:, :3], np.float32)))
        else:
            # 无里程计或关闭跟踪: 外观法回退(整掩码删除)
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
