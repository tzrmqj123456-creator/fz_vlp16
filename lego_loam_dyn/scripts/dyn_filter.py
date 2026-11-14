#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-seg 视觉辅助动态点滤除节点
  相机图 -> YOLOv8-seg 实例掩膜(动态类别) -> 掩膜膨胀 -> 雷达点投影到像素平面
  -> 落在动态掩膜内的点剔除 -> 发布静态点云 /points_dyn_filtered
KITTI 投影链: X_cam = P2 @ R0_rect @ Tr_velo_to_cam @ [x,y,z,1]
评估 GT 建议: SemanticKITTI moving 标注
"""
import numpy as np
import cv2
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from message_filters import TimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO

# COCO 动态类别白名单: person, bicycle, car, motorcycle, bus, train, truck
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]


class DynFilter:
    def __init__(self):
        calib_path = rospy.get_param('~calib_path')
        model_name = rospy.get_param('~model', 'yolov8s-seg.pt')
        self.conf = rospy.get_param('~conf_thres', 0.35)
        self.dilate_px = rospy.get_param('~dilate_px', 3)
        self.pub_debug = rospy.get_param('~pub_debug', True)

        self.P2, self.R0, self.Tr = self.load_calib(calib_path)
        self.P = self.P2 @ self.R0 @ self.Tr  # 3x4: velodyne -> 左目像素
        rospy.loginfo('[dyn_filter] 投影矩阵 P = %s', np.array2string(self.P, precision=3))
        self.model = YOLO(model_name)
        self.bridge = CvBridge()

        img_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        pc_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
        # 图像与点云由 kitti_dyn_player 同进程同时间戳发布 -> 精确同步
        ts = TimeSynchronizer([img_sub, pc_sub], 10)
        ts.registerCallback(self.cb)

        self.pub = rospy.Publisher('/points_dyn_filtered', PointCloud2, queue_size=2)
        self.pub_mask_img = rospy.Publisher('/dyn_filter/mask_debug', Image, queue_size=2) if self.pub_debug else None
        rospy.loginfo('[dyn_filter] 就绪: model=%s conf=%.2f dilate=%dpx', model_name, self.conf, self.dilate_px)

    @staticmethod
    def load_calib(path):
        """解析 KITTI calib.txt: P2(3x4), R0_rect(3x3), Tr_velo_to_cam(3x4)"""
        data = {}
        with open(path) as f:
            for line in f:
                k, v = line.split(':', 1)
                data[k.strip()] = np.array([float(x) for x in v.split()])
        P2 = data['P2'].reshape(3, 4)
        R0 = np.eye(4)
        R0[:3, :3] = data['R0_rect'].reshape(3, 3)
        Tr = data['Tr_velo_to_cam'].reshape(3, 4)
        return P2, R0, Tr

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

        # 3. 投影滤除 (向量化)
        N = len(pts)
        xyz1 = np.hstack([pts[:, :3], np.ones((N, 1))]).T
        X = self.P @ xyz1
        Z = X[2]
        valid = Z > 0
        u = np.round(X[0] / np.maximum(Z, 1e-6)).astype(np.int64)
        v = np.round(X[1] / np.maximum(Z, 1e-6)).astype(np.int64)
        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        drop = np.zeros(N, bool)
        drop[in_img] = mask[v[in_img], u[in_img]]
        keep = ~drop

        # 4. 发布静态点云 (保留原字段结构)
        out = pc2.create_cloud(pc_msg.header, pc_msg.fields, pts[keep])
        self.pub.publish(out)
        if self.pub_mask_img is not None:
            dbg = img.copy()
            dbg[mask] = (dbg[mask] * 0.4 + np.array([0, 0, 180], np.uint8) * 0.6).astype(np.uint8)
            self.pub_mask_img.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))
        rospy.loginfo_throttle(5, '[dyn_filter] 帧 %s: %d/%d 点剔除, 掩膜实例 %d',
                               pc_msg.header.stamp, int(drop.sum()), N, int(mask.sum() > 0))


if __name__ == '__main__':
    rospy.init_node('dyn_filter')
    DynFilter()
    rospy.spin()
