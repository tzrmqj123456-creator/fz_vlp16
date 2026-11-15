#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI 同步播放器(动态滤除管线专用): 同一进程内以同一时间戳发布 图像 + 点云,
避免两个独立 10Hz 播放器的起始偏差导致掩膜与点云错位。
用法: rosrun lego_loam_dyn kitti_dyn_player.py _img_dir:=... _velodyne_dir:=...
话题: /kitti/camera_color_left/image_raw (1242x375 标定原生尺寸)
      /velodyne_points (XYZI, frame_id=velodyne, 与 kitti_player 完全一致)
"""
import os
import numpy as np
import cv2
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge


def main():
    rospy.init_node('kitti_dyn_player')
    img_dir = rospy.get_param('~img_dir', '')
    vel_dir = rospy.get_param('~velodyne_dir', '')
    rate = rospy.get_param('~rate', 10.0)
    img_w, img_h = rospy.get_param('~img_size', [1242, 375])  # 标定原生尺寸

    files_img = sorted(f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg')))
    files_vel = sorted(f for f in os.listdir(vel_dir) if f.endswith('.bin'))
    n = min(len(files_img), len(files_vel))
    rospy.loginfo('kitti_dyn_player: 同步播放 %d 帧 @ %.1f Hz (图像 %d, 点云 %d)',
                  n, rate, len(files_img), len(files_vel))

    pub_img = rospy.Publisher('/kitti/camera_color_left/image_raw', Image, queue_size=2)
    pub_pc = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=2)
    bridge = CvBridge()
    fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
              pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
              pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
              pc2.PointField('intensity', 12, pc2.PointField.FLOAT32, 1)]

    # 仿真时间戳: 帧号 x 0.1s(与 KITTI 原始 10Hz 对齐), 与播放耗时无关。
    # 否则 Python 播放器每帧 175ms 会把时间轴拉长 1.75 倍, evo 按时间配对 GT 全错位。
    t0_sim = rospy.Time.now()
    dt = 1.0 / rate

    r = rospy.Rate(rate)
    for i in range(n):
        if rospy.is_shutdown():
            break
        img = cv2.imread(os.path.join(img_dir, files_img[i]))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (img_w, img_h):
            # 下载图被统一缩放(1226x370), 还原标定原生尺寸使 P2 投影有效
            img = cv2.resize(img, (img_w, img_h))
        pts = np.fromfile(os.path.join(vel_dir, files_vel[i]), np.float32).reshape(-1, 4)

        stamp = rospy.Time.from_sec(t0_sim.to_sec() + i * dt)
        m_img = bridge.cv2_to_imgmsg(img, 'bgr8')
        m_img.header.stamp = stamp
        m_img.header.frame_id = 'camera_color_left'
        m_pc = pc2.create_cloud(rospy.Header(stamp=stamp, frame_id='velodyne'), fields, pts)
        pub_img.publish(m_img)
        pub_pc.publish(m_pc)
        r.sleep()

    rospy.loginfo('kitti_dyn_player: 播放完成')


if __name__ == '__main__':
    main()
