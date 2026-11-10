#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI raw sync 图像播放器: 与 kitti_player 的点云(10Hz)同步播放 left color 图像
用法: rosrun lego_loam_dyn kitti_image_player.py <图像目录>
  或: rosrun lego_loam_dyn kitti_image_player.py _img_dir:=/path/to/image_02/data
话题: /kitti/camera_color_left/image_raw (sensor_msgs/Image, 10Hz)
"""
import os
import sys
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    rospy.init_node('kitti_image_player', anonymous=True)
    img_dir = rospy.get_param('~img_dir', sys.argv[1] if len(sys.argv) > 1 else '')
    rate = rospy.get_param('~rate', 10.0)

    if not img_dir or not os.path.isdir(img_dir):
        rospy.logerr('图像目录不存在: %s', img_dir)
        sys.exit(1)
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg')))
    rospy.loginfo('kitti_image_player: %d 张图像 @ %.1f Hz, 目录 %s', len(files), rate, img_dir)

    pub = rospy.Publisher('/kitti/camera_color_left/image_raw', Image, queue_size=2)
    bridge = CvBridge()
    r = rospy.Rate(rate)
    for f in files:
        if rospy.is_shutdown():
            break
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            rospy.logwarn('读取失败: %s', f)
            continue
        msg = bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'camera_color_left'
        pub.publish(msg)
        r.sleep()
    rospy.loginfo('kitti_image_player: 播放完成')


if __name__ == '__main__':
    main()
