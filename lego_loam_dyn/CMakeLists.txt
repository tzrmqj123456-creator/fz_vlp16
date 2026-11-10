cmake_minimum_required(VERSION 3.0.2)
project(lego_loam_dyn)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_filters
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/dyn_filter.py
  scripts/kitti_image_player.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
