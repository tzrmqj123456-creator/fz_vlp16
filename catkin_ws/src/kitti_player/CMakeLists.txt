cmake_minimum_required(VERSION 3.0.2)
project(kitti_player)

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs pcl_conversions pcl_ros)
find_package(PCL REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

add_executable(kitti_player src/kitti_player.cpp)
target_link_libraries(kitti_player ${catkin_LIBRARIES} ${PCL_LIBRARIES})
