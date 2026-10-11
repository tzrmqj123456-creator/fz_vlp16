// 读取 KITTI odometry velodyne(.bin), 以 10Hz 发布 /velodyne_points
// 用法: rosrun kitti_player kitti_player _data_path:=~/kitti/dataset _sequence:=05
// VLP-16 仿真: 追加 _vlp16:=true, 按 VLP-16 的 16 线(俯仰±15°,2°间隔) x 1800 列(方位0.2°)重采样
//   (KITTI 数据俯仰最高 +2°, 故模拟的 16 线中高于 +2° 的线为空, 比真实 VLP-16 略保守)
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <XmlRpcValue.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_player");
    ros::NodeHandle nh("~");

    std::string dataPath;
    bool vlp16;
    nh.param<std::string>("data_path", dataPath, "");
    nh.param<bool>("vlp16", vlp16, false);

    // ROS 会把 _sequence:=05 推断成整数 5(前导零被吃掉), 而 param<std::string> 对整数类型会
    // 静默失败回退默认值; 这里用 XmlRpc 直接读原始值, 兼容 int/string, 并归一化为两位数字符串
    std::string sequence = "00";
    {
        XmlRpc::XmlRpcValue seqVal;
        if (nh.getParam("sequence", seqVal))
        {
            int seqNum = 0;
            if (seqVal.getType() == XmlRpc::XmlRpcValue::TypeInt)
                seqNum = (int)seqVal;
            else
                seqNum = std::atoi(std::string(seqVal).c_str());
            char seqBuf[8];
            std::snprintf(seqBuf, sizeof(seqBuf), "%02d", seqNum);
            sequence = seqBuf;
        }
    }

    std::string veloDir = dataPath + "/sequences/" + sequence + "/velodyne/";
    ROS_INFO("kitti_player: data_dir=%s (vlp16=%s)", veloDir.c_str(), vlp16 ? "true" : "false");
    ros::Publisher pubCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);

    ros::Rate rate(10);  // KITTI 10Hz
    int frameID = 0, missCount = 0;
    while (ros::ok())
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s%06d.bin", veloDir.c_str(), frameID);
        std::ifstream in(buf, std::ios::binary);
        if (!in.good())
        {
            // KITTI 序列有缺帧(如 seq 00 缺 0436~0439), 跳过而不是退出; 连续缺 20 帧视为播完
            if (++missCount > 20) break;
            ++frameID;
            rate.sleep();
            continue;
        }
        missCount = 0;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        float f[4];
        pcl::PointXYZI p;
        while (in.read(reinterpret_cast<char*>(f), sizeof(f)))
        {
            p.x = f[0]; p.y = f[1]; p.z = f[2]; p.intensity = f[3];
            cloud->push_back(p);
        }
        in.close();

        if (vlp16)
        {
            // VLP-16: 16 线(俯仰 +15°~-15°, 2°间隔), 10Hz 方位分辨率 0.2° -> 1800 列
            const int N_VLP = 16, H_VLP = 1800;
            const double resY = 2.0, resX = 0.2;
            std::vector<float> bestRange(N_VLP * H_VLP, -1.0f);
            std::vector<float> bestInten(N_VLP * H_VLP, 0.0f);

            for (const pcl::PointXYZI& p : cloud->points)
            {
                float r = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
                if (r < 1e-3) continue;
                float el = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y)) * 180.0f / M_PI;
                int beam = (int)std::lround((el + 15.0f) / resY);     // 最接近的 VLP-16 线号
                if (beam < 0 || beam > N_VLP - 1) continue;           // 超出 ±15° 视场或高于 +2° 数据上限
                float az = std::atan2(p.y, p.x) * 180.0f / M_PI;      // 方位角(度) [-180,180)
                int col = (int)std::lround(az / resX) + H_VLP / 2;
                if (col < 0) col += H_VLP;
                if (col >= H_VLP) col -= H_VLP;
                int idx = beam * H_VLP + col;
                if (bestRange[idx] < 0.0f || r < bestRange[idx])      // 同格取最近点(模拟首个回波)
                {
                    bestRange[idx] = r;
                    bestInten[idx] = p.intensity;
                }
            }

            cloud->clear();
            for (int beam = 0; beam < N_VLP; ++beam)
            {
                float elRad = (-15.0f + resY * beam) * M_PI / 180.0f;
                float ce = std::cos(elRad), se = std::sin(elRad);
                for (int col = 0; col < H_VLP; ++col)
                {
                    float r = bestRange[beam * H_VLP + col];
                    if (r < 0.0f) continue;
                    float azRad = (col - H_VLP / 2) * resX * M_PI / 180.0f;
                    pcl::PointXYZI p;
                    p.x = r * ce * std::cos(azRad);
                    p.y = r * ce * std::sin(azRad);
                    p.z = r * se;
                    p.intensity = bestInten[beam * H_VLP + col];
                    cloud->push_back(p);
                }
            }
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = "velodyne";
        msg.header.stamp = ros::Time::now();
        pubCloud.publish(msg);

        ROS_INFO_ONCE("kitti_player: seq %s, frame %d (%zu points)", sequence.c_str(), frameID, cloud->size());
        ++frameID;
        rate.sleep();
    }
    ROS_INFO("kitti_player: sequence finished (%d frames).", frameID);
    return 0;
}
