# lego_loam_dyn — YOLOv8-seg 视觉辅助动态点滤除

在 kitti_player 与 LeGO-LOAM 之间插入滤除节点：相机实例掩膜（车辆/行人等）投影到雷达点云，
剔除动态点，输出静态点云供建图。LeGO-LOAM 源码零改动（话题重映射完成接入）。

## 管线

```
kitti_player(/velodyne_points) ──┐
kitti_image_player(图像) → dyn_filter(YOLOv8-seg 掩膜+投影滤除) → /points_dyn_filtered
                                    └──> imageProjection / mapOptmization (remap 后订阅)
```

## 环境准备（一次性）

```bash
# 1. GPU 推理环境 (RTX 3050 4GB: 用 n/s 版模型即可, m/l/x 会爆显存)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python
python3 -c "import torch; print(torch.cuda.is_available())"   # 应输出 True

# 2. KITTI 05 图像: odometry 包不含图像, 需下载 raw 包
#    seq 05 对应 2011_09_30_drive_0018_sync (用 KITTI raw_data_downloader.sh)
#    解压后图像目录例如: dataset/raw_05/image_02/data/*.png (共 2761 张, 10Hz)

# 3. 编译注册本包 (已软链到 catkin_ws/src/lego_loam_dyn)
cd ~/fz_vlp16/catkin_ws && source devel/setup.bash && catkin_make
```

## 运行（KITTI 05, 动态滤除版）

```bash
# 终端1
roslaunch lego_loam_dyn dyn_run.launch img_dir:=/home/trq/fz_vlp16/dataset/raw_05/image_02/data
# 终端2 (与往常相同)
bash ~/fz_vlp16/tools/kitti_play.sh 05
# 跑完 Ctrl+C 终端1, 等 pcd 保存, 评估
bash ~/fz_vlp16/tools/run_eval.sh 05
```

## 调参（dyn_filter 节点参数）

| 参数 | 默认 | 说明 |
|---|---|---|
| model | yolov8s-seg.pt | 模型文件; 4GB 显存可用 yolov8n-seg 更快 |
| conf_thres | 0.35 | 检测置信度阈值 |
| dilate_px | 3 | 掩膜膨胀像素(标定误差/截断补偿) |
| pub_debug | true | 发布掩膜可视化图 /dyn_filter/mask_debug |

动态类别白名单(COCO): person, bicycle, car, motorcycle, bus, train, truck —— 静态类别一律不删。

## 评估

- **点云层**: 对照 SemanticKITTI moving 标注算 Dynamic Recall/Precision/Static Retention
- **SLAM 层**: 与基线对比 ATE(run_eval.sh) + RPE(evo_rpe); 基线: main 分支原版 4.65 / sc-lego-loam 分支 4.83
- **注意**: KITTI 05 动态物较少, SLAM 指标可能不敏感; 点云层指标 + 残影点数是更可靠的滤除效果度量

## 待办（下一步）

- [ ] 掩膜内点云 3D 聚类精修(只删与目标同深度的簇, 提升 Static Retention)
- [ ] 点云读取 frombuffer 快速路径已做; 帧率不足时再用多线程/半精度
- [ ] 几何残差二次确认(分割漏检的截断/远处车辆)
- [ ] 与 kitti_player 严格时间同步(当前 ApproximateTime 0.08s 近似同步, 两路独立 10Hz 回放)
