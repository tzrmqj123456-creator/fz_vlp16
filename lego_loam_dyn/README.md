# lego_loam_dyn — YOLOv8-seg 视觉辅助动态点滤除

在数据播放与 LeGO-LOAM 之间插入滤除节点：相机实例掩膜（车辆/行人等）投影到雷达点云，
剔除动态点，输出静态点云供建图。LeGO-LOAM 源码零改动（话题重映射完成接入）。

## 管线

```
kitti_dyn_player(图像+点云 同时间戳 10Hz) ──> dyn_filter(YOLOv8-seg 掩膜+投影滤除)
                                                    └──> /points_dyn_filtered
                                                          └──> imageProjection / mapOptmization (remap)
```

## 环境（已就绪）

- torch 2.4.1+cu121 / ultralytics 8.4 / CUDA 可用（RTX 3050 4GB，s-seg 稳态 ~10ms/帧）
- 图像: `~/fz_vlp16/05_image/image_2/*.png` (2761 张, 播放时还原为标定原生 1242x375)
- 标定: `~/fz_vlp16/05_image/calib.txt` (P2/R0_rect/Tr_velo_to_cam)
- 模型: `lego_loam_dyn/models/yolov8s-seg.pt`（已下载）

## 运行（KITTI 05, 动态滤除版）

```bash
# 终端1
roslaunch lego_loam_dyn dyn_run.launch
# 终端2 (与往常相同: 记录轨迹 + 同步播放图像和点云)
bash ~/fz_vlp16/tools/kitti_play_dyn.sh 05
# 跑完: 终端1 Ctrl+C, 等 pcd 保存, 评估
bash ~/fz_vlp16/tools/run_eval.sh 05
```

## 调参（dyn_filter 节点参数, 见 dyn_run.launch）

| 参数 | 默认 | 说明 |
|---|---|---|
| model | models/yolov8s-seg.pt | 4GB 显存可用 yolov8n-seg 更快(7ms) |
| conf_thres | 0.35 | 检测置信度阈值 |
| dilate_px | 3 | 掩膜膨胀像素(标定误差/截断补偿) |
| pub_debug | true | 掩膜可视化图 /dyn_filter/mask_debug |

动态类别白名单(COCO): person, bicycle, car, motorcycle, bus, train, truck —— 静态类别一律不删。

## 评估

- **点云层**: 对照 SemanticKITTI moving 标注算 Dynamic Recall/Precision/Static Retention
- **SLAM 层**: 与基线对比 ATE(run_eval.sh) + RPE(evo_rpe)
  基线: main 分支原版 4.65 / sc-lego-loam 分支 SC 版 4.83 (XYZ rmse)
- **注意**: KITTI 05 动态物较少, SLAM 指标可能不敏感; 点云层指标 + 残影点数是更可靠的度量

## 待办（下一步）

- [ ] 掩膜内点云 3D 聚类精修(只删与目标同深度的簇, 提升 Static Retention)
- [ ] 几何残差二次确认(分割漏检的截断/远处车辆)
- [ ] 帧率不足时: 半精度/线程化推理
