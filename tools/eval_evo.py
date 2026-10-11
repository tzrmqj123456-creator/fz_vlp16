#!/usr/bin/env python3
"""LeGO-LOAM 轨迹评估 —— 全部基于官方 evo 库 (v1.31+)
用法: python3 eval_evo.py <est.tum> <gt.tum> <输出目录>
流程: evo 读取 TUM -> 官方时间同步(associate_trajectories) -> 官方 umeyama 对齐(traj.align)
      -> 官方 APE 指标(metrics.APE) -> 官方绘图(evo.tools.plot)
五轴(X/Y/Z/XY/XYZ)分解: 对官方对齐后的轨迹位置逐轴切片, 统计公式与 evo 相同。
输出: eval_report.txt(指标表) / eval_errors.csv(逐位姿误差) / plots/*.png(五轴轨迹对比图)
自校验: 脚本输出的 3D APE 应与 evo_ape CLI 的输出完全一致(同库同参数)。
"""
import sys
import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface, plot

STAT_KEYS = ['rmse', 'max', 'min', 'mean', 'median', 'std']


def evo_stats(err_abs):
    """与 evo 相同公式的统计(rmse/mean/median/std/min/max)"""
    e = np.asarray(err_abs)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()),
                min=float(e.min()), mean=float(e.mean()), median=float(np.median(e)),
                std=float(e.std()))


def main():
    est_file, gt_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

    # 1. evo 官方读取
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    traj_gt = file_interface.read_tum_trajectory_file(gt_file)

    # 2. evo 官方时间同步
    traj_gt_s, traj_est_s = sync.associate_trajectories(traj_gt, traj_est, max_diff=0.03)
    n = traj_est_s.num_poses

    # 3. evo 官方 umeyama 对齐(SE3, 无缩放): align() 返回 (R, t, s), 再用 transform() 应用
    r_a, t_a, _ = traj_est_s.align(traj_gt_s, correct_scale=False)
    T_align = np.eye(4)
    T_align[:3, :3] = r_a
    T_align[:3, 3] = t_a
    traj_est_a = copy.deepcopy(traj_est_s)
    traj_est_a.transform(T_align)

    # 4. evo 官方 APE(3D 全平移)
    ape = metrics.APE(metrics.PoseRelation.full_transformation)
    ape.process_data((traj_gt_s, traj_est_a))
    stats3d = ape.get_all_statistics()

    # 5. 五轴分解(官方对齐轨迹的位置切片)
    p_ref = traj_gt_s.positions_xyz
    p_est = traj_est_a.positions_xyz
    exyz = p_est - p_ref
    errs = {'X': np.abs(exyz[:, 0]), 'Y': np.abs(exyz[:, 1]), 'Z': np.abs(exyz[:, 2]),
            'XY': np.linalg.norm(exyz[:, :2], axis=1),
            'XYZ': np.linalg.norm(exyz, axis=1)}

    # 6. 报告
    lines = [
        'LeGO-LOAM KITTI 轨迹评估报告 (全部基于官方 evo v1.31.1)',
        f'est={os.path.basename(est_file)} ({traj_est.num_poses} 位姿), '
        f'gt={os.path.basename(gt_file)} ({traj_gt.num_poses} 位姿)',
        f'同步: evo associate_trajectories(max_diff=0.03) -> {n} 对;  对齐: evo umeyama(SE3, 无缩放)',
        '坐标系: KITTI 相机系 (x=右, y=下, z=前/行驶方向)',
        '',
        '[官方 evo APE - 3D 全平移]',
        '  rmse    mean    median  std     min     max',
        '  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % tuple(stats3d[k] for k in
                                                      ['rmse', 'mean', 'median', 'std', 'min', 'max']),
        '',
        '[五轴分解 - 官方对齐轨迹逐轴切片]',
        '轴     RMSE(m)    最大(m)    最小(m)    平均(m)    中位(m)    标准差(m)',
    ]
    for k in ['X', 'Y', 'Z', 'XY', 'XYZ']:
        s = evo_stats(errs[k])
        lines.append(f'{k:<6} {s["rmse"]:8.3f}  {s["max"]:8.3f}  {s["min"]:8.3f}  '
                     f'{s["mean"]:8.3f}  {s["median"]:8.3f}  {s["std"]:8.3f}')
    with open(os.path.join(out_dir, 'eval_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # 7. 逐位姿误差 CSV
    t = traj_gt_s.timestamps - traj_gt_s.timestamps[0]
    with open(os.path.join(out_dir, 'eval_errors.csv'), 'w') as f:
        f.write('t,err_X,err_Y,err_Z,err_XY,err_XYZ\n')
        for i in range(n):
            f.write(f'{t[i]:.6f},{errs["X"][i]:.6f},{errs["Y"][i]:.6f},'
                    f'{errs["Z"][i]:.6f},{errs["XY"][i]:.6f},{errs["XYZ"][i]:.6f}\n')

    # 8. 官方 evo 绘图: 五轴轨迹对比图
    pd_ = os.path.join(out_dir, 'plots')
    style_gt = dict(style='--', color='gray', label='ground truth')
    style_es = dict(style='-', color='tab:blue', label='estimated')

    fig = plt.figure(figsize=(8, 8))
    ax = plot.prepare_axis(fig, plot.PlotMode.xyz)
    plot.traj(ax, plot.PlotMode.xyz, traj_gt_s, **style_gt)
    plot.traj(ax, plot.PlotMode.xyz, traj_est_a, **style_es)
    fig.savefig(os.path.join(pd_, 'traj_xyz.png')); plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    ax = plot.prepare_axis(fig, plot.PlotMode.xy)
    plot.traj(ax, plot.PlotMode.xy, traj_gt_s, **style_gt)
    plot.traj(ax, plot.PlotMode.xy, traj_est_a, **style_es)
    fig.savefig(os.path.join(pd_, 'traj_xy.png')); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, name in enumerate(['X (right)', 'Y (down)', 'Z (forward)']):
        axs[i].plot(t, p_ref[:, i], '--', color='gray', label='GT')
        axs[i].plot(t, p_est[:, i], '-', color='tab:blue', label='est')
        axs[i].set_ylabel(name + ' [m]'); axs[i].legend(); axs[i].grid(alpha=0.3)
    axs[2].set_xlabel('t [s]')
    fig.savefig(os.path.join(pd_, 'traj_axes_time.png')); plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plot.error_array(fig.gca(), ape.error, x_array=t,
                     statistics={k: stats3d[k] for k in ['mean', 'median', 'rmse', 'std']},
                     name='APE', title='APE w.r.t. full transformation')
    fig.savefig(os.path.join(pd_, 'ape_error.png')); plt.close(fig)

    print('\n'.join(lines[5:]))
    print(f'\n报告: {out_dir}/eval_report.txt   误差序列: {out_dir}/eval_errors.csv')
    print(f'图: {pd_}/ (traj_xyz.png, traj_xy.png, traj_axes_time.png, ape_error.png)')
    print('自校验: 上方 [官方 evo APE] 的 rmse 应与 evo_ape CLI 输出完全一致')


if __name__ == '__main__':
    main()
