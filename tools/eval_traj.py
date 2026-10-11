#!/usr/bin/env python3
"""LeGO-LOAM 轨迹评估: 全局轨迹 vs 真值, 按 X/Y/Z/XY/XYZ 五轴分解对比
用法: python3 eval_traj.py <est.tum> <gt.tum> <输出目录>
输出:
  <输出目录>/eval_report.txt   - 各轴 rmse/最大/最小/平均/中位误差
  <输出目录>/eval_errors.csv   - 逐位姿误差序列(时间, X, Y, Z, XY, XYZ)
说明:
  - est 与 gt 先按时间戳配对(容差 0.03s), 再做一次全局 umeyama 对齐(含旋转+平移)
  - 坐标系: KITTI 相机系 (x=右, y=下, z=前/行驶方向)
"""
import sys
import os
import numpy as np


def umeyama_row(X, Y):
    """umeyama 对齐(无缩放), X/Y 为 Nxd 行向量; 返回 R, t 使 Y ≈ X@R.T + t
    注: R = V·D·Uᵀ (与 evo 的 sim3 输出数值校验一致; 教科书的 U·D·Vᵀ 在此约定下是转置)"""
    cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(X.shape[1])
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ D @ U.T
    t = cy - cx @ R.T
    return R, t


def match_pairs(est, gt, tol=0.03):
    """gt -> 最近 est 时间戳配对"""
    m = []
    for j, g in enumerate(gt):
        i = int(np.argmin(np.abs(est[:, 0] - g[0])))
        if abs(est[i, 0] - g[0]) < tol:
            m.append((i, j))
    return m


def metrics(err):
    err = np.abs(np.asarray(err))
    return dict(rmse=np.sqrt(np.mean(err**2)), max=err.max(), min=err.min(),
                mean=err.mean(), median=np.median(err))


def main():
    est_file, gt_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    est = np.loadtxt(est_file)
    gt = np.loadtxt(gt_file)

    pairs = match_pairs(est, gt)
    pe = np.array([est[i, 1:4] for i, _ in pairs])
    pg = np.array([gt[j, 1:4] for _, j in pairs])
    ts = np.array([est[i, 0] for i, _ in pairs])
    print(f'配对 {len(pairs)} 对 (est {len(est)} 位姿, gt {len(gt)} 位姿)')

    R, t = umeyama_row(pe, pg)
    pea = pe @ R.T + t
    exyz = pea - pg

    # 五轴误差序列
    errs = {
        'X':   np.abs(exyz[:, 0]),
        'Y':   np.abs(exyz[:, 1]),
        'Z':   np.abs(exyz[:, 2]),
        'XY':  np.linalg.norm(exyz[:, :2], axis=1),
        'XYZ': np.linalg.norm(exyz, axis=1),
    }

    lines = [
        'LeGO-LOAM 轨迹评估报告 (全局轨迹 vs 真值, umeyama 对齐)',
        f'数据: est={os.path.basename(est_file)} ({len(est)} 位姿), '
        f'gt={os.path.basename(gt_file)} ({len(gt)} 位姿), 时间同步配对 {len(pairs)} 对',
        '坐标系: KITTI 相机系 (x=右, y=下, z=前/行驶方向)',
        '',
        '轴     RMSE(m)    最大(m)    最小(m)    平均(m)    中位(m)',
    ]
    for k in ['X', 'Y', 'Z', 'XY', 'XYZ']:
        m = metrics(errs[k])
        lines.append(f'{k:<6} {m["rmse"]:8.3f}  {m["max"]:8.3f}  {m["min"]:8.3f}  '
                     f'{m["mean"]:8.3f}  {m["median"]:8.3f}')
        print(lines[-1])

    with open(os.path.join(out_dir, 'eval_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    with open(os.path.join(out_dir, 'eval_errors.csv'), 'w') as f:
        f.write('t,err_X,err_Y,err_Z,err_XY,err_XYZ\n')
        for k in range(len(ts)):
            f.write(f'{ts[k]:.6f},{errs["X"][k]:.6f},{errs["Y"][k]:.6f},'
                    f'{errs["Z"][k]:.6f},{errs["XY"][k]:.6f},{errs["XYZ"][k]:.6f}\n')

    print(f'\n报告已写入: {out_dir}/eval_report.txt')
    print(f'逐位姿误差序列: {out_dir}/eval_errors.csv')


if __name__ == '__main__':
    main()
