#  load packages
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
import torch
import gc
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import matplotlib.pyplot as plt
from train_current_best_age_agg import BrainADHDModel 
from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each, color_map_DK, plot_original, plot_DK_map
from helper_func_prep import SOI_array_per_left,SOI_array_per_right, min_max_normalize, plot_SOI, robust_scale_normalize, log_transform_and_scale
import seaborn as sns
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, gaussian_filter
import shutil
import time
from pathlib import Path
import json
torch.set_num_threads(1)
"""

brain data
  > 3d-to-2d projection (512*512 images, robust scale, smoothed) @/home/jouyang/brain-3d-to-2d-projection.py

"""
# 
def check_disk_space(required_space, directory):
    """检查指定目录是否有足够的磁盘空间"""
    try:
        stats = shutil.disk_usage(directory)
        available_space = stats.free
        if available_space < required_space:
            return False, f"需要 {required_space/(1024**3):.2f}GB, 但只有 {available_space/(1024**3):.2f}GB 可用"
        return True, "enough space"
    except Exception as e:
        return False, str(e)

def estimate_batch_size(image_shape, num_features=4, safety_factor=2):
    """估计单个批次所需的内存大小（以字节为单位）"""
    single_subject_size = np.prod(image_shape) * num_features * 4  # 4 bytes per float32
    return single_subject_size * safety_factor

def normalize_coordinates(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    normalized = coordinates / r[:, np.newaxis]
    return normalized

# cheap sanity check, making test set with north pole not at top
def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """欧拉角（度）→ 3x3 旋转矩阵，按 X→Y→Z 依次旋转"""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])
    return (Rz @ Ry @ Rx)  # 注意顺序


def random_so3_uniform() -> np.ndarray:
    """
    在 SO(3) 上均匀采样一个随机旋转矩阵（用于随机大角度旋转的 sanity check）。
    """
    # 使用Shoemake方法
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)   * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)   * np.cos(2*np.pi*u3)
    # 四元数 -> 旋转矩阵
    qx, qy, qz, qw = q1, q2, q3, q4
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R


def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
    if features.ndim == 3:
        features = features.reshape(-1, 4)
    
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError(f"features numbers ({features.shape[0]}) does not match with coordinates numbers ({coordinates.shape[0]}).")

    valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
                 ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    coordinates = coordinates[valid_mask]
    features = features[valid_mask]

    normalized_coords = normalize_coordinates(coordinates)
    x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    
    u = phi
    v = np.log(np.tan(theta/2 + np.pi/4))
    
    # Handle potential NaN or inf values in v
    v = np.nan_to_num(v, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    
    max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))
    v = np.clip(v, -max_v, max_v)
    
    valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
    u = u[valid_uv_mask]
    v = v[valid_uv_mask]
    features = features[valid_uv_mask]
    
    u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
    v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)
    bins = [u_bins, v_bins]

    image = np.zeros((*image_size, 4))
    
    for i in range(4):
        feature = features[:, i]
        result = binned_statistic_2d(u, v, feature, 
                                     statistic='mean', 
                                     bins=bins)
        projection = result.statistic
        
        image[:, :, i] = projection.T
    
    image = np.nan_to_num(image)
    
    return image

def improved_fill_gaps(image, max_distance=10):
    mask = np.isnan(image)
    filled_image = np.copy(image)

    for c in range(image.shape[2]):
        channel = image[:, :, c]
        channel_mask = mask[:, :, c]
        
        if not np.any(channel_mask):
            continue  # Skip if there are no gaps to fill
        
        dist = distance_transform_edt(channel_mask)
        
        weights = np.exp(-dist / max_distance)
        weights[dist > max_distance] = 0
        
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        
        filled = np.sum(channel[:, :, np.newaxis] * weights, axis=(0, 1))
        
        filled_image[channel_mask, c] = filled

    return filled_image

def smooth_image(image, kernel_size=3):
    smoothed = np.copy(image)
    for c in range(image.shape[2]):
        smoothed[:, :, c] = gaussian_filter(image[:, :, c], sigma=kernel_size/2)
    return smoothed
@torch.no_grad()
def evaluate_streaming(model, idx_list, fetch_image_fn, phenos_all, mean, std, device='cuda', batch_size=4):
    """
    流式评估：每次取少量样本，规范化->前向->累积，不把整套 (N,4,H,W) 放进内存。
    fetch_image_fn(i) 需要返回一个 np.ndarray，形状 (4,512,512)，为第 i 个样本的图像（baseline 或 on-the-fly 旋转后的）。
    """
    # 安全 std
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1e-6

    # 收集器
    preds_all = []
    y_all = []

    # 小批次缓存
    xb_list, eb_list = [], []

    def flush_batch():
        if not xb_list:
            return None
        xb = torch.from_numpy(np.stack(xb_list, axis=0)).to(device)  # (B,4,512,512)
        eb = torch.from_numpy(np.stack(eb_list, axis=0)).to(device)  # (B,3)
        out = model(xb, eb)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        preds_all.append(out)
        xb_list.clear(); eb_list.clear()

    # 遍历索引
    for i in idx_list:
        # 取一张图（baseline 或 旋转）
        x = fetch_image_fn(i)  # (4,512,512), np.float32
        # 规范化（就地不扩内存）
        x = ((x - mean[:,None,None]) / (std_safe[:,None,None])).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 取标签/额外变量
        p = phenos_all[i]
        extra = np.array([p[1], p[3], p[4]], dtype=np.float32)    # [agg, sex, edu]
        target = np.array([p[0], p[2]], dtype=np.float32)         # [att, age]

        xb_list.append(x)
        eb_list.append(extra)
        y_all.append(target)

        if len(xb_list) >= batch_size:
            flush_batch()

    flush_batch()
    preds = np.concatenate(preds_all, axis=0)
    targets = np.stack(y_all, axis=0)

    # 评估
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {}
    for j, name in enumerate(['sum_att','age']):
        y_true = np.nan_to_num(targets[:,j], nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(preds[:,j],   nan=0.0, posinf=0.0, neginf=0.0)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        metrics[name] = {'MSE':mse,'MAE':mae,'R^2':r2}
    return metrics, preds, targets


def evaluate_memmap_stream(model, X_memmap, idxs, phenos_all, mean, std, device='cuda', batch_size=8):
    # 预处理
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1e-6

    # 目标拆分
    phenos = phenos_all[idxs]
    extra = np.stack([phenos[:,1], phenos[:,3], phenos[:,4]], axis=1).astype(np.float32)
    targets = np.stack([phenos[:,0], phenos[:,2]], axis=1).astype(np.float32)

    # 为 R2 的在线计算做准备：SST=∑(y-ȳ)^2, SSE=∑(y-ŷ)^2
    y_mean = targets.mean(axis=0)           # (2,)
    sst = np.sum((targets - y_mean)**2, axis=0).astype(np.float64)  # (2,)
    sse = np.zeros(2, dtype=np.float64)
    mae_sum = np.zeros(2, dtype=np.float64)
    n = len(idxs)

    Xb = np.empty((batch_size, 4, 512, 512), dtype=np.float32)
    Eb = np.empty((batch_size, 3), dtype=np.float32)
    Yb = np.empty((batch_size, 2), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            bsz = e - s
            # 小批量从 memmap 读
            for k, i in enumerate(idxs[s:e]):
                Xb[k] = X_memmap[i]
                Eb[k] = extra[s+k]
                Yb[k] = targets[s+k]

            # 归一化（就地/小批做）
            X_norm = (Xb[:bsz] - mean[:, None, None]) / (std_safe[:, None, None])
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            xb = torch.from_numpy(X_norm).to(device)
            eb = torch.from_numpy(Eb[:bsz]).to(device)
            pred = model(xb, eb).cpu().numpy()    # (bsz,2)
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            # 在线累计
            diff = (Yb[:bsz] - pred).astype(np.float64)
            sse += np.sum(diff**2, axis=0)
            mae_sum += np.sum(np.abs(diff), axis=0)

    # 输出指标
    mse = (sse / n).astype(float)
    mae = (mae_sum / n).astype(float)
    r2 = 1.0 - (sse / np.maximum(sst, 1e-12))
    return {
        'sum_att': {'MSE': float(mse[0]), 'MAE': float(mae[0]), 'R^2': float(r2[0])},
        'age':     {'MSE': float(mse[1]), 'MAE': float(mae[1]), 'R^2': float(r2[1])},
    }

def evaluate_rotated_stream(model, subjects_info, coords, ID_per_half, idxs, R,
                            phenos_all, mean, std, device='cuda', batch_size=4, left_flag=False):
    std_safe = mean.copy()
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1e-6

    phenos = phenos_all[idxs]
    extra = np.stack([phenos[:,1], phenos[:,3], phenos[:,4]], axis=1).astype(np.float32)
    targets = np.stack([phenos[:,0], phenos[:,2]], axis=1).astype(np.float32)

    y_mean = targets.mean(axis=0)
    sst = np.sum((targets - y_mean)**2, axis=0).astype(np.float64)
    sse = np.zeros(2, dtype=np.float64)
    mae_sum = np.zeros(2, dtype=np.float64)
    n = len(idxs)

    Xb = np.empty((batch_size, 4, 512, 512), dtype=np.float32)
    Eb = np.empty((batch_size, 3), dtype=np.float32)
    Yb = np.empty((batch_size, 2), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            bsz = e - s
            # 逐个 subject 生成旋转后的 2D（不落盘）
            for k, i in enumerate(idxs[s:e]):
                img = process_subject(subjects_info[i], coords, ID_per_half, left=left_flag, R=R)  # (4,512,512)
                Xb[k] = img
                Eb[k] = extra[s+k]
                Yb[k] = targets[s+k]

            X_norm = (Xb[:bsz] - mean[:, None, None]) / (std_safe[:, None, None])
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            xb = torch.from_numpy(X_norm).to(device)
            eb = torch.from_numpy(Eb[:bsz]).to(device)
            pred = model(xb, eb).cpu().numpy()
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            diff = (Yb[:bsz] - pred).astype(np.float64)
            sse += np.sum(diff**2, axis=0)
            mae_sum += np.sum(np.abs(diff), axis=0)

    mse = (sse / n).astype(float)
    mae = (mae_sum / n).astype(float)
    r2 = 1.0 - (sse / np.maximum(sst, 1e-12))
    return {
        'sum_att': {'MSE': float(mse[0]), 'MAE': float(mae[0]), 'R^2': float(r2[0])},
        'age':     {'MSE': float(mse[1]), 'MAE': float(mae[1]), 'R^2': float(r2[1])},
    }


def process_subject(subject_info, coordinates, ID_per_half, left=True, R: np.ndarray=None):
    feature_files = [
        f"{'lh' if left else 'rh'}.thickness.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.volume.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.area.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.w_g.pct.mgh.fwhm10.fsaverage.mgh"
    ]
    
    features = []
    for feature_file in feature_files:
        file_path = subject_info[2][feature_file]
        # feature_data = SOI_array_per_right(ID_per_half, file_path) #SOI_array_per_left if left = True, no worries, you used the correct function for lh.
        feature_data = (SOI_array_per_left if left else SOI_array_per_right)(ID_per_half, file_path)
        feature_norm = robust_scale_normalize(feature_data)
        features.append(feature_norm)
        
    SOI_mx_minmax = np.stack(features, axis=-1) # (N_vertices, 4)
    #坐标旋转（右乘 R^T 等价于左乘 R)
    # 在 process_subject 一进函数就加：
    print("[dbg] R is None?", (R is None))
    print("[dbg] coords before rot:", coordinates[:3])      # 前3个点

    coords = coordinates.copy()
    if R is not None:
        coords = coords @ R.T    # (N_vertices, 3)   
        print("[dbg] coords after  rot:", coords[:3]) 

    image = optimized_mercator_projection(coords, SOI_mx_minmax)
    filled_image = improved_fill_gaps(image)
    smoothed_image = smooth_image(filled_image)
    
    # Clear some memory
    del image, filled_image, features, SOI_mx_minmax
    gc.collect()
    
    return smoothed_image.transpose(2, 0, 1)  # (C,H,W)

def safe_memmap_create(path: str, shape, dtype=np.float32):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"[refuse to overwrite] {path} already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)

def make_rotated_test_images(subjects_info, coordinates, ID_per_half, test_idx, R, left_flag=False,
                             save_path=None, mmap=True):
    n_test = len(test_idx); C,H,W = 4,512,512
    if save_path and mmap:
        arr = safe_memmap_create(save_path, shape=(n_test, C, H, W), dtype=np.float32)
    else:
        arr = np.zeros((n_test, C, H, W), dtype=np.float32)


    for k,i in enumerate(test_idx):
        img = process_subject(subjects_info[i], coordinates, ID_per_half, left=left_flag, R=R)
        arr[k] = img
        if (k+1)%50==0: print(f"[rotate] {k+1}/{n_test}")

    if save_path and mmap:
        del arr
        return save_path
    if save_path:
        np.save(save_path, arr)
        return save_path+'.npy'
    return arr

def split_extra_and_targets(phenos_test):
    # Dataset 里 targets = [att_score, age] = idx 0,2
    # extra = [agg, sex, edu] = idx 1,3,4
    targets = np.stack([phenos_test[:,0], phenos_test[:,2]], axis=1).astype(np.float32)
    extra   = np.stack([phenos_test[:,1], phenos_test[:,3], phenos_test[:,4]], axis=1).astype(np.float32)
    return extra, targets

@torch.no_grad()
def evaluate_numpy(model, X_np, phenos_test, mean, std, device='cuda', batch_size=8):
    # 1) 防止 std 过小导致除零/爆大
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1e-6

    # 2) 归一化 + 清洗 NaN/Inf
    X = ((X_np - mean[:,None,None]) / (std_safe[:,None,None])).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) phenotypes 拆分，并确保是有限数
    extra, targets = split_extra_and_targets(phenos_test)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) 转 tensor 前再检查一次
    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        print(f"[warn] X has non-finite values at indices: {bad[0][:5]} (showing first few)")

    X_t      = torch.from_numpy(X).to(device)
    extra_t  = torch.from_numpy(extra).to(device)

    preds = []
    for i in range(0, len(X), batch_size):
        xb = X_t[i:i+batch_size]
        eb = extra_t[i:i+batch_size]
        out = model(xb, eb)            # (B,2)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)  # 5) 输出兜底
        preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 6) 评估前严查
    if not np.isfinite(preds).all():
        n_nans = np.isnan(preds).sum()
        n_infs = np.isinf(preds).sum()
        print(f"[error] preds has NaN/Inf: NaN={n_nans}, Inf={n_infs}. "
              f"示例值: {preds[~np.isfinite(preds)][:10]}")
        # 可选：直接 return 一个空结果，避免 sklearn 抛错
        # return {"sum_att":{}, "age":{}}, preds, targets

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {}
    for j, name in enumerate(['sum_att','age']):
        vt = np.var(targets[:, j])
        vp = np.var(preds[:, j])
        print(f"[var] {name}: var(y_true)={vt:.6g}, var(y_pred)={vp:.6g}, "
            f"finite_pred?={np.isfinite(preds[:, j]).all()} unique_y={np.unique(targets[:, j]).size}")
        
        y_true = targets[:,j]
        y_pred = preds[:,j]
        # 再保底
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        metrics[name] = {'MSE':mse,'MAE':mae,'R^2':r2}
    return metrics, preds, targets



def main():
    print("Easy Sanity Check...")
    print("---For Left Brain---")
    # ===== 0) 固定路径（训练产物只读）=====
    #exp_dir = "/home/jouyang/training_all_data_rh_mirror_attmap_20250205_041248"
    exp_dir = "/home/jouyang/training_all_data_lh_mirror_attmap_20250205_031917"
    exp_logs = os.path.join(exp_dir, "logs")
    exp_models = os.path.join(exp_dir, "models")
    best_model_path = os.path.join(exp_models, "all_lh_best_model.pth")

    # ===== 1) 新建评估目录（只写到这里，避免覆盖训练目录）=====
    tag = time.strftime("sanity_eval_%Y%m%d_%H%M%S")
    eval_dir = os.path.join("/home/jouyang", tag)
    eval_logs = os.path.join(eval_dir, "logs")
    os.makedirs(eval_logs, exist_ok=True)

    # ===== 2) 数据路径 =====
    X_all_path = "/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy"
    phenos_path = "/home/jouyang/all_normalised_phenotypes_correct.npy"  # 确认的正确版本
    subj_info_path = "/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy"
    coords_label_path = "/projects/0/einf1049/scratch/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label"
    id_per_half_path = "ij_id_lh.npy"  # 工作目录里

    # ===== 3) 载入影像与表型 =====
    print(f"[info] eval_dir = {eval_dir}")
    X_all = np.load(X_all_path, mmap_mode='r')
    phenos_all = np.load(phenos_path, mmap_mode='r')
    if len(X_all) != len(phenos_all):
        print(f"[FATAL] X_all({len(X_all)}) vs phenos_all({len(phenos_all)}) 不一致！请检查：{X_all_path} 与 {phenos_path}")
        return


    # ===== 4) 尝试复用训练时的 split 与统计量（只读）；失败则在 eval_dir 重建 =====
    def _try_load_training_stats():
        try:
            tr = np.load(os.path.join(exp_logs, "train_idx.npy"))
            va = np.load(os.path.join(exp_logs, "val_idx.npy"))
            te = np.load(os.path.join(exp_logs, "test_idx.npy"))
            mean = np.load(os.path.join(exp_logs, "mean.npy"))
            std  = np.load(os.path.join(exp_logs, "std.npy"))
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                print("[warn] 训练目录 mean/std 含有 NaN/Inf，放弃使用。")
                return None
            return {"train":tr, "val":va, "test":te, "mean":mean, "std":std}
        except Exception as e:
            print(f"[warn] 读取训练 split/统计失败：{e}")
            return None

    pack = _try_load_training_stats()
    if pack is None:
        print("[info] 无法复用训练统计，按训练脚本方式在 eval_dir 重建一次（不写训练目录）…")
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X_all))
        train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)
        # 只用训练集计算 mean/std
        mean = np.mean(X_all[train_idx], axis=(0,2,3)).astype(np.float32)
        std  = np.std (X_all[train_idx], axis=(0,2,3)).astype(np.float32)
        # 防止除以零
        std[np.isnan(std) | (std < 1e-6)] = 1e-6

        np.save(os.path.join(eval_logs, "train_idx.npy"), train_idx)
        np.save(os.path.join(eval_logs, "val_idx.npy"),   val_idx)
        np.save(os.path.join(eval_logs, "test_idx.npy"),  test_idx)
        np.save(os.path.join(eval_logs, "mean.npy"), mean)
        np.save(os.path.join(eval_logs, "std.npy"),  std)

        pack = {"train":train_idx, "val":val_idx, "test":test_idx, "mean":mean, "std":std}
        print("[info] 新的 split/统计已保存到评估目录 logs/ 下。")
    else:
        print(f"[info] 复用 split 与统计量 from: {exp_logs}")

    train_idx, val_idx, test_idx = pack["train"], pack["val"], pack["test"]
    mean, std = pack["mean"], pack["std"]

    # 打印检查，避免 test 太小导致 R²=nan
    print(f"[info] split sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    # 检查 y 的方差（R² 需要 y_true 有方差）
    def _check_target_variance(name, y):
        v = float(np.var(y))
        print(f"[check] var({name}) = {v:.6g} (unique={np.unique(y).size})")
    phenos_test = phenos_all[test_idx]
    _check_target_variance("sum_att", phenos_test[:,0])
    _check_target_variance("age",     phenos_test[:,2])

    # ===== 5) 读 3D 投影输入（坐标与索引）=====
    subjects_info = np.load(subj_info_path, allow_pickle=True)
    # 读取右半球球坐标
    with open(coords_label_path, "r") as rh:
        data = rh.read().splitlines()[2:]
    data_arr_rh = np.array([list(map(float, line.split())) for line in data])
    coordinates_right = np.column_stack((data_arr_rh[:,1], data_arr_rh[:,2], data_arr_rh[:,3]))
    ID_per_half_right = np.load(id_per_half_path, allow_pickle=True).astype(int)

    def make_fetch_rot(R):
        def _fetch(i):
            # 只生成“第 i 个”的旋转投影图像并返回 (4,512,512)
            return process_subject(subjects_info[i], coordinates_right, ID_per_half_right, left=False, R=R)
        return _fetch
    
    # ===== 6) 生成几组旋转版 test 图像（保存到 eval_dir）=====
    R_list = [
        rotation_matrix_xyz(90,0,0),
        rotation_matrix_xyz(0,90,0),
        rotation_matrix_xyz(0,0,180),
    ]
    rot_paths = []
    for ridx, R in enumerate(R_list):
        save_path = os.path.join(eval_dir, f"rot_lh_test_R{ridx}.npy")
        print(f"[info] 生成旋转视角 R{ridx} -> {save_path}")
        make_rotated_test_images(
            subjects_info, coordinates_right, ID_per_half_right,
            test_idx, R, left_flag=True, save_path=save_path, mmap=True
        ) ##left_flag=False，right brain 
        #FIXME:here need to change when need to take it to left brain.
        rot_paths.append(save_path)

    # ===== 7) 载入 best model（只读）=====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainADHDModel(num_phenotypes=3).to(device)
    print(f"[info] loading best model: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ===== 8) baseline 评估（未旋转）=====
    def fetch_baseline(i):
    # 单样本按需取出，不会构造 (N,...) 大数组
        return X_all[i]  # shape (4,512,512)

    base_metrics, base_preds, base_targets = evaluate_streaming(
        model, test_idx, fetch_baseline, phenos_all, mean, std, device=device, batch_size=4
    )
    print("Baseline:", base_metrics)


    # ===== 9) 依次评估旋转版 =====
    all_metrics = {"baseline": base_metrics}
    for ridx, R in enumerate(R_list):
        print(f"[info] streaming Rot R{ridx} …")
        # m_rot = evaluate_rotated_stream(
        #     model, subjects_info, coordinates_right, ID_per_half_right,
        #     test_idx, R, phenos_all, mean, std, device=device, batch_size=4, left_flag=False
        # )
        m_rot, _, _ = evaluate_streaming(
            model, test_idx, make_fetch_rot(R),
            phenos_all, mean, std, device=device, batch_size=2  # 旋转更费，batch 更小
        )
        all_metrics[f"rot_R{ridx}"] = m_rot
        print(f"Rot R{ridx}:", m_rot)

    # ===== 10) 保存结果（CSV + JSON）到 eval_dir =====
    import csv
    csv_path = os.path.join(eval_dir, "sanity_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting","target","MSE","MAE","R2"])
        for setting, d in all_metrics.items():
            for target, md in d.items():
                w.writerow([setting, target, md.get("MSE"), md.get("MAE"), md.get("R^2")])

    with open(os.path.join(eval_dir, "sanity_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 也把这次评估使用到的 split/统计拷贝一份到 eval_dir，便于以后复现
    np.save(os.path.join(eval_logs, "test_idx_used.npy"), test_idx)
    np.save(os.path.join(eval_logs, "mean_used.npy"), mean)
    np.save(os.path.join(eval_logs, "std_used.npy"),  std)

    print(f"[done] CSV saved -> {csv_path}")
    print(f"[done] eval logs -> {eval_logs}")

# 
if __name__ == "__main__":
    main()


