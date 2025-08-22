# quick_sanity_32.py
import os, time, json, csv, gc, numpy as np, torch
from pathlib import Path

# ==== 你项目里已有的函数 / 类（从你当前脚本复用）====
from train_current_best_age_agg import BrainADHDModel
from helper_func_prep import SOI_array_per_right, robust_scale_normalize
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, gaussian_filter

torch.set_num_threads(1)  # 限制CPU线程，降低内存抖动

# ====== 轻量工具 ======
def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[ np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return (Rz @ Ry @ Rx)

def normalize_coordinates(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    return coordinates / r[:, None]

def optimized_mercator_projection(coordinates, features, image_size=(512,512)):
    if features.ndim == 3: features = features.reshape(-1,4)
    mask = (~np.isnan(coordinates).any(1) & ~np.isinf(coordinates).any(1) &
            ~np.isnan(features).any(1) & ~np.isinf(features).any(1))
    coordinates = coordinates[mask]; features = features[mask]
    xyz = normalize_coordinates(coordinates); x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(np.clip(z/r, -1, 1)); phi = np.arctan2(y,x)
    u = phi; v = np.log(np.tan(theta/2 + np.pi/4))
    v = np.nan_to_num(v, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4))); v = np.clip(v, -max_v, max_v)
    valmask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
    u = u[valmask]; v = v[valmask]; features = features[valmask]
    u_bins = np.linspace(u.min(), u.max(), image_size[0]+1)
    v_bins = np.linspace(v.min(), v.max(), image_size[1]+1)
    image = np.zeros((*image_size,4), dtype=np.float16)
    for i in range(4):
        proj = binned_statistic_2d(u, v, features[:,i], statistic='mean', bins=[u_bins, v_bins]).statistic
        image[:,:,i] = np.nan_to_num(proj.T, nan=0.0)
    return image

def improved_fill_gaps(image, max_distance=10):
    out = image.copy()
    mask = np.isnan(image)
    for c in range(image.shape[2]):
        if not np.any(mask[:,:,c]): continue
        # 简化：用最近邻距离权重近似
        dist = distance_transform_edt(mask[:,:,c])
        w = np.exp(-dist/max_distance); w[dist>max_distance] = 0
        ws = w.sum()
        if ws>0:
            out[:,:,c][mask[:,:,c]] = (image[:,:,c]*w).sum()/ws
    return np.nan_to_num(out)

def smooth_image(image, sigma=1.5):
    sm = image.copy()
    for c in range(image.shape[2]):
        sm[:,:,c] = gaussian_filter(sm[:,:,c], sigma=sigma)
    return sm

def process_subject_one_pass(subject_info, coordinates, ID_per_half, left=False, R=None):
    hemi = 'lh' if left else 'rh'
    feats = []
    for fname in [f"{hemi}.thickness.fwhm10.fsaverage.mgh",
                  f"{hemi}.volume.fwhm10.fsaverage.mgh",
                  f"{hemi}.area.fwhm10.fsaverage.mgh",
                  f"{hemi}.w_g.pct.mgh.fwhm10.fsaverage.mgh"]:
        arr = SOI_array_per_right(ID_per_half, subject_info[2][fname])  # 右半球函数，你之前确认OK
        feats.append(robust_scale_normalize(arr))
    SOI = np.stack(feats, axis=-1)  # (V,4)
    coords = coordinates if R is None else (coordinates @ R.T)
    img = optimized_mercator_projection(coords, SOI)
    img = improved_fill_gaps(img)
    img = smooth_image(img)
    return img.transpose(2,0,1).astype(np.floa16)  # (4,512,512)

@torch.no_grad()
def evaluate_streaming(model, mean, std, phenos, subjects_info, coords, IDidx, indices, device):
    # phenos: (N,5)  -> targets: [:,[0,2]]; extra: [:,[1,3,4]]
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    std_safe = std.copy(); std_safe[std_safe<1e-6] = 1e-6
    y_true = []; y_pred = []
    for i in indices:
        # baseline（不旋转）
        img = process_subject_one_pass(subjects_info[i], coords, IDidx, left=False, R=None)
        x = (img - mean[:,None,None]) / std_safe[:,None,None]
        x = np.nan_to_num(x)
        e = np.array([phenos[i,1], phenos[i,3], phenos[i,4]], dtype=np.float16)[None,:]
        xb = torch.from_numpy(x[None,...]).to(device)
        eb = torch.from_numpy(e).to(device)
        out = model(xb, eb).cpu().numpy()[0]
        y_pred.append(out)
        y_true.append([phenos[i,0], phenos[i,2]])
        # 释放
        del img, xb, eb; gc.collect()
    y_true = np.array(y_true, dtype=np.float16)
    y_pred = np.array(y_pred, dtype=np.float16)
    metrics = {}
    for j, name in enumerate(['sum_att','age']):
        mse = mean_squared_error(y_true[:,j], y_pred[:,j])
        mae = mean_absolute_error(y_true[:,j], y_pred[:,j])
        try:
            r2  = r2_score(y_true[:,j], y_pred[:,j])
        except Exception:
            r2 = float('nan')
        metrics[name] = {'MSE':mse,'MAE':mae,'R^2':r2}
    return metrics, y_pred, y_true

@torch.no_grad()
def evaluate_rotations_streaming(model, mean, std, phenos, subjects_info, coords, IDidx, indices, R_list, device):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    std_safe = std.copy(); std_safe[std_safe<1e-6] = 1e-6
    results = []
    for ridx, R in enumerate(R_list):
        y_true = []; y_pred = []
        for i in indices:
            img = process_subject_one_pass(subjects_info[i], coords, IDidx, left=False, R=R)
            x = (img - mean[:,None,None]) / std_safe[:,None,None]
            x = np.nan_to_num(x)
            e = np.array([phenos[i,1], phenos[i,3], phenos[i,4]], dtype=np.float16)[None,:]
            xb = torch.from_numpy(x[None,...]).to(device)
            eb = torch.from_numpy(e).to(device)
            out = model(xb, eb).cpu().numpy()[0]
            y_pred.append(out); y_true.append([phenos[i,0], phenos[i,2]])
            del img, xb, eb; gc.collect()
        y_true = np.array(y_true, dtype=np.float16)
        y_pred = np.array(y_pred, dtype=np.float16)
        metrics = {}
        for j, name in enumerate(['sum_att','age']):
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            mae = mean_absolute_error(y_true[:,j], y_pred[:,j])
            try:
                r2  = r2_score(y_true[:,j], y_pred[:,j])
            except Exception:
                r2 = float('nan')
            metrics[name] = {'MSE':mse,'MAE':mae,'R^2':r2}
        results.append(metrics)
    return results

def main():
    # ===== 固定只读路径 =====
    exp_dir = "/home/jouyang/training_all_data_rh_mirror_attmap_20250205_041248"
    exp_logs = os.path.join(exp_dir, "logs")
    best_model_path = os.path.join(exp_dir, "models", "all_rh_best_model.pth")
    X_all_path   = "/projects/0/einf1049/scratch/jouyang/all_cnn_rh_brainimages.npy"   # 只用于 shape/meanstd（可读）
    phenos_path  = "/home/jouyang/all_normalised_phenotypes_correct.npy"
    subj_info_path = "/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy"
    coords_label_path = "/projects/0/einf1049/scratch/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label"
    id_per_half_path = "ij_id_rh.npy"

    # ===== 结果目录（新建，不覆盖旧文件）=====
    tag = time.strftime("quick16_%Y%m%d_%H%M%S")
    eval_dir = os.path.join("/home/jouyang", tag)
    os.makedirs(eval_dir, exist_ok=True)
    print("[info] eval_dir:", eval_dir)

    # ===== 载入表型与分割 =====
    phenos = np.load(phenos_path, mmap_mode='r')  # (12595,5)
    # 优先复用训练 split 和 mean/std
    try:
        te = np.load(os.path.join(exp_logs, "test_idx.npy"))
        mean = np.load(os.path.join(exp_logs, "mean.npy"))
        std  = np.load(os.path.join(exp_logs, "std.npy"))
        if (~np.isfinite(mean).all()) or (~np.isfinite(std).all()):
            raise RuntimeError("mean/std contain non-finite")
        print("[info] reuse split/mean/std from training logs.")
    except Exception as e:
        print("[warn] fallback to re-split/compute mean/std:", e)
        X_all = np.load(X_all_path, mmap_mode='r')  # 只读
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(X_all))
        trv, te = train_test_split(idx, test_size=0.1, random_state=42)
        tr, _ = train_test_split(trv, test_size=0.11111, random_state=42)
        mean = np.mean(X_all[tr], axis=(0,2,3)).astype(np.float16)
        std  = np.std (X_all[tr], axis=(0,2,3)).astype(np.float16)
        std[(~np.isfinite(std)) | (std<1e-6)] = 1e-6
        del X_all; gc.collect()

    # 取 test 集的前/随机 16 个索引
    if len(te) < 16:
        sel = te
    else:
        # 固定随机子集，保证可重复
        rng = np.random.default_rng(0)
        sel = rng.choice(te, size=16, replace=False)
    sel = np.sort(sel)
    print(f"[info] sample size: {len(sel)} (from test set)")

    # ===== 读取坐标和索引 =====
    with open(coords_label_path, "r") as rh:
        data = rh.read().splitlines()[2:]
    data_arr = np.array([list(map(float, line.split())) for line in data])
    coords = np.column_stack((data_arr[:,1], data_arr[:,2], data_arr[:,3]))
    IDidx  = np.load(id_per_half_path, allow_pickle=True).astype(int)
    subjects_info = np.load(subj_info_path, allow_pickle=True)

    # ===== 载入模型（只读）=====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainADHDModel(num_phenotypes=3).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ===== baseline（不旋转）=====
    base_metrics, base_pred, base_true = evaluate_streaming(
        model, mean, std, phenos, subjects_info, coords, IDidx, sel, device
    )
    print("[baseline]", base_metrics)

    # ===== 旋转设置：±15°, 45°, 90°（只做 X/Y/Z 三个轴的代表角度，量小但有代表性）=====
    R_list = [
        # rotation_matrix_xyz( 15, 0, 0),
        # rotation_matrix_xyz(-15, 0, 0),
        # rotation_matrix_xyz( 45, 0, 0),
        rotation_matrix_xyz( 90, 0, 0),
        # rotation_matrix_xyz( 0, 45, 0),
        rotation_matrix_xyz( 0, 90, 0),
        # rotation_matrix_xyz( 0, 0, 45),
        rotation_matrix_xyz( 0, 0, 90),
    ]
    rot_metrics_list = evaluate_rotations_streaming(
        model, mean, std, phenos, subjects_info, coords, IDidx, sel, R_list, device
    )
    for ridx, m in enumerate(rot_metrics_list):
        print(f"[rot {ridx}]", m)

    # ===== 保存结果（CSV/JSON）到新的 eval_dir =====
    out_csv = os.path.join(eval_dir, "quick16_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting","target","MSE","MAE","R2"])
        for target, md in base_metrics.items():
            w.writerow(["baseline", target, md["MSE"], md["MAE"], md["R^2"]])
        for ridx, m in enumerate(rot_metrics_list):
            for target, md in m.items():
                w.writerow([f"rot_{ridx}", target, md["MSE"], md["MAE"], md["R^2"]])

    out_json = os.path.join(eval_dir, "quick16_metrics.json")
    with open(out_json, "w") as f:
        json.dump({"baseline":base_metrics,
                   **{f"rot_{i}":rm for i,rm in enumerate(rot_metrics_list)}},
                  f, indent=2)

    # 记录这次用到的样本索引
    np.save(os.path.join(eval_dir, "quick16_indices.npy"), sel)

    print("[done] results ->", out_csv)

if __name__ == "__main__":
    main()
