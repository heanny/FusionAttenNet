import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic_2d
import nibabel as nib
from pathlib import Path
from matplotlib.patches import Patch
from scipy.interpolate import griddata

class RegionalAttentionAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """设置必要的目录结构"""
        self.plots_dir = Path('vis_clau_ablation_study_rh_last')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in ['attention', 'regions']:
            (self.plots_dir / subdir).mkdir(exist_ok=True)
    
    def analyze_regional_attention_with_atlas(self, baseline_maps, no_feature_maps, data_arr_lh):
        """使用DK Atlas分析不同脑区的attention分布"""
        
        # 1. 读取和准备DK Atlas数据
        annot_path = '/projects/0/einf1049/scratch/jouyang/rh.aparc.annot'
        labels, ctab, names = nib.freesurfer.read_annot(annot_path)
        ij_id_lh = np.load('ij_id_rh.npy', allow_pickle=True)
        ij_id_lh = ij_id_lh.astype(int)
        labels = labels[ij_id_lh]
        label_names = {i: name.decode('utf-8') for i, name in enumerate(names)}
        for label_id, name in label_names.items():
            print(f"Label {label_id}: {name}")
        # 2. 准备attention maps
        # 打印shape以便调试
        # print("Original baseline_maps shape:", baseline_maps.shape)
        # print("Original no_feature_maps shape:", no_feature_maps.shape)
        # 2. 重塑attention maps到正确的维度
        # 首先重塑为 (1260, 512, 512, 2)
        baseline_reshaped = baseline_maps.reshape(1260, 512, 512, 2)
        no_feature_reshaped = no_feature_maps.reshape(1260, 512, 512, 2)
        
        # 只取第一个通道 (假设这是我们需要的attention)
        baseline_attention = baseline_reshaped[:, :, :, 0]  # shape: (1260, 512, 512)
        no_feature_attention = no_feature_reshaped[:, :, :, 0]  # shape: (1260, 512, 512)
    
        # 确保attention maps是正确的shape (N, 16, 16) 或 (N, 4, 16, 16)
        # if len(baseline_maps.shape) == 2:
        #     baseline_maps = baseline_maps.reshape(-1, 16, 16)
        #     no_feature_maps = no_feature_maps.reshape(-1, 16, 16)
        # elif len(baseline_maps.shape) == 1:
        #     baseline_maps = baseline_maps.reshape(-1, 4, 16, 16)
        #     no_feature_maps = no_feature_maps.reshape(-1, 4, 16, 16)
        
        # print("Reshaped baseline_maps shape:", baseline_maps.shape)
        
        # 3. 准备坐标和投影
        x_left = data_arr_lh[:, 1]
        y_left = data_arr_lh[:, 2]
        z_left = data_arr_lh[:, 3]
        coordinates_left = np.column_stack((x_left, y_left, z_left))
        
        # 创建label特征
        label_features = np.zeros((len(labels), 4))
        label_features[:, 0] = labels
        
        # 投影labels
        projected_labels = optimized_mercator_projection(coordinates_left, label_features)
        label_map = projected_labels[:, :, 0]
        # 使用函数
        # label_map = projected_labels[:, :, 0]  # 假设这是你的标签投影数据
        create_brain_atlas_visualization(label_map, label_names)
        
        # print("Label map shape:", label_map.shape)
        
            # 4. 初始化结果字典
        results = {
            'FusionAttenNet': {},
            'No-Feature-Attention': {}
        }
        
        # 5. 计算每个区域的attention
        for label_id, region_name in label_names.items():
            if region_name == 'unknown':  # 跳过背景
                continue
                
            # 创建区域掩码
            mask = (label_map == label_id)
            
            if not np.any(mask):
                print(f"Warning: No pixels found for region: {region_name}")
                continue
            
            # 对每个样本计算该区域的attention
            baseline_region = baseline_attention[:, mask]  # shape: (1260, N_pixels)
            no_feature_region = no_feature_attention[:, mask]
            
            # 计算统计量
            results['FusionAttenNet'][region_name] = {
                'mean': np.mean(baseline_region),
                'std': np.std(baseline_region),
                'max': np.max(baseline_region),
                'min': np.min(baseline_region)
            }
            
            results['No-Feature-Attention'][region_name] = {
                'mean': np.mean(no_feature_region),
                'std': np.std(no_feature_region),
                'max': np.max(no_feature_region),
                'min': np.min(no_feature_region)
            }
        
        # 6. 绘制结果
        self._plot_regional_analysis(results)

            # 在返回结果前打印top regions
        print("\nTop 10 regions with highest attention (FusionAttenNet):")
        sorted_regions = sorted(results['FusionAttenNet'].items(), 
                            key=lambda x: x[1]['mean'], 
                            reverse=True)
        for i, (region, stats) in enumerate(sorted_regions[:10], 1):
            print(f"{i}. {region}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 打印统计信息
        print(f"\nTotal number of regions processed: {len(results['FusionAttenNet'])}")
        print("\nRegions without any pixels in the projected space:")
        missing_regions = set(label_names.values()) - set(results['FusionAttenNet'].keys()) - {'unknown'}
        for region in missing_regions:
            print(f"- {region}")
        
        return results
    
    def _plot_regional_analysis(self, results):
        """绘制区域分析结果"""
        # 准备数据
        regions = list(results['FusionAttenNet'].keys())
        baseline_means = [results['FusionAttenNet'][r]['mean'] for r in regions]
        baseline_stds = [results['FusionAttenNet'][r]['std'] for r in regions]
        no_feature_means = [results['No-Feature-Attention'][r]['mean'] for r in regions]
        no_feature_stds = [results['No-Feature-Attention'][r]['std'] for r in regions]
        
        # 排序
        sorted_indices = np.argsort(baseline_means)[::-1][:15]
        regions = [regions[i] for i in sorted_indices]
        baseline_means = [baseline_means[i] for i in sorted_indices]
        baseline_stds = [baseline_stds[i] for i in sorted_indices]
        no_feature_means = [no_feature_means[i] for i in sorted_indices]
        no_feature_stds = [no_feature_stds[i] for i in sorted_indices]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 8))
        x = np.arange(len(regions))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, baseline_means, width,
                        yerr=baseline_stds,
                        label='FusionAttenNet',
                        color='#4169E1',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        rects2 = ax.bar(x + width/2, no_feature_means, width,
                        yerr=no_feature_stds,
                        label='No-Feature-Attention',
                        color='#40C8E3',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        # 添加数值标注
        def autolabel(rects, values):
            for rect, val in zip(rects, values):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom')
        
        autolabel(rects1, baseline_means)
        autolabel(rects2, no_feature_means)
        
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title('Regional Attention Analysis (Spatial Attention Module, Right-semi) - Top 15 Regions')
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'regional_attention_analysis.png',
                    bbox_inches='tight')
        plt.close()

def normalize_coordinates(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    normalized = coordinates / r[:, np.newaxis]
    return normalized

# def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
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

# def analyze_regional_attention_with_atlas(self, baseline_maps, no_feature_maps, data_arr_lh):
    """使用DK Atlas分析不同脑区的attention分布"""
    
    # 1. 读取和准备DK Atlas数据
    annot_path = 'rh.aparc.annot'
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    ij_id_lh = np.load('ij_id_rh.npy', allow_pickle=True)
    ij_id_lh = ij_id_lh.astype(int)
    labels = labels[ij_id_lh]
    label_names = {i: name.decode('utf-8') for i, name in enumerate(names)}

    # 2. 将labels投影到16x16网格
    # 使用与attention map相同的投影方法
    x_left = data_arr_lh[:, 1]
    y_left = data_arr_lh[:, 2]
    z_left = data_arr_lh[:, 3]
    coordinates_left = np.column_stack((x_left, y_left, z_left))
    
    # 创建一个虚拟特征矩阵，用labels填充
    label_features = np.zeros((len(labels), 4))
    label_features[:, 0] = labels  # 我们只需要用第一个通道
    
    # 投影labels到16x16网格
    projected_labels = optimized_mercator_projection(coordinates_left, label_features)
    label_map = projected_labels[:, :, 0]  # 取第一个通道作为label map
    
    # 3. 计算每个区域的平均attention
    results = {
        'FusionAttenNet': {},
        'No-Feature-Attention': {}
    }
    
    # Reshape attention maps
    baseline_reshaped = baseline_maps.reshape(-1, 4, 16, 16)  # (N, 4, 16, 16)
    no_feature_reshaped = no_feature_maps.reshape(-1, 4, 16, 16)
    
    features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
    
    # 为每个特征计算区域attention
    for feature_idx, feature_name in enumerate(features):
        region_stats_baseline = {}
        region_stats_no_feature = {}
        
        for label_id, region_name in label_names.items():
            if label_id == 0:  # 跳过背景
                continue
                
            # 创建区域掩码
            mask = (label_map == label_id)
            
            if not np.any(mask):  # 如果该区域在16x16网格中没有像素
                continue
            
            # 计算该区域的attention统计量
            baseline_attention = baseline_reshaped[:, feature_idx][mask]
            no_feature_attention = no_feature_reshaped[:, feature_idx][mask]
            
            # 存储统计信息
            region_stats_baseline[region_name] = {
                'mean': np.mean(baseline_attention),
                'std': np.std(baseline_attention),
                'max': np.max(baseline_attention),
                'min': np.min(baseline_attention)
            }
            
            region_stats_no_feature[region_name] = {
                'mean': np.mean(no_feature_attention),
                'std': np.std(no_feature_attention),
                'max': np.max(no_feature_attention),
                'min': np.min(no_feature_attention)
            }
        
        results['FusionAttenNet'][feature_name] = region_stats_baseline
        results['No-Feature-Attention'][feature_name] = region_stats_no_feature
        
        # 绘制该特征的区域分析图
        self._plot_feature_regional_analysis(
            feature_name,
            region_stats_baseline,
            region_stats_no_feature
        )
    
    return results

# def _plot_feature_regional_analysis(self, feature_name, baseline_stats, no_feature_stats):
    """为单个特征绘制区域分析图"""
    # 按平均attention值排序
    regions = sorted(baseline_stats.keys(), 
                    key=lambda x: baseline_stats[x]['mean'],
                    reverse=True)
    
    baseline_means = [baseline_stats[r]['mean'] for r in regions]
    baseline_stds = [baseline_stats[r]['std'] for r in regions]
    no_feature_means = [no_feature_stats[r]['mean'] for r in regions]
    no_feature_stds = [no_feature_stats[r]['std'] for r in regions]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(regions))
    width = 0.35
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, baseline_means, width,
                    yerr=baseline_stds,
                    label='FusionAttenNet',
                    color='#4169E1',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
    
    rects2 = ax.bar(x + width/2, no_feature_means, width,
                    yerr=no_feature_stds,
                    label='No-Feature-Attention',
                    color='#F08080',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
    
    # 添加标注
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom')
    
    autolabel(rects1, baseline_means)
    autolabel(rects2, no_feature_means)
    
    # 设置图形属性
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title(f'Regional Attention Analysis - {feature_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(self.plots_dir / 'attention' / f'regional_attention_{feature_name}.png',
                bbox_inches='tight')
    plt.close()

# def analyze_regional_attention_with_atlas(self, baseline_maps, no_feature_maps, data_arr_lh):
    """使用DK Atlas分析不同脑区的attention分布"""
    
    # 1. 读取和准备DK Atlas数据
    annot_path = '/projects/0/einf1049/scratch/jouyang/rh.aparc.annot'
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    ij_id_lh = np.load('ij_id_rh.npy', allow_pickle=True)
    ij_id_lh = ij_id_lh.astype(int)
    labels = labels[ij_id_lh]
    label_names = {i: name.decode('utf-8') for i, name in enumerate(names)}
    for label_id, name in label_names.items():
        print(f"Label {label_id}: {name}")
    # 打印原始标签信息
    print("\n=== 原始标签统计 ===")
    for label_id, region_name in label_names.items():
        count = np.sum(labels == label_id)
        print(f"{region_name}: {count} vertices")
    
    # 2. 准备坐标数据
    x_left = data_arr_lh[:, 1]
    y_left = data_arr_lh[:, 2]
    z_left = data_arr_lh[:, 3]
    coordinates_left = np.column_stack((x_left, y_left, z_left))
    
    # 打印坐标范围
    print("\n=== 坐标范围 ===")
    print(f"X range: {x_left.min():.2f} to {x_left.max():.2f}")
    print(f"Y range: {y_left.min():.2f} to {y_left.max():.2f}")
    print(f"Z range: {z_left.min():.2f} to {z_left.max():.2f}")
    
    # 3. 投影前的数据验证
    print("\n=== 投影前数据验证 ===")
    print(f"总顶点数: {len(coordinates_left)}")
    print(f"有效坐标数: {np.sum(~np.isnan(coordinates_left).any(axis=1))}")
    
    # 4. 执行投影
    label_features = np.zeros((len(labels), 4))
    label_features[:, 0] = labels
    projected_labels = optimized_mercator_projection(coordinates_left, label_features)
    label_map = projected_labels[:, :, 0]
    
    # 5. 投影后的验证
    print("\n=== 投影后数据验证 ===")
    unique_labels = np.unique(label_map)
    print(f"投影后的唯一标签数: {len(unique_labels)}")
    
    # 检查每个区域的像素覆盖
    print("\n=== 投影后区域像素统计 ===")
    for label_id, region_name in label_names.items():
        if label_id == 0:  # 跳过背景
            continue
        pixel_count = np.sum(label_map == label_id)
        original_count = np.sum(labels == label_id)
        if pixel_count == 0:
            print(f"警告: {region_name} 没有像素")
            print(f"  - 原始顶点数: {original_count}")
        else:
            print(f"{region_name}: {pixel_count} pixels (原始顶点数: {original_count})")
    
    # 继续原有的分析...
    results = {
        'FusionAttenNet': {},
        'No-Feature-Attention': {}
    }

    return results
def custom_mode(x):
    """计算众数，如果数组为空返回nan"""
    if len(x) == 0:
        return np.nan
    # 对数值进行四舍五入以处理浮点数误差
    rounded = np.round(x)
    # 找出出现最多的值
    values, counts = np.unique(rounded, return_counts=True)
    return values[np.argmax(counts)]

def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
    if features.ndim == 3:
        features = features.reshape(-1, 4)
    
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError(f"features numbers ({features.shape[0]}) does not match with coordinates numbers ({coordinates.shape[0]}).")
    # 初始验证和打印
    print("\n原始数据统计：")
    unique_labels = np.unique(features[:, 0])
    for label in unique_labels:
        count = np.sum(features[:, 0] == label)
        print(f"Label {label}: {count} vertices")

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
    
    max_v = np.log(np.tan(np.pi/4 + 0.99*(np.pi/4)))
    v = np.clip(v, -max_v, max_v)
    
    valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
    u = u[valid_uv_mask]
    v = v[valid_uv_mask]
    features = features[valid_uv_mask]
    
    image_size = (1024, 1024) 
    u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
    v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)
    bins = [u_bins, v_bins]

    image = np.zeros((*image_size, 4))
    
    # 处理标签通道（第一个通道）
    result = binned_statistic_2d(u, v, features[:, 0], 
                               statistic=lambda x: custom_mode(x) if len(x) > 0 else np.nan,  # 使用mode而不是mean
                               bins=bins)
    projection = result.statistic

    # 验证投影后的标签
    print("\n投影后标签统计：")
    unique_projected = np.unique(projection[~np.isnan(projection)])
    for label in unique_projected:
        count = np.sum(projection == label)
        print(f"Label {label}: {count} pixels")

    image[:, :, 0] = projection.T
    
    # 处理其他通道（保持不变）
    for i in range(1, 4):
        feature = features[:, i]
        result = binned_statistic_2d(u, v, feature, 
                                   statistic='mean', 
                                   bins=bins)
        projection = result.statistic
        image[:, :, i] = projection.T
    
    image = np.nan_to_num(image)
    
    # 将图像大小调整回原始尺寸
    if image_size != (512, 512):
        from scipy.ndimage import zoom
        zoom_factor = (512/image_size[0], 512/image_size[1], 1)
        image = zoom(image, zoom_factor, order=0)  # order=0 保持标签值的离散性

    
    return image


def adjust_color(base_color, factor):
    """调整颜色的亮度"""
    rgb = mcolors.to_rgb(base_color)
    # 调整亮度但保持在有效范围内
    adjusted = [min(max(c * factor, 0), 1) for c in rgb]
    return adjusted

def hex_to_rgb(hex_color):
    """将十六进制颜色代码转换为RGB值"""
    return mcolors.to_rgb(hex_color)

def create_brain_atlas_visualization(label_map, label_names):
    """创建更清晰的脑区图谱可视化"""
    
    # 定义更饱和的颜色
    base_colors = {
        'frontal': '#FF0000',     # 鲜红色
        'parietal': '#00A0A0',    # 深青色
        'temporal': '#0000FF',    # 纯蓝色
        'occipital': '#00FF00',   # 鲜绿色
        'cingulate': '#FFD700',   # 金黄色
        'other': '#808080'        # 深灰色
    }
    
    # 定义区域分类
    region_types = {
        'superiorfrontal': 'frontal',
        'rostralmiddlefrontal': 'frontal',
        'caudalmiddlefrontal': 'frontal',
        'parsopercularis': 'frontal',
        'parstriangularis': 'frontal',
        'parsorbitalis': 'frontal',
        'lateralorbitofrontal': 'frontal',
        'medialorbitofrontal': 'frontal',
        'precentral': 'frontal',
        'frontalpole': 'frontal',
        
        'superiorparietal': 'parietal',
        'inferiorparietal': 'parietal',
        'supramarginal': 'parietal',
        'postcentral': 'parietal',
        'precuneus': 'parietal',
        
        'superiortemporal': 'temporal',
        'middletemporal': 'temporal',
        'inferiortemporal': 'temporal',
        'temporalpole': 'temporal',
        'transversetemporal': 'temporal',
        'fusiform': 'temporal',
        
        'lateraloccipital': 'occipital',
        'lingual': 'occipital',
        'cuneus': 'occipital',
        'pericalcarine': 'occipital',
        
        'caudalanteriorcingulate': 'cingulate',
        'rostralanteriorcingulate': 'cingulate',
        'posteriorcingulate': 'cingulate',
        'isthmuscingulate': 'cingulate',
        
        'insula': 'other',
        'parahippocampal': 'other',
        'entorhinal': 'other'
    }
    
    # 创建颜色映射
    label_colors = {
        0: np.array([1, 1, 1])  # 背景为白色
    }
    
    # 为每个标签分配颜色
    for label_id, name in label_names.items():
        if name in region_types:
            base_color = base_colors[region_types[name]]
            rgb_color = np.array(mcolors.to_rgb(base_color))
            # 添加微小变化以区分同一区域的不同部分
            factor = np.random.uniform(0.85, 1.0)
            label_colors[label_id] = rgb_color * factor
    
    # 创建彩色图像
    colored_map = np.ones((*label_map.shape, 3))  # 初始化为白色
    for label_id, color in label_colors.items():
        mask = (label_map == label_id)
        colored_map[mask] = color
    
    # 创建图形
    plt.figure(figsize=(20, 15))
    
    # 设置背景颜色
    plt.gca().set_facecolor('white')
    
    # 绘制主图
    plt.imshow(colored_map, interpolation='nearest')
    plt.axis('off')
    plt.title('2D Right-semi Brain Regions Visualization with DK Atlas', fontsize=16, pad=20)
    
    # 添加清晰的图例
    legend_elements = [Patch(facecolor=base_colors[region],
                           edgecolor='black',
                           label=region.capitalize())
                      for region in base_colors.keys()]
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              fontsize=12,
              frameon=True,
              edgecolor='black')
    
    # 保存高清图像
    plt.savefig('/home/jouyang1/vis_clau_ablation_study_rh_last/attention/brain_atlas_clear.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    return colored_map


def main():
    """主函数运行完整的区域分析"""
    # 1. 设置路径
    base_dir = Path('/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815')  # 替换为你的实际路径
    
    # 2. 加载attention maps
    baseline_maps = np.load(base_dir / 'attention' / 'raw_data' / 'baseline_spatial_attention_maps.npy')
    no_feature_maps = np.load(base_dir / 'attention' / 'raw_data' / 'no_feature_attention_spatial_attention_maps.npy')
    
    # # 使用函数
    # label_map = projected_labels[:, :, 0]  # 假设这是你的标签投影数据
    # create_brain_atlas_visualization(label_map)
    # 3. 加载坐标数据
    head = open("/projects/0/einf1049/scratch/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
    # data is the raw data 
    data = head.read().splitlines()
    # data_truc is the raw data without the header
    data_truc = data[2:]

    data_lines = []
    data_words = []

    for i in range(len(data_truc)):
        data_line = data_truc[i].split()
        data_words = []
        for j in range(len(data_line)):
            data_word = float(data_line[j])
            data_words.append(data_word)
        data_lines.append(np.array(data_words))

    # data_arr is the data array with correct datatype of each coloumn
    data_arr_lh = np.array(data_lines)
    
    # 4. 创建分析器实例
    analyzer = RegionalAttentionAnalyzer(base_dir)
    
     # 5. 运行分析
    try:
        print("开始进行区域分析...")
        results = analyzer.analyze_regional_attention_with_atlas(
            baseline_maps=baseline_maps,
            no_feature_maps=no_feature_maps,
            data_arr_lh=data_arr_lh
        )
        
        # 6. 保存结果
        output_file = Path('vis_clau_ablation_study_rh_last') / 'analysis_results' / 'regional_attention_analysis.npy'
        output_file.parent.mkdir(exist_ok=True)
        np.save(output_file, results)
        
        print("分析完成！结果已保存到:", output_file)
        
        # 7. 打印主要发现
        print("\n主要发现:")
        # 检查结果字典的结构
        if isinstance(results, dict) and 'FusionAttenNet' in results:
            for region_name, stats in results['FusionAttenNet'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"\n{region_name}区域的统计数据:")
                    print(f"  平均attention: {stats['mean']:.3f}")
                    print(f"  标准差: {stats['std']:.3f}")
                    print(f"  最大值: {stats['max']:.3f}")
                    print(f"  最小值: {stats['min']:.3f}")
        else:
            print("结果格式不符合预期，请检查analyze_regional_attention_with_atlas的返回值")
        
    except Exception as e:
        print("分析过程中出现错误:")
        print(str(e))
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()