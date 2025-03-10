import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_func_prep import SOI_array_per_left, SOI_array_per_right, robust_scale_normalize, robust_scale_normalize_display

def plot_feature_distributions(subject_info, ID_per_half_left, ID_per_half_right, output_path=None):
    """
    Plot distributions of brain features before and after robust scaling with color distinction
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    feature_names = ['Thickness', 'Volume', 'Surface Area', 'W/G Ratio']
    
    # 更改颜色方案
    raw_color = 'skyblue'
    norm_color = 'mediumpurple'  # 使用紫色系
    
    lh_files = [
        "lh.thickness.fwhm10.fsaverage.mgh",
        "lh.volume.fwhm10.fsaverage.mgh",
        "lh.area.fwhm10.fsaverage.mgh",
        "lh.w_g.pct.mgh.fwhm10.fsaverage.mgh"
    ]
    
    rh_files = [
        "rh.thickness.fwhm10.fsaverage.mgh",
        "rh.volume.fwhm10.fsaverage.mgh",
        "rh.area.fwhm10.fsaverage.mgh",
        "rh.w_g.pct.mgh.fwhm10.fsaverage.mgh"
    ]

    for col, (lh_file, rh_file) in enumerate(zip(lh_files, rh_files)):
        lh_path = subject_info[2][lh_file]
        rh_path = subject_info[2][rh_file]
        
        # 获取数据
        lh_data = SOI_array_per_left(ID_per_half_left, lh_path)
        rh_data = SOI_array_per_right(ID_per_half_right, rh_path)
        
        # 标准化数据
        lh_data_norm = robust_scale_normalize_display(lh_data, is_right_brain=False)
        rh_data_norm = robust_scale_normalize_display(rh_data, is_right_brain=True)
        
        # 重新排列绘图顺序：左脑原始 -> 左脑标准化 -> 右脑原始 -> 右脑标准化
        # 左脑原始
        sns.histplot(lh_data.flatten(), ax=axes[0, col], kde=True, 
                    color=raw_color, alpha=0.6)
        axes[0, col].set_title(f'Left Brain Raw\n{feature_names[col]}')
        
        # 左脑标准化
        sns.histplot(lh_data_norm.flatten(), ax=axes[1, col], kde=True, 
                    color=norm_color, alpha=0.6)
        axes[1, col].set_title(f'Left Brain Normalized\n{feature_names[col]}')
        
        # 右脑原始
        sns.histplot(rh_data.flatten(), ax=axes[2, col], kde=True, 
                    color=raw_color, alpha=0.6)
        axes[2, col].set_title(f'Right Brain Raw\n{feature_names[col]}')
        
        # 右脑标准化
        sns.histplot(rh_data_norm.flatten(), ax=axes[3, col], kde=True, 
                    color=norm_color, alpha=0.6)
        axes[3, col].set_title(f'Right Brain Normalized\n{feature_names[col]}')

    plt.tight_layout()
    
    # 添加总标题
    # fig.suptitle('Cortical Features Distribution Before and After Robust Scaling Normalization', 
                # y=1.02, fontsize=16)
    
    # 创建图例并放在右上角
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=raw_color, lw=4),
                   Line2D([0], [0], color=norm_color, lw=4)]
    fig.legend(custom_lines, ['Raw Data', 'Normalized Data'], 
              loc='upper right',  # 改为右上角
              bbox_to_anchor=(0.07, 1.0))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例:
subjects_info = np.load('/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy', 
                       allow_pickle=True)

# 选择第一个subject的数据进行展示
subject_info = subjects_info[0]

# 加载vertex IDs
ID_per_half_left = np.load('ij_id_lh.npy', allow_pickle=True).astype(int)
ID_per_half_right = np.load('ij_id_rh.npy', allow_pickle=True).astype(int)

# 生成并保存图
plot_feature_distributions(
    subject_info,
    ID_per_half_left,
    ID_per_half_right,
    output_path='brain_features_distribution.png'
)