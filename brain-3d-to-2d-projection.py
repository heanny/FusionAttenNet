"""
Improved new projection method
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from helper_func_prep import SOI_array_per_left, SOI_array_per_right, robust_scale_normalize, brain_SOI_matrix_left, brain_SOI_matrix_right
from scipy.ndimage import distance_transform_edt, gaussian_filter
import os
import torch
import nibabel as nib

def normalize_coordinates(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    normalized = coordinates / r[:, np.newaxis]
    return normalized

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


def save_image(image, filename):
    """
    保存四通道图像并输出调试信息。
    
    :param image: 形状为(height, width, 4)的numpy数组
    :param filename: 输出文件名
    """
    print(f"Image shape: {image.shape}")
    print(f"Image min value: {np.min(image)}")
    print(f"Image max value: {np.max(image)}")
    print(f"Image mean value: {np.mean(image)}")
    
    plt.figure(figsize=(20, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(image[:, :, i], cmap='viridis', vmin=-2, vmax=2)
        plt.title(f'Channel {i}')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '_channels_unsmoothed.png'))
    plt.close()

    # 创建RGB图像
    rgb_image = np.zeros((*image.shape[:2], 3))
    for i in range(3):
        channel = image[:, :, i]
        rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title('RGB Channels')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_subject(subject_info, coordinates, ID_per_half):
    """
    处理单个受试者的数据。
    
    :param subject_info: 包含受试者信息的字典
    :param coordinates: 坐标数据
    :param ID_per_half: 用于SOI_array_per_left的ID数据
    :return: 处理后的图像数据，形状为(4, 512, 512)
    """
    # feature_files = [
    #     'lh.thickness.fwhm10.fsaverage.mgh',
    #     'lh.volume.fwhm10.fsaverage.mgh',
    #     'lh.area.fwhm10.fsaverage.mgh',
    #     'lh.w_g.pct.mgh.fwhm10.fsaverage.mgh'
    # ]
    feature_files = [
        'rh.thickness.fwhm10.fsaverage.mgh',
        'rh.volume.fwhm10.fsaverage.mgh',
        'rh.area.fwhm10.fsaverage.mgh',
        'rh.w_g.pct.mgh.fwhm10.fsaverage.mgh'
    ]
    
    features = []
    for feature_file in feature_files:
        file_path = subject_info[2][feature_file]
        feature_data = SOI_array_per_right(ID_per_half, file_path)
        feature_norm = robust_scale_normalize(feature_data)
        features.append(feature_norm)
    
    SOI_mx_minmax_rh = np.stack(features, axis=-1)
    
    image = optimized_mercator_projection(coordinates, SOI_mx_minmax_rh)
    filename = f"subject_{subject_info[0]}_rh_robust_scale_onlyprojected.png"
    save_image(image, filename)

    filled_image = improved_fill_gaps(image)
    filename = f"subject_{subject_info[0]}_rh_robust_scale_filledgaps.png"
    save_image(filled_image, filename)

    smoothed_image = smooth_image(filled_image)
    filename = f"subject_{subject_info[0]}_rh_robust_scale_smoothed.png"
    save_image(smoothed_image, filename)

    #     image = optimized_mercator_projection(coordinates, features)
#     filename = f"subject_{subject_id}_{hemisphere}_robust_scale_unsmoothed.png"
#     save_image(image, filename)
    
    return smoothed_image.transpose(2, 0, 1)  # 调整为(4, 512, 512)


# 
def main():
    # 加载受试者信息
    subjects_info = np.load('sample_ids_filename_updated.npy', allow_pickle=True)
    
    total_subjects = len(subjects_info)
    print(f"Total subjects to process: {total_subjects}")

    # 加载坐标数据（假设所有受试者使用相同的坐标）
    rh = open("/projects/0/einf1049/scratch/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
    data = rh.read().splitlines()[2:]  # 跳过前两行
    data_arr_rh = np.array([list(map(float, line.split())) for line in data])
    
    x_left = data_arr_rh[:, 1]
    y_left = data_arr_rh[:, 2]
    z_left = data_arr_rh[:, 3]
    coordinates_right = np.column_stack((x_left, y_left, z_left))

    # 加载 ID_per_half
    ID_per_half = np.load('ij_id_rh.npy', allow_pickle=True).astype(int)

    all_subjects_data = []

    for i, subject_info in enumerate(subjects_info):
        try:
            subject_id = subject_info[0]
            subject_data = process_subject(subject_info, coordinates_right, ID_per_half)
            all_subjects_data.append(subject_data)
            
            # 打印进度
            if (i + 1) % 100 == 0 or (i + 1) == total_subjects:
                print(f"Processed {i + 1}/{total_subjects} subjects")
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")

    # 将所有受试者的数据合并为一个大的numpy数组
    all_subjects_array = np.array(all_subjects_data)
    
    # 保存合并后的数据
    output_file = "sample_cnn_rh_brainimages.npy"
    np.save(output_file, all_subjects_array)
    print(f"Saved combined data with shape: {all_subjects_array.shape} to {output_file}")

if __name__ == "__main__":
    main() 
    
    
# %%
# plot the selected subject brain feature images
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_subject_features(subject_data, subject_index):
#     """
#     绘制指定受试者的四个特征图像。

#     :param subject_data: 形状为(4, 512, 512)的numpy数组，包含四个特征的图像数据
#     :param subject_index: 受试者的索引号
#     """
#     feature_names = ['Thickness', 'Volume', 'Surface Area', 'White-Gray Contrast']

#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#     fig.suptitle(f'Features for Subject {subject_index}', fontsize=16)

#     for i, ax in enumerate(axes.flat):
#         im = ax.imshow(subject_data[i], cmap='viridis')
#         ax.set_title(feature_names[i])
#         ax.axis('off')
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.show()

# # 主程序
# def main():
#     # 加载数据
#     data = np.load('sample_brain_4images_all.npy')
#     print(f"Loaded data shape: {data.shape}")

#     # 指定要可视化的受试者索引
#     subject_index =55  # 您可以更改这个值来查看不同的受试者, index is +1 for *index of array*

#     # 确保索引有效
#     if subject_index >= data.shape[0]:
#         print(f"Error: Subject index {subject_index} is out of range. Total subjects: {data.shape[0]}")
#         return

#     # 提取指定受试者的数据
#     subject_data = data[subject_index]

#     # 绘图
#     plot_subject_features(subject_data, subject_index)

# if __name__ == "__main__":
#     main()

# %%
"""
new projection method
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic_2d
# from helper_func_prep import SOI_array_per_left, SOI_array_per_right, min_max_normalize, plot_SOI, robust_scale_normalize, quantile_normalize, log_transform_and_scale
# from scipy.signal import convolve2d

# def normalize_coordinates(coordinates):
#     """
#     将坐标归一化到单位球体上。
    
#     :param coordinates: 形状为(n_points, 3)的numpy数组，包含x, y, z坐标
#     :return: 归一化后的坐标
#     """
#     # 计算径向距离
#     r = np.sqrt(np.sum(coordinates**2, axis=1))
#     # 归一化到单位球面
#     normalized = coordinates / r[:, np.newaxis]
#     return normalized

# def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
#     """
#     使用优化的Mercator投影将3D坐标点和特征数据转换为2D图像。
    
#     :param coordinates: 形状为(n_points, 3)的numpy数组，包含x, y, z坐标
#     :param features: 形状为(n_points, 4)的numpy数组，包含A, B, C, D特征值
#     :param image_size: 输出图像的尺寸，默认为(512, 512)
#     :return: 形状为(512, 512, 4)的numpy数组，表示四通道图像
#     """

#      # 确保features的形状正确
#     if features.ndim == 3:
#         features = features.reshape(-1, 4)
    
#     if features.shape[0] != coordinates.shape[0]:
#         raise ValueError(f"特征数量 ({features.shape[0]}) 与坐标点数量 ({coordinates.shape[0]}) 不匹配")
    
#     # 移除包含NaN的行
#     valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
#                  ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
#     coordinates = coordinates[valid_mask]
#     features = features[valid_mask]

#     # 首先归一化坐标
#     normalized_coords = normalize_coordinates(coordinates)
#     x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
#     # 将笛卡尔坐标转换为球坐标
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(np.clip(z / r, -1, 1))
#     phi = np.arctan2(y, x)
    
#     # 优化的Mercator投影
#     u = phi
#     v = np.log(np.tan(theta/2 + np.pi/4))
    
#     # 处理极点附近的变形
#     max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))  # 限制在极点附近0.95范围内
#     v = np.clip(v, -max_v, max_v)
    
#     # 确保u和v不包含NaN或无穷大
#     valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
#     u = u[valid_uv_mask]
#     v = v[valid_uv_mask]
#     features = features[valid_uv_mask]
    
#     # 定义bins
#     u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
#     v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)
#     bins = [u_bins, v_bins]

#     image = np.zeros((*image_size, 4))
    
#     for i in range(4):
#         feature = features[:, i]
#         result = binned_statistic_2d(u, v, feature, 
#                                      statistic='mean', 
#                                      bins=bins)
#         projection = result.statistic
        
#         image[:, :, i] = projection.T
    
#     # 处理NaN值
#     image = np.nan_to_num(image)
    
#     return image


# def fill_gaps(image):
#     """
#     填充图像中的空白区域。
#     """
#     mask = np.isnan(image)
#     image[mask] = 0
    
#     # 使用简单的邻域平均填充
#     kernel = np.ones((3, 3)) / 9
#     for i in range(image.shape[2]):  # 遍历每个通道
#         channel = image[:, :, i]
#         filled = channel.copy()
#         for _ in range(3):  # 重复几次以填充较大的空白
#             filled = convolve2d(filled, kernel, mode='same', boundary='symm')
#             channel[mask[:, :, i]] = filled[mask[:, :, i]]
#         image[:, :, i] = channel
    
#     return image

# def save_image(image, filename):
#     """
#     保存四通道图像并输出调试信息。
    
#     :param image: 形状为(height, width, 4)的numpy数组
#     :param filename: 输出文件名
#     """
#     print(f"Image shape: {image.shape}")
#     print(f"Image min value: {np.min(image)}")
#     print(f"Image max value: {np.max(image)}")
#     print(f"Image mean value: {np.mean(image)}")
    
#     plt.figure(figsize=(20, 5))
#     for i in range(4):
#         plt.subplot(1, 4, i+1)
#         plt.imshow(image[:, :, i], cmap='viridis', vmin=-2, vmax=2)
#         plt.title(f'Channel {i}')
#         plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(filename.replace('.png', '_channels_unsmoothed.png'))
#     plt.close()

#     # 创建RGB图像
#     rgb_image = np.zeros((*image.shape[:2], 3))
#     for i in range(3):
#         channel = image[:, :, i]
#         rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

#     plt.figure(figsize=(10, 10))
#     plt.imshow(rgb_image)
#     plt.title('RGB Channels')
#     plt.axis('off')
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.close()


# def process_subject(subject_id, hemisphere, coordinates, features):
#     """
#     处理单个受试者的数据。
    
#     :param subject_id: 受试者ID
#     :param hemisphere: 'left' 或 'right'
#     :param coordinates: 坐标数据
#     :param features: 特征数据
#     """
#     image = optimized_mercator_projection(coordinates, features)
#     filename = f"subject_{subject_id}_{hemisphere}_robust_scale_unsmoothed.png"
#     save_image(image, filename)
#     print(f"Saved image for subject {subject_id}, {hemisphere} hemisphere")

# # 主处理流程 （这里有错！！）
# def main():
#     # 这里应该是您读取数据的代码
#     n_subjects = 1  # 实际使用时改为12595
#     n_points_left = 149955
#     n_points_right = 149926
#     subjects_info = np.load('sample_ids_filename.npy', allow_pickle=True)
#     subjects_info = subjects_info[:2]

#     # load xyz 3D coordinates
#     # left brain
#     lh = open("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = lh.read().splitlines()
#     # data_truc is the raw data without the header
#     data_truc = data[2:]

#     data_lines = []
#     data_words = []

#     for i in range(len(data_truc)):
#         data_line = data_truc[i].split()
#         data_words = []
#         for j in range(len(data_line)):
#             data_word = float(data_line[j])
#             data_words.append(data_word)
#         data_lines.append(np.array(data_words))

#     # data_arr is the data array with correct datatype of each coloumn
#     data_arr_lh = np.array(data_lines)

#     ij_id_left = np.load('ij_id_lh.npy', allow_pickle=True)
#     ID_per_left = ij_id_left.astype('int')
#     input_data = []
#     for index, row in subjects_info.iterrows():
#         paths_dict = row['file_paths']

#         # Extract paths based on the filename-parameter mapping
#         thick_path = paths_dict["lh.thickness.fwhm10.fsaverage.mgh"]
#         volume_path = paths_dict["lh.volume.fwhm10.fsaverage.mgh"]
#         SA_path = paths_dict["lh.area.fwhm10.fsaverage.mgh"]
#         w_g_pct_path = paths_dict["lh.w_g.pct.mgh.fwhm10.fsaverage.mgh"]

#         # Call the function
#         input_data.append(brain_SOI_matrix_left(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_left))
#     input_tensor = torch.tensor(input_data, dtype=torch.float32)



#     for subject_id in range(n_subjects):
#         # 处理左半脑
#         x_left = data_arr_lh[:, 1]
#         y_left = data_arr_lh[:, 2]
#         z_left = data_arr_lh[:, 3]
#         coordinates_left = np.column_stack((x_left, y_left, z_left))
#         features_left = SOI_mx_minmax_lh
#         process_subject(subject_id, 'left', coordinates_left, features_left)

#         # # 处理右半脑
#         # x_right = data_arr_rh[:][1]
#         # y_right = data_arr_rh[:][2]
#         # z_right = data_arr_rh[:][3]
#         # coordinates_right = np.column_stack((x_right, y_right, z_right))
#         # features_right = np.random.randn(n_points_right, 4)
#         # process_subject(subject_id, 'right', coordinates_right, features_right)

# # 
# if __name__ == "__main__":
#     main()


"""
plot the brain features with DK atals

"""

# 
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from scipy.stats import binned_statistic_2d
# import nibabel as nib
# from scipy.interpolate import griddata
# from helper_func_prep import SOI_array_per,SOI_array_per_fs5, min_max_normalize, plot_SOI, robust_scale_normalize, quantile_normalize, log_transform_and_scale
# # from scipy.signal import convolve2d

# def normalize_coordinates(coordinates):
#     r = np.sqrt(np.sum(coordinates**2, axis=1))
#     return coordinates / r[:, np.newaxis]

# def optimized_mercator_projection(coordinates, features, dk_labels, image_size=(512, 512)):
#     normalized_coords = normalize_coordinates(coordinates)
#     x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]

#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(np.clip(z / r, -1, 1))
#     phi = np.arctan2(y, x)

#     u = phi
#     v = np.log(np.tan(theta/2 + np.pi/4))

#     max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))
#     v = np.clip(v, -max_v, max_v)

#     valid_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
#     u, v = u[valid_mask], v[valid_mask]
#     features, dk_labels = features[valid_mask], dk_labels[valid_mask]

#     u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
#     v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)
#     bins = [u_bins, v_bins]

#     image = np.zeros((*image_size, 5))  # 5 channels: 4 features + 1 DK atlas

#     for i in range(4):
#         feature = features[:, i]
#         result = binned_statistic_2d(u, v, feature, statistic='mean', bins=bins)
#         image[:, :, i] = result.statistic.T

#     # DK atlas projection
#     dk_result = binned_statistic_2d(u, v, dk_labels, statistic=lambda x: np.nanmean(x), bins=bins)
#     image[:, :, 4] = dk_result.statistic.T

#     return np.nan_to_num(image)
# def create_custom_colormap():
#     # Define colors for each region based on the provided image
#     custom_colors = {
#         'unknown': '#000000',
#         'bankssts': '#A08020',
#         'caudalanteriorcingulate': '#7030A0',
#         'caudalmiddlefrontal': '#008000',
#         'corpuscallosum': '#FFFFFF',
#         'cuneus': '#7F6000',
#         'entorhinal': '#FF0000',
#         'fusiform': '#197F00',
#         'inferiorparietal': '#00FFFF',
#         'inferiortemporal': '#3F7F7F',
#         'isthmuscingulate': '#7F00FF',
#         'lateraloccipital': '#0000FF',
#         'lateralorbitofrontal': '#7F7F00',
#         'lingual': '#BFBF00',
#         'medialorbitofrontal': '#00FF00',
#         'middletemporal': '#00007F',
#         'parahippocampal': '#7F0000',
#         'paracentral': '#7FFF00',
#         'parsopercularis': '#0080FF',
#         'parsorbitalis': '#FF7F00',
#         'parstriangularis': '#7FFFFF',
#         'pericalcarine': '#007F7F',
#         'postcentral': '#FF7FFF',
#         'posteriorcingulate': '#7F007F',
#         'precentral': '#FF00FF',
#         'precuneus': '#FFFF00',
#         'rostralanteriorcingulate': '#00FF7F',
#         'rostralmiddlefrontal': '#00FFFF',
#         'superiorfrontal': '#7F7FFF',
#         'superiorparietal': '#7FFF7F',
#         'superiortemporal': '#FF7F7F',
#         'supramarginal': '#BFFF00',
#         'frontalpole': '#FFBF00',
#         'temporalpole': '#BF7F3F',
#         'transversetemporal': '#3FBFFF',
#         'insula': '#FFFF7F'
#     }
#     return {k: mcolors.to_rgba(v) for k, v in custom_colors.items()}


# def plot_projection_with_dk_atlas(image, dk_labels, label_names):
#     fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # 2行3列的布局
#     feature_names = ['Thickness', 'Volume', 'Surface Area', 'White/Gray Ratio']
    
#     for i, ax in enumerate(axes.flatten()[:4]):  # 前4个子图用于特征
#         im = ax.imshow(image[:, :, i], cmap='viridis')
#         ax.set_title(feature_names[i])
#         plt.colorbar(im, ax=ax)
    
#     # DK Atlas 图
#     color_map = create_custom_colormap()
#     dk_image = np.zeros((*image.shape[:2], 4))
#     for label in np.unique(dk_labels):
#         mask = image[:, :, 4] == label
#         dk_image[mask] = color_map[label_names[label]]
    
#     ax_dk = axes[1, 2]  # 使用第2行第3列的子图
#     ax_dk.imshow(dk_image)
#     ax_dk.set_title('DK Atlas')
    
#     # 移除多余的子图
#     fig.delaxes(axes[1, 1])
    
#     # 创建DK Atlas的图例
#     legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[name]) for name in label_names.values()]
#     fig.legend(legend_elements, label_names.values(), 
#                loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
    
#     plt.tight_layout()
#     plt.savefig('brain_projection_with_dk_atlas.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
# def main():
# # 这里应该是您读取数据的代码
#     n_subjects = 1  # 实际使用时改为12595
#     n_points_left = 149955
#     n_points_right = 149926

#     # load xyz 3D coordinates
#     # left brain
#     lh = open("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = lh.read().splitlines()
#     # data_truc is the raw data without the header
#     data_truc = data[2:]

#     data_lines = []
#     data_words = []

#     for i in range(len(data_truc)):
#         data_line = data_truc[i].split()
#         data_words = []
#         for j in range(len(data_line)):
#             data_word = float(data_line[j])
#             data_words.append(data_word)
#         data_lines.append(np.array(data_words))

#     # data_arr is the data array with correct datatype of each coloumn
#     data_arr_lh = np.array(data_lines)

#     # right brain
#     # rh = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
#     # # data is the raw data 
#     # data = rh.read().splitlines()
#     # # data_truc is the raw data without the header
#     # data_truc = data[2:]

#     # data_lines = []
#     # data_words = []

#     # for i in range(len(data_truc)):
#     #     data_line = data_truc[i].split()
#     #     data_words = []
#     #     for j in range(len(data_line)):
#     #         data_word = float(data_line[j])
#     #         data_words.append(data_word)
#     #     data_lines.append(np.array(data_words))

#     # # data_arr is the data array with correct datatype of each coloumn
# #     # data_arr_rh = np.array(data_lines)

#     # left brain
#     thick_path_lh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.thickness.fwhm10.fsaverage.mgh'
#     volume_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.volume.fwhm10.fsaverage.mgh'
#     SA_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.area.fwhm10.fsaverage.mgh'
#     # For ABCD, it could be the one with "-"
#     w_g_pct_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.w-g.pct.mgh.fwhm10.fsaverage.mgh'

#     # # right brain
#     # thick_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.thickness.fwhm10.fsaverage.mgh'
#     # volume_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.volume.fwhm10.fsaverage.mgh'
#     # SA_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.area.fwhm10.fsaverage.mgh'
#     # # For ABCD, it could be the one with "-"
#     # w_g_pct_path_rh = 'rh.w-g.pct.mgh.fwhm10.fsaverage.mgh'

#     ID_per_half = np.load('ij_id_lh.npy', allow_pickle=True)
#     ID_per_half = ID_per_half.astype(int)

#     # load MRI files
#     # left brain
#     thickness_array_re = SOI_array_per(ID_per_half, thick_path_lh)  # thickness in [0, 4.37891531]
#     volume_array_re = SOI_array_per(ID_per_half, volume_path_lh)   # volume in [0, 5.9636817]
#     SA_array_re = SOI_array_per(ID_per_half, SA_path_lh)   # surface_area in [0, 1.40500367]
#     w_g_array_re = SOI_array_per(ID_per_half, w_g_pct_path_lh) # w/g ratio in [0, 48.43599319]

#     # # right brain
#     # thickness_array_re_rh = SOI_array_per(ID_per_half, thick_path_rh)  
#     # volume_array_re_rh = SOI_array_per(ID_per_half, volume_path_rh)   
#     # SA_array_re_rh = SOI_array_per(ID_per_half, SA_path_rh)   
#     # w_g_array_re_rh = SOI_array_per(ID_per_half, w_g_pct_path_rh) 

#     # quantile normalization the SOI data
#     # left brain
#     thickness_mx_norm = quantile_normalize(thickness_array_re)
#     volume_mx_norm = quantile_normalize(volume_array_re)
#     SA_mx_norm = quantile_normalize(SA_array_re)
#     w_g_ar_norm = quantile_normalize(w_g_array_re)

#     # # right brain
#     # thickness_mx_norm_rh = quantile_normalize(thickness_array_re_rh)
#     # volume_mx_norm_rh = quantile_normalize(volume_array_re_rh)
#     # SA_mx_norm_rh = quantile_normalize(SA_array_re_rh)
#     # w_g_ar_norm_rh = quantile_normalize(w_g_array_re_rh)

#     # stack them as a matrix
#     SOI_mx_minmax_lh = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)
#     # SOI_mx_minmax_rh = np.stack([thickness_mx_norm_rh, volume_mx_norm_rh, SA_mx_norm_rh, w_g_ar_norm_rh], axis=-1)




#     for subject_id in range(n_subjects):
#         # 处理左半脑
#         x_left = data_arr_lh[:, 1]
#         y_left = data_arr_lh[:, 2]
#         z_left = data_arr_lh[:, 3]
#         coordinates_left = np.column_stack((x_left, y_left, z_left))
#         features_left = SOI_mx_minmax_lh.reshape((-1,4))

#         # Load DK atlas labels
#         annot_path = 'lh.aparc.annot'
#         labels, ctab, names = nib.freesurfer.read_annot(annot_path)
#         ij_id_lh = np.load('ij_id_lh.npy', allow_pickle=True)
#         ij_id_lh = ij_id_lh.astype(int)
#         labels = labels[ij_id_lh]
#         label_names = dict(zip(range(len(names)), [name.decode('utf-8') for name in names]))
#         # Perform projection
#         projected_image = optimized_mercator_projection(coordinates_left, features_left, labels)
        
#         # Plot results
#         plot_projection_with_dk_atlas(projected_image, labels, label_names)




#         # process_subject(subject_id, 'left', coordinates_left, features_left)

#         # # 处理右半脑
#         # x_right = data_arr_rh[:][1]
#         # y_right = data_arr_rh[:][2]
#         # z_right = data_arr_rh[:][3]
#         # coordinates_right = np.column_stack((x_right, y_right, z_right))
#         # features_right = np.random.randn(n_points_right, 4)
#         # process_subject(subject_id, 'right', coordinates_right, features_right)

# # %%
# if __name__ == "__main__":
#     main()

"""
old projection method
"""

# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from math import atan2, sqrt
# import numpy as np

# def xyz_to_longtitudinal(xyz_data_id):
#     """
#     Convert (x,y,z) coordinates to (phi, theta) spherical coordinates.
    
#     :param xyz_data_id: List containing [id, x, y, z]
#     :return: List containing [id, phi, theta] (phi and theta in radians)
#     """
#     id = xyz_data_id[0]
#     x = float(xyz_data_id[1])
#     y = float(xyz_data_id[2])
#     z = float(xyz_data_id[3])

#     phi = atan2(y, x)
#     theta = atan2(sqrt(x * x + y * y), z)

#     return [id, phi, theta]

# def get_longitudinal_map_each(each_xyz_data_id):
#     """
#     Transform xyz coordinates to longitude/colatitude space for each vertex.
    
#     :param each_xyz_data_id: List of strings, each containing "id x y z"
#     :return: List of [id, phi, theta] for each vertex
#     """
#     all_vertex_each = []
#     for vertex in each_xyz_data_id:
#         data_split = vertex.split()
#         temp = xyz_to_longtitudinal(data_split)
#         all_vertex_each.append(temp)
#     return all_vertex_each

# def get_ij_from_sphere(sphere_data_id, radius):
#     """
#     Sample (phi, theta) to (i, j) grid coordinates.
    
#     :param sphere_data_id: List containing [id, phi, theta]
#     :param radius: Radius of the sphere
#     :return: List containing [id, i, j] in 2D-grid format
#     """
#     id, phi, theta = sphere_data_id

#     i = radius * phi
    
#     if theta == 0 or theta == np.pi:
#         j = 195/2  # Half of the grid width at pole points
#     else:
#         j = radius * np.log(np.tan(((np.pi/2 - theta) + (np.pi/2))/2))

#     return [id, i, j]

# def sphere_to_grid_each(longitudinal_each_person, radius):
#     """
#     Convert spherical coordinates to 2D grid for each person's hemisphere.
    
#     :param longitudinal_each_person: List of [id, phi, theta] for each vertex
#     :param radius: Radius of the sphere
#     :return: List of [id, i, j] and numpy array of the same
#     """
#     list_each_half = [get_ij_from_sphere(vertex, radius) for vertex in longitudinal_each_person]
#     grid_each_half = np.array(list_each_half, dtype=object)
#     print(f"Grid shape: {grid_each_half.shape}")
#     return list_each_half, grid_each_half

# def plot_original(origin_ij_grid):
#     """
#     Plot the original (i,j) coordinates.
    
#     :param origin_ij_grid: numpy array of shape (n, 3) containing [id, i, j]
#     """
#     i_mx = np.matrix(origin_ij_grid[:, 1].astype('float').reshape((14,10709)))
#     j_mx = np.matrix(origin_ij_grid[:, 2].astype('float').reshape((14,10709)))

#     print(f"i range: {np.min(i_mx)} to {np.max(i_mx)}")
#     print(f"j range: {np.min(j_mx)} to {np.max(j_mx)}")

#     plt.figure(figsize=(12, 10))
#     plt.plot(i_mx, j_mx, 'b.')
#     plt.title('Original (i,j) Grid')
#     plt.xlabel('i')
#     plt.ylabel('j')
#     plt.show()

# def main():
#     # Example usage
#     radius = 100

#     # Generate sample data (replace this with your actual data)
#     # data_arr_lh = np.loadtxt("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", skiprows=2)
#     # left_coordinates = data_arr_lh[:, 1:4]
#     # x = left_coordinates[:,0]
#     # y = left_coordinates[:,1]
#     # z = left_coordinates[:,2]
    
#     # Create sample input data
#     a = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = a.read().splitlines()
#     # data_truc is the raw data without the header
#     each_xyz_data_id = data[2:]

#     # Process the data
#     longitudinal_map = get_longitudinal_map_each(each_xyz_data_id)
#     list_each_half, grid_each_half = sphere_to_grid_each(longitudinal_map, radius)

#     # Plot the result
#     plot_original(grid_each_half)
# # %%
# if __name__ == "__main__":
#     main()

# # # %%

# %%
