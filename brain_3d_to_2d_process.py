# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import nibabel as nib
# from scipy.ndimage import distance_transform_edt, gaussian_filter, rotate
# import seaborn as sns
# from sklearn.preprocessing import RobustScaler
import os
# import scipy.ndimage
# import pandas as pd
# import numpy as np
# import gc
# import os
# import matplotlib.pyplot as plt
from helper_func_prep import SOI_array_per_left, min_max_normalize, plot_SOI, robust_scale_normalize, log_transform_and_scale
# import seaborn as sns
# from scipy.stats import binned_statistic_2d
# from scipy.ndimage import distance_transform_edt, gaussian_filter
# import shutil

# """

# 2. brain data
#   > 3d-to-2d projection (512*512 images, robust scale, smoothed) @/home/jouyang/brain-3d-to-2d-projection.py

# """
# # 
# def check_disk_space(required_space, directory):
#     """检查指定目录是否有足够的磁盘空间"""
#     try:
#         stats = shutil.disk_usage(directory)
#         available_space = stats.free
#         if available_space < required_space:
#             return False, f"需要 {required_space/(1024**3):.2f}GB, 但只有 {available_space/(1024**3):.2f}GB 可用"
#         return True, "enough space"
#     except Exception as e:
#         return False, str(e)

# def estimate_batch_size(image_shape, num_features=4, safety_factor=2):
#     """估计单个批次所需的内存大小（以字节为单位）"""
#     single_subject_size = np.prod(image_shape) * num_features * 4  # 4 bytes per float32
#     return single_subject_size * safety_factor

# def normalize_coordinates(coordinates):
#     r = np.sqrt(np.sum(coordinates**2, axis=1))
#     normalized = coordinates / r[:, np.newaxis]
#     return normalized

# def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
#     if features.ndim == 3:
#         features = features.reshape(-1, 4)
    
#     if features.shape[0] != coordinates.shape[0]:
#         raise ValueError(f"features numbers ({features.shape[0]}) does not match with coordinates numbers ({coordinates.shape[0]}).")

#     valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
#                  ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
#     coordinates = coordinates[valid_mask]
#     features = features[valid_mask]

#     normalized_coords = normalize_coordinates(coordinates)
#     x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(np.clip(z / r, -1, 1))
#     phi = np.arctan2(y, x)
    
#     u = phi
#     v = np.log(np.tan(theta/2 + np.pi/4))
    
#     # Handle potential NaN or inf values in v
#     v = np.nan_to_num(v, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    
#     max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))
#     v = np.clip(v, -max_v, max_v)
    
#     valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
#     u = u[valid_uv_mask]
#     v = v[valid_uv_mask]
#     features = features[valid_uv_mask]
    
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
    
#     image = np.nan_to_num(image)
    
#     return image

# def improved_fill_gaps(image, max_distance=10):
#     mask = np.isnan(image)
#     filled_image = np.copy(image)

#     for c in range(image.shape[2]):
#         channel = image[:, :, c]
#         channel_mask = mask[:, :, c]
        
#         if not np.any(channel_mask):
#             continue  # Skip if there are no gaps to fill
        
#         dist = distance_transform_edt(channel_mask)  # 计算空白区域到最近数据点的距离
        
#         weights = np.exp(-dist / max_distance)# 指数加权插值
#         weights[dist > max_distance] = 0
        
#         weight_sum = np.sum(weights)
#         if weight_sum > 0:
#             weights /= weight_sum
        
#         filled = np.sum(channel[:, :, np.newaxis] * weights, axis=(0, 1))
        
#         filled_image[channel_mask, c] = filled

#     return filled_image

# def smooth_image(image, kernel_size=3):
#     smoothed = np.copy(image)
#     for c in range(image.shape[2]):
#         smoothed[:, :, c] = gaussian_filter(image[:, :, c], sigma=kernel_size/2)
#     return smoothed

# def process_subject(subject_info, coordinates, ID_per_half, left=True):
#     feature_files = [
#         f"{'lh' if left else 'rh'}.thickness.fwhm10.fsaverage.mgh",
#         f"{'lh' if left else 'rh'}.volume.fwhm10.fsaverage.mgh",
#         f"{'lh' if left else 'rh'}.area.fwhm10.fsaverage.mgh",
#         f"{'lh' if left else 'rh'}.w_g.pct.mgh.fwhm10.fsaverage.mgh"
#     ]
    
#     features = []
#     for feature_file in feature_files:
#         file_path = subject_info[2][feature_file]
#         feature_data = SOI_array_per_left(ID_per_half, file_path) #SOI_array_per_left if left = True, no worries, you used the correct function for lh.
#         feature_norm = robust_scale_normalize(feature_data)
#         features.append(feature_norm)
        
#     SOI_mx_minmax = np.stack(features, axis=-1)
    
#     image = optimized_mercator_projection(coordinates, SOI_mx_minmax)
#     filled_image = improved_fill_gaps(image)
#     smoothed_image = smooth_image(filled_image)
    
#     # Clear some memory
#     # del image, filled_image, features, SOI_mx_minmax
#     # gc.collect()
    
#     return image, filled_image, smoothed_image.transpose(2, 0, 1)


# def process_hemisphere(subjects_info, coordinates, ID_per_half, is_left):
#     output_dir = '/projects/0/einf1049/scratch/jouyang'
#     output_file = os.path.join(output_dir, f"all_cnn_{'lh' if is_left else 'rh'}_brainimages.npy")
#     os.makedirs(output_dir, exist_ok=True)
    
#     n_subjects = len(subjects_info)
    
#     # 检查是否存在未完成的文件
#     if os.path.exists(output_file):
#         try:
#             existing_data = np.load(output_file, mmap_mode='r')
#             if existing_data.shape == (n_subjects, 4, 512, 512):
#                 print(f"Found existing file: {output_file}")
#                 # 查找最后处理的subject
#                 for i in range(n_subjects):
#                     if np.all(np.isnan(existing_data[i])):
#                         start_index = i
#                         break
#                 else:
#                     print("File seems complete, skipping processing")
#                     return output_file
#                 print(f"Resuming from subject {start_index}")
#             else:
#                 print("Existing file has wrong shape, starting over")
#                 start_index = 0
#         except Exception as e:
#             print(f"Error reading existing file: {e}")
#             start_index = 0
#     else:
#         start_index = 0
    
#     # 创建或打开内存映射文件
#     mmap_array = np.lib.format.open_memmap(
#         output_file,
#         mode='w+' if start_index == 0 else 'r+',
#         dtype=np.float32,
#         shape=(n_subjects, 4, 512, 512)
#     )
    
#     # 如果是新文件，用NaN初始化
#     if start_index == 0:
#         mmap_array[:] = np.nan
#         mmap_array.flush()
    
#     for i in range(start_index, n_subjects):
#         try:
#             print(f"Processing {'left' if is_left else 'right'} brain: subject {i+1}/{n_subjects}")
#             image, filled_image, subject_data = process_subject(subjects_info[i], coordinates, ID_per_half, is_left)
#             mmap_array[i] = subject_data
            
#             # 每100个subject保存一次
#             if (i + 1) % 100 == 0:
#                 print(f"Saving progress... ({i+1}/{n_subjects})")
#                 mmap_array.flush()
            
#             # Clear memory
#             # del subject_data
#             # gc.collect()
            
#         except Exception as e:
#             print(f"Error processing {'left' if is_left else 'right'} brain subject {i+1}: {str(e)}")
#             mmap_array.flush()  # 保存已处理的数据
#             continue
    
#     # 最终保存
#     mmap_array.flush()
    
#     # 验证数据
#     try:
#         verify_data = np.load(output_file, mmap_mode='r')
#         print(f"Verification - Data shape: {verify_data.shape}")
#         print(f"Verification - Any NaN: {np.any(np.isnan(verify_data))}")
#         del verify_data
#     except Exception as e:
#         print(f"Warning: Verification failed: {e}")
    
#     print(f"Saved data to: {output_file}")
#     return image, filled_image, subject_data

# def main():
#     # required_space = 50 * (1024**3)
#     # output_dir = '/projects/0/einf1049/scratch/jouyang'
#     # # space_ok, message = check_disk_space(required_space, output_dir)  # 检查输出目录的空间
#     # if not space_ok:
#     #     print(f"Insufficient disk space: {message}")
#     #     return

#     # print("Loading subject information...")
#     subjects_info = np.load('/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy', allow_pickle=True)
    
#     # Process left brain
#     # print("\nProcessing left brain...")
#     # with open("/projects/0/einf1049/scratch/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r") as lh:
#     #     data = lh.read().splitlines()[2:]
#     # data_arr_lh = np.array([list(map(float, line.split())) for line in data])
#     # coordinates_left = np.column_stack((data_arr_lh[:, 1], data_arr_lh[:, 2], data_arr_lh[:, 3]))
#     # ID_per_half_left = np.load('ij_id_lh.npy', allow_pickle=True).astype(int)
    
#     # left_file = process_hemisphere(subjects_info, coordinates_left, ID_per_half_left, True)

#     # # check left brain shape
#     # left_data = np.load(left_file, mmap_mode='r')
#     # print(f"Left brain data shape: {left_data.shape}")
#     # del left_data
    
#     # # Clear memory
#     # del coordinates_left, ID_per_half_left
#     # gc.collect()
    
#     # Process right brain
#     print("\nProcessing left brain...")
#     with open("/projects/0/einf1049/scratch/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r") as rh:
#         data = rh.read().splitlines()[2:]
#     data_arr_rh = np.array([list(map(float, line.split())) for line in data])
#     coordinates_right = np.column_stack((data_arr_rh[:, 1], data_arr_rh[:, 2], data_arr_rh[:, 3]))
#     ID_per_half_right = np.load('ij_id_lh.npy', allow_pickle=True).astype(int)
#     subjects_info_ = subjects_info[0]

#     plt.figure(figsize=(20, 15))
#     plt.suptitle('Brain MRI Processing Pipeline for Single Subject', fontsize=16)
    
#     # 1. Original 3D coordinates
#     ax1 = plt.subplot(331, projection='3d')
#     ax1.scatter(coordinates_right[:, 0], coordinates_right[:, 1], coordinates_right[:, 2], 
#                 c='blue', s=1, alpha=0.6)
#     ax1.set_title('1. Original 3D Coordinates')


#     image, filled_image, subject_data = process_subject(subjects_info_, coordinates_right, ID_per_half_right, False)
#     ax2 = plt.subplot(332)
#     im = ax2.imshow(image, cmap='viridis')
#     plt.colorbar(im, ax=ax2)
#     ax2.set_title('2. Mercator Projection\n(Thickness Channel)')

#     # ax3 = plt.subplot(333)
#     # im = ax3.imshow(filled_image, cmap='viridis')
#     # plt.colorbar(im, ax=ax3)
#     # ax3.set_title('3. Filled gap in 2D image')

#     # ax4 = plt.subplot(334)
#     # im = ax4.imshow(subject_data, cmap='viridis')
#     # plt.colorbar(im, ax=ax4)
#     # ax3.set_title('4. Smoothing out the 2D image')

#     # Process all channels
#     processed_channels = []
#     titles = ['Thickness', 'Volume', 'Area', 'W/G Ratio']
#     for i in range(4):
#         channel_data = subject_data[i]
#         processed_channels.append(channel_data)
#         ax = plt.subplot(3, 4, i+5)
#         im = ax.imshow(channel_data, cmap='viridis')
#         plt.colorbar(im, ax=ax)
#         ax.set_title(f'3. Mapped {titles[i]} Channel')

#     # Standardization
#     mean = np.mean(subject_data, axis=1).astype(np.float32)# image shape: (N, C, H, W)
#     std = np.std(subject_data, axis=1).astype(np.float32)
#     # thickness_data = subject_data[0]
#     # mean = np.mean(thickness_data)
#     # std = np.std(thickness_data)
#     standardized = (subject_data - mean[:, None, None]) / (std[:, None, None] + 1e-8)
#     ax5 = plt.subplot(335)
#     im = ax5.imshow(standardized[0], cmap='viridis')
#     plt.colorbar(im, ax=ax5)
#     ax5.set_title('4. Standardization on feature thickness')

#     # Random rotation
#     angle = np.random.uniform(-10, 10)
#     rotated_images = scipy.ndimage.rotate(standardized, angle, axes=(1,2), reshape=False, mode='nearest')
#     ax10 = plt.subplot(336)
#     im = ax10.imshow(rotated_images[0], cmap='viridis')
#     plt.colorbar(im, ax=ax10)
#     ax10.set_title(f'5. Random Rotation (angle={angle}°) on feature thickness')
    
#     # Add noise
#     noise = np.random.normal(0, 0.02, rotated_images.shape).astype(np.float32)
#     noised_images = rotated_images + noise
        
#     ax11 = plt.subplot(337)
#     im = ax11.imshow(noised_images[0], cmap='viridis')
#     plt.colorbar(im, ax=ax11)
#     ax11.set_title('6. Noise Addition on feature thickness')
    
#     plt.tight_layout()
#     plt.savefig('single_subject_processing.png', dpi=300, bbox_inches='tight')
#     plt.close()
# # 
# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, gaussian_filter
import scipy.ndimage

# 修正 1: 预防 log(tan()) 计算异常
# def safe_mercator_projection(theta):
#     tan_theta = np.tan(theta / 2 + np.pi / 4)
#     tan_theta = np.maximum(tan_theta, 1e-6)  # 避免 log(0) 或负数
#     return np.log(tan_theta)
# def normalize_coordinates(coordinates):
#     r = np.sqrt(np.sum(coordinates**2, axis=1))
#     normalized = coordinates / r[:, np.newaxis]
#     return normalized
# # 修正 2: 重新定义 Mercator 投影
# def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
#     if features.ndim == 3:
#         features = features.reshape(-1, 4)
    
#     if features.shape[0] != coordinates.shape[0]:
#         raise ValueError(f"features numbers ({features.shape[0]}) does not match with coordinates numbers ({coordinates.shape[0]}).")

#     valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
#                  ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
#     coordinates = coordinates[valid_mask]
#     features = features[valid_mask]

#     normalized_coords = normalize_coordinates(coordinates)
#     x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(np.clip(z / r, -1, 1))
#     phi = np.arctan2(y, x)

#     u = phi
#     v = safe_mercator_projection(theta)

#     # 限制 v 的最大值，避免极端情况
#     max_v = safe_mercator_projection(np.pi * 0.95 / 2)
#     v = np.clip(v, -max_v, max_v)

#     # 创建 512x512 映射
#     u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
#     v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)

#     image = np.zeros((*image_size, 4))
#     for i in range(4):
#         result = binned_statistic_2d(u, v, features[:, i], statistic='mean', bins=[u_bins, v_bins])
#         image[:, :, i] = np.nan_to_num(result.statistic.T)  # 避免 NaN

#     return image

# # 修正 3: 处理 `imshow()` 形状错误
# def visualize_transformation(subject_data):
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     feature_names = ['Thickness', 'Volume', 'Area', 'W/G Ratio']
    
#     for i, ax in enumerate(axes.flat):
#         im = ax.imshow(subject_data[i], cmap='viridis')
#         ax.set_title(feature_names[i])
#         ax.axis('off')
#         plt.colorbar(im, ax=ax)

#     plt.suptitle("Brain Feature Mapping (512x512)")
#     # plt.show()
#     plt.savefig('single_subject_processing.png', dpi=300, bbox_inches='tight')
#     plt.close()

# # 假设 `subject_data` 是 `(4, 512, 512)`，进行标准化处理
# def preprocess_for_cnn(subject_data):
#     mean = np.mean(subject_data, axis=(1, 2), keepdims=True)
#     std = np.std(subject_data, axis=(1, 2), keepdims=True) + 1e-8  # 避免除零
#     standardized = (subject_data - mean) / std

#     # 旋转变换
#     angle = np.random.uniform(-10, 10)
#     rotated_images = scipy.ndimage.rotate(standardized, angle, axes=(1, 2), reshape=False, mode='nearest')

#     # 添加噪声
#     noise = np.random.normal(0, 0.02, rotated_images.shape).astype(np.float32)
#     noised_images = rotated_images + noise

#     return noised_images
def load_data(image_path, phenotype_path, use_mmap=True):
    """
    Load data with mmap mode

    Parameters:
    image_path : str
        brain image data path
    phenotype_path : str
        phenotype data path
    use_mmap : bool
        use mmap mode or not
    
    Returns:
    tuple : (image_data, phenotype_data)

    """
    try:
        # check file size
        image_size = os.path.getsize(image_path) / (1024 ** 3)  # to GB
        phenotype_size = os.path.getsize(phenotype_path) / (1024 ** 3)
        
        # If the file is large and mmap is enabled, use memory mapped mode
        if use_mmap and (image_size > 1 or phenotype_size > 1):  # if it's larger than 1GB
            image_data = np.load(image_path, mmap_mode='r')  # read-only mode
            
            phenotype_data = np.load(phenotype_path, mmap_mode='r')
            print(f"Loaded data using memory mapping. Image data shape: {image_data.shape}")
        else:
            image_data = np.load(image_path)
            phenotype_data = np.load(phenotype_path)
            print(f"Loaded data into memory. Image data shape: {image_data.shape}")
        
        return image_data, phenotype_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
import numpy as np
import matplotlib.pyplot as plt

def visualize_and_save(image, save_path="brain_features.png"):
    """
    Visualize and save the four-channel (4, 512, 512) image.
    
    Parameters:
    - image: numpy array of shape (4, 512, 512)
    - save_path: str, the file path to save the generated image
    """
    feature_names = ['Cortical Thickness', 'Surface Area', 'Volume', 'White-to-Gray Matter Signal Intensity Ratio']

    # 创建 1x4 子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, ax in enumerate(axes):
        im = ax.imshow(image[i], cmap='gray')  # 用 viridis 颜色映射
        ax.set_title(feature_names[i])  # 设置标题
        ax.axis('off')  # 关闭坐标轴
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # 添加 colorbar

    plt.suptitle("Brain Feature Maps (512x512)")  # 总标题
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization as {save_path}")

# 假设你的数据是 (4, 512, 512)
# 这里用随机数据进行测试
# image = np.random.rand(4, 512, 512)  # 4 通道 512x512 图像

# # 运行可视化并保存
# visualize_and_save(image, save_path="brain_features_visualization.png")

# 测试代码
if __name__ == "__main__":
    # np.random.seed(42)
    image_path = '/home/jouyang1/sample_cnn_lh_brainimages.npy'
    phenotype_path = '/home/jouyang1/sample_normalised_phenotype.npy'
    image_data, loaded_phenotype_tensor = load_data(
        image_path, 
        phenotype_path,
        use_mmap=True  # 启用内存映射
    )
    print(image_data[0].shape)
    visualize_and_save(image_data[0], save_path="brain_features_visualization.png")

