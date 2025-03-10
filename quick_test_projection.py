import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from helper_func_prep import SOI_array_per,SOI_array_per_fs5, min_max_normalize, plot_SOI, robust_scale_normalize, quantile_normalize, log_transform_and_scale
from scipy.signal import convolve2d

def normalize_coordinates(coordinates):
    """
    将坐标归一化到单位球体上。
    
    :param coordinates: 形状为(n_points, 3)的numpy数组，包含x, y, z坐标
    :return: 归一化后的坐标
    """
    # 计算径向距离
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    # 归一化到单位球面
    normalized = coordinates / r[:, np.newaxis]
    return normalized

def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
    """
    使用优化的Mercator投影将3D坐标点和特征数据转换为2D图像。
    
    :param coordinates: 形状为(n_points, 3)的numpy数组，包含x, y, z坐标
    :param features: 形状为(n_points, 4)的numpy数组，包含A, B, C, D特征值
    :param image_size: 输出图像的尺寸，默认为(512, 512)
    :return: 形状为(512, 512, 4)的numpy数组，表示四通道图像
    """

     # 确保features的形状正确
    if features.ndim == 3:
        features = features.reshape(-1, 4)
    
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError(f"特征数量 ({features.shape[0]}) 与坐标点数量 ({coordinates.shape[0]}) 不匹配")
    

    # 移除包含NaN的行
    valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
                 ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    coordinates = coordinates[valid_mask]
    features = features[valid_mask]

    # 首先归一化坐标
    normalized_coords = normalize_coordinates(coordinates)
    x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
    # 将笛卡尔坐标转换为球坐标
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    
    # 优化的Mercator投影
    u = phi
    v = np.log(np.tan(theta/2 + np.pi/4))
    
    # 处理极点附近的变形
    max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))  # 限制在极点附近0.95范围内
    v = np.clip(v, -max_v, max_v)
    
    # 确保u和v不包含NaN或无穷大
    valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
    u = u[valid_uv_mask]
    v = v[valid_uv_mask]
    features = features[valid_uv_mask]
    
    # 定义bins
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
    
    # 处理NaN值
    image = np.nan_to_num(image)
    
    return image


def fill_gaps(image):
    """
    填充图像中的空白区域。
    """
    mask = np.isnan(image)
    image[mask] = 0
    
    # 使用简单的邻域平均填充
    kernel = np.ones((3, 3)) / 9
    for i in range(image.shape[2]):  # 遍历每个通道
        channel = image[:, :, i]
        filled = channel.copy()
        for _ in range(3):  # 重复几次以填充较大的空白
            filled = convolve2d(filled, kernel, mode='same', boundary='symm')
            channel[mask[:, :, i]] = filled[mask[:, :, i]]
        image[:, :, i] = channel
    
    return image

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
    plt.savefig(filename.replace('.png', '_channels.png'))
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


def process_subject(subject_id, hemisphere, coordinates, features):
    """
    处理单个受试者的数据。
    
    :param subject_id: 受试者ID
    :param hemisphere: 'left' 或 'right'
    :param coordinates: 坐标数据
    :param features: 特征数据
    """
    image = optimized_mercator_projection(coordinates, features)
    filename = f"subject_{subject_id}_{hemisphere}.png"
    save_image(image, filename)
    print(f"Saved image for subject {subject_id}, {hemisphere} hemisphere")

# 主处理流程
def main():
    # 这里应该是您读取数据的代码
    n_subjects = 1  # 实际使用时改为12595
    n_points_left = 149955
    n_points_right = 149926

    # load xyz 3D coordinates
    # left brain
    lh = open("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
    # data is the raw data 
    data = lh.read().splitlines()
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

    # right brain
    # rh = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
    # # data is the raw data 
    # data = rh.read().splitlines()
    # # data_truc is the raw data without the header
    # data_truc = data[2:]

    # data_lines = []
    # data_words = []

    # for i in range(len(data_truc)):
    #     data_line = data_truc[i].split()
    #     data_words = []
    #     for j in range(len(data_line)):
    #         data_word = float(data_line[j])
    #         data_words.append(data_word)
    #     data_lines.append(np.array(data_words))

    # # data_arr is the data array with correct datatype of each coloumn
#     # data_arr_rh = np.array(data_lines)

    # left brain
    thick_path_lh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.thickness.fwhm10.fsaverage.mgh'
    volume_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.volume.fwhm10.fsaverage.mgh'
    SA_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.area.fwhm10.fsaverage.mgh'
    # For ABCD, it could be the one with "-"
    w_g_pct_path_lh  = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.w-g.pct.mgh.fwhm10.fsaverage.mgh'

    # # right brain
    # thick_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.thickness.fwhm10.fsaverage.mgh'
    # volume_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.volume.fwhm10.fsaverage.mgh'
    # SA_path_rh = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/rh.area.fwhm10.fsaverage.mgh'
    # # For ABCD, it could be the one with "-"
    # w_g_pct_path_rh = 'rh.w-g.pct.mgh.fwhm10.fsaverage.mgh'

    ID_per_half = np.load('ij_id_lh.npy', allow_pickle=True)
    ID_per_half = ID_per_half.astype(int)

    # load MRI files
    # left brain
    thickness_array_re = SOI_array_per(ID_per_half, thick_path_lh)  # thickness in [0, 4.37891531]
    volume_array_re = SOI_array_per(ID_per_half, volume_path_lh)   # volume in [0, 5.9636817]
    SA_array_re = SOI_array_per(ID_per_half, SA_path_lh)   # surface_area in [0, 1.40500367]
    w_g_array_re = SOI_array_per(ID_per_half, w_g_pct_path_lh) # w/g ratio in [0, 48.43599319]

    # # right brain
    # thickness_array_re_rh = SOI_array_per(ID_per_half, thick_path_rh)  
    # volume_array_re_rh = SOI_array_per(ID_per_half, volume_path_rh)   
    # SA_array_re_rh = SOI_array_per(ID_per_half, SA_path_rh)   
    # w_g_array_re_rh = SOI_array_per(ID_per_half, w_g_pct_path_rh) 

    # quantile normalization the SOI data
    # left brain
    thickness_mx_norm = quantile_normalize(thickness_array_re)
    volume_mx_norm = quantile_normalize(volume_array_re)
    SA_mx_norm = quantile_normalize(SA_array_re)
    w_g_ar_norm = quantile_normalize(w_g_array_re)

    # # right brain
    # thickness_mx_norm_rh = quantile_normalize(thickness_array_re_rh)
    # volume_mx_norm_rh = quantile_normalize(volume_array_re_rh)
    # SA_mx_norm_rh = quantile_normalize(SA_array_re_rh)
    # w_g_ar_norm_rh = quantile_normalize(w_g_array_re_rh)

    # stack them as a matrix
    SOI_mx_minmax_lh = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)
    # SOI_mx_minmax_rh = np.stack([thickness_mx_norm_rh, volume_mx_norm_rh, SA_mx_norm_rh, w_g_ar_norm_rh], axis=-1)




    for subject_id in range(n_subjects):
        # 处理左半脑
        x_left = data_arr_lh[:, 1]
        y_left = data_arr_lh[:, 2]
        z_left = data_arr_lh[:, 3]
        coordinates_left = np.column_stack((x_left, y_left, z_left))
        features_left = SOI_mx_minmax_lh
        process_subject(subject_id, 'left', coordinates_left, features_left)

        # # 处理右半脑
        # x_right = data_arr_rh[:][1]
        # y_right = data_arr_rh[:][2]
        # z_right = data_arr_rh[:][3]
        # coordinates_right = np.column_stack((x_right, y_right, z_right))
        # features_right = np.random.randn(n_points_right, 4)
        # process_subject(subject_id, 'right', coordinates_right, features_right)

# 
if __name__ == "__main__":
    main()


# # 
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def visualize_raw_3d_brain(coordinates, filename):
#     """
#     直接可视化原始3D脑部坐标。
    
#     :param coordinates: 形状为(n_points, 3)的numpy数组，包含x, y, z坐标
#     :param filename: 输出文件名
#     """
#     x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 使用散点图绘制3D点
#     scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.1)
    
#     # 设置坐标轴标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
    
#     # 设置标题
#     ax.set_title('Raw 3D Left Brain Coordinates')
    
#     # 添加颜色条
#     plt.colorbar(scatter, ax=ax, label='Z coordinate')
    
#     # 保存图像
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Raw 3D visualization saved as {filename}")

# # 使用示例
# def main():
#     # 使用您提供的坐标范围生成示例数据
#    # load xyz 3D coordinates
#     # left brain
#     lh = open("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
#     # rh = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
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

#     x = data_arr_lh[:, 1]
#     y = data_arr_lh[:, 2]
#     z = data_arr_lh[:, 3]
#     coordinates = np.column_stack((x, y, z))
#     visualize_raw_3d_brain(coordinates, "raw_3d_right_brain.png")

# if __name__ == "__main__":
#     main()



# # 
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # def visualize_whole_brain_3d(left_coordinates, right_coordinates, filename):
# #     """
# #     可视化左右半脑的原始3D坐标，提供更清晰的区分。
    
# #     :param left_coordinates: 形状为(n_points, 3)的numpy数组，包含左半脑的x, y, z坐标
# #     :param right_coordinates: 形状为(n_points, 3)的numpy数组，包含右半脑的x, y, z坐标
# #     :param filename: 输出文件名
# #     """
# #     fig = plt.figure(figsize=(15, 12))
# #     ax = fig.add_subplot(111, projection='3d')
    
# #     # 绘制左半脑
# #     ax.scatter(left_coordinates[:, 0], left_coordinates[:, 1], left_coordinates[:, 2], 
# #                c='red', s=0.5, alpha=0.1, label='Left Hemisphere')
    
# #     # 绘制右半脑
# #     ax.scatter(right_coordinates[:, 0], right_coordinates[:, 1], right_coordinates[:, 2], 
# #                c='blue', s=0.5, alpha=0.1, label='Right Hemisphere')
    
# #     # 设置坐标轴标签
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_zlabel('Z')
    
# #     # 设置标题
# #     ax.set_title('Whole Brain 3D Visualization')
    
# #     # 添加图例
# #     ax.legend()
    
# #     # 调整视角以更好地展示左右半脑
# #     ax.view_init(elev=20, azim=60)
    
# #     # 设置坐标轴范围，确保两个半脑都完全可见
# #     max_range = np.array([left_coordinates.max(axis=0), right_coordinates.max(axis=0)]).max()
# #     ax.set_xlim([-max_range, max_range])
# #     ax.set_ylim([-max_range, max_range])
# #     ax.set_zlim([-max_range, max_range])
    
# #     # 保存图像
# #     plt.savefig(filename, dpi=300, bbox_inches='tight')
# #     plt.close()
    
# #     print(f"Improved whole brain 3D visualization saved as {filename}")
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation

# def create_dynamic_brain_3d(left_coordinates, right_coordinates, filename):
#     """
#     创建动态的3D大脑可视化，并将结果保存为GIF。
    
#     :param left_coordinates: 形状为(n_points, 3)的numpy数组，包含左半脑的x, y, z坐标
#     :param right_coordinates: 形状为(n_points, 3)的numpy数组，包含右半脑的x, y, z坐标
#     :param filename: 输出文件名（应以.gif结尾）
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制左半脑和右半脑
#     scatter_left = ax.scatter(left_coordinates[:, 0], left_coordinates[:, 1], left_coordinates[:, 2], 
#                               c='red', s=0.5, alpha=0.1, label='Left Hemisphere')
#     scatter_right = ax.scatter(right_coordinates[:, 0], right_coordinates[:, 1], right_coordinates[:, 2], 
#                                c='blue', s=0.5, alpha=0.1, label='Right Hemisphere')

#     # 设置标题和图例
#     ax.set_title('Dynamic 3D Brain Visualization')
#     ax.legend()

#     # 设置坐标轴范围
#     max_range = np.array([left_coordinates.max(axis=0), right_coordinates.max(axis=0)]).max()
#     ax.set_xlim([-max_range, max_range])
#     ax.set_ylim([-max_range, max_range])
#     ax.set_zlim([-max_range, max_range])

#     # 动画更新函数
#     def update(frame):
#         ax.view_init(elev=20, azim=frame)
#         return scatter_left, scatter_right

#     # 创建动画
#     anim = FuncAnimation(fig, update, frames=np.linspace(0, 360, 180), 
#                          interval=50, blit=False)

#     # 保存为GIF
#     anim.save(filename, writer='pillow', fps=30)
#     plt.close()

#     print(f"Dynamic 3D brain visualization saved as {filename}")

# # 使用示例
# def main():
#         # load xyz 3D coordinates
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
#     rh = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = rh.read().splitlines()
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
#     data_arr_rh = np.array(data_lines)


#     # 左半脑数据
#     x_left = data_arr_lh[:, 1]
#     y_left = data_arr_lh[:, 2]
#     z_left = data_arr_lh[:, 3]
#     left_coordinates = np.column_stack((x_left, y_left, z_left))
    
#     # 右半脑数据（稍微平移以区分）
#     x_right = data_arr_rh[:, 1]
#     y_right = data_arr_rh[:, 2]
#     z_right = data_arr_rh[:, 3]
#     right_coordinates = np.column_stack((x_right, y_right, z_right))
    
#     # visualize_whole_brain_3d(left_coordinates, right_coordinates, "whole_brain_3d.png")
#     create_dynamic_brain_3d(left_coordinates, right_coordinates, "dynamic_brain_3d.gif")
# if __name__ == "__main__":
#     main()
# 

# 
# from longitude_transform import color_map_DK

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns

# def create_static_brain_3d(left_coordinates, right_coordinates, left_dk_values, right_dk_values, filename):
#     fig = plt.figure(figsize=(20, 16))
#     ax = fig.add_subplot(111, projection='3d')

#     # 创建一个离散的颜色映射
#     unique_dk_values = np.unique(np.concatenate((left_dk_values, right_dk_values)))
#     n_colors = len(unique_dk_values)
#     colors = sns.color_palette("husl", n_colors=n_colors)
#     color_dict = {value: colors[i] for i, value in enumerate(unique_dk_values)}

#     # 创建字典来存储散点对象
#     scatter_objects = {'Left': {}, 'Right': {}}

#     # 为每个半脑的每个DK值创建一个散点图
#     for hemisphere, coordinates, dk_values in [("Left", left_coordinates, left_dk_values), 
#                                                ("Right", right_coordinates, right_dk_values)]:
#         for value in unique_dk_values:
#             mask = dk_values == value
#             scatter = ax.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2],
#                                  c=[color_dict[value]], s=20, alpha=1, 
#                                  label=f'{hemisphere} {value}')
#             scatter_objects[hemisphere][value] = scatter

#     # 设置标题
#     ax.set_title('Debugging Static 3D Brain Visualization (Desikan-Killiany Atlas)', fontsize=16)

#     # 设置坐标轴范围
#     all_coordinates = np.vstack((left_coordinates, right_coordinates))
#     max_range = np.max(all_coordinates.max(axis=0) - all_coordinates.min(axis=0)) / 2
#     mid_x = (all_coordinates[:, 0].max() + all_coordinates[:, 0].min()) * 0.5
#     mid_y = (all_coordinates[:, 1].max() + all_coordinates[:, 1].min()) * 0.5
#     mid_z = (all_coordinates[:, 2].max() + all_coordinates[:, 2].min()) * 0.5
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)

#     # 设置视角
#     ax.view_init(elev=20, azim=135)

#     # 移除坐标轴刻度标签
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])

#     # 创建自定义图例
#     handles_left = [scatter_objects['Left'][value] for value in unique_dk_values if value in scatter_objects['Left']]
#     handles_right = [scatter_objects['Right'][value] for value in unique_dk_values if value in scatter_objects['Right']]
#     labels_left = [f'Left {value}' for value in unique_dk_values if value in scatter_objects['Left']]
#     labels_right = [f'Right {value}' for value in unique_dk_values if value in scatter_objects['Right']]

#     # 添加图例
#     fig.legend(handles_left + handles_right, labels_left + labels_right, 
#                loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='xx-small')

#     # 调整布局并保存
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"Debugging static 3D brain visualization saved as {filename}")
    
#     # 输出一些调试信息
#     print(f"Number of points: Left - {left_coordinates.shape[0]}, Right - {right_coordinates.shape[0]}")
#     print(f"Coordinate ranges: X [{all_coordinates[:, 0].min()}, {all_coordinates[:, 0].max()}], "
#           f"Y [{all_coordinates[:, 1].min()}, {all_coordinates[:, 1].max()}], "
#           f"Z [{all_coordinates[:, 2].min()}, {all_coordinates[:, 2].max()}]")


# def main():
#         # load xyz 3D coordinates
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
#     rh = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = rh.read().splitlines()
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
#     data_arr_rh = np.array(data_lines)


#     # 左半脑数据
#     x_left = data_arr_lh[:, 1]
#     y_left = data_arr_lh[:, 2]
#     z_left = data_arr_lh[:, 3]
#     left_coordinates = np.column_stack((x_left, y_left, z_left))
#     annot_path_lh = 'lh.aparc.annot'
#     ID_per_half_lh = np.load('ij_id_lh.npy', allow_pickle=True)
#     # ID_per_half_lh = ID_per_half_lh.astype(int)
#     _, c_group_id_lh, c_group_name_lh, _ = color_map_DK(annot_path_lh, ID_per_half_lh)
#     left_dk_values = c_group_name_lh


#     # 右半脑数据
#     x_right = data_arr_rh[:, 1]
#     y_right = data_arr_rh[:, 2]
#     z_right = data_arr_rh[:, 3]
#     right_coordinates = np.column_stack((x_right, y_right, z_right))
#     annot_path_rh = 'rh.aparc.annot'
#     ID_per_half_rh = np.load('ij_id_rh.npy', allow_pickle=True)
#     # ID_per_half_rh = ID_per_half_lh.astype(int)
#     _, c_group_id_rh, c_group_name_rh, _ = color_map_DK(annot_path_rh, ID_per_half_rh)
#     right_dk_values = c_group_name_rh
    
    
#     create_static_brain_3d(left_coordinates, right_coordinates, left_dk_values, right_dk_values, "static_brain_3d_dk_atlas.png")



# if __name__ == "__main__":
#     main()


# 
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# from longitude_transform import color_map_DK

# def create_separate_brain_3d(left_coordinates, right_coordinates, left_dk_values, right_dk_values, filename):
#     fig = plt.figure(figsize=(20, 16))
#     ax = fig.add_subplot(111, projection='3d')

#     # 将 dk_values 转换为 NumPy 数组
#     left_dk_values = np.array(left_dk_values)
#     right_dk_values = np.array(right_dk_values)

#     # 创建一个映射，将字符串 DK 值映射到整数
#     unique_dk_values = np.unique(np.concatenate((left_dk_values, right_dk_values)))
#     dk_to_int = {dk: i for i, dk in enumerate(unique_dk_values)}

#     # 创建颜色映射
#     n_colors = len(unique_dk_values)
#     colors = sns.color_palette("husl", n_colors=n_colors)
#     color_dict = {value: colors[i] for i, value in enumerate(unique_dk_values)}

#     def plot_hemisphere(coordinates, dk_values, hemisphere):
#         print(f"Debug: {hemisphere} hemisphere")
#         print(f"Debug: coordinates shape: {coordinates.shape}")
#         print(f"Debug: dk_values shape: {dk_values.shape}")

#         for value in unique_dk_values:
#             mask = dk_values == value
#             if np.sum(mask) > 0:
#                 ax.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2],
#                            c=[color_dict[value]], s=1, alpha=0.5, label=f'{hemisphere} {value.decode("utf-8")}', marker='.')

#     # 绘制左脑和右脑
#     plot_hemisphere(left_coordinates, left_dk_values, "Left")
#     plot_hemisphere(right_coordinates, right_dk_values, "Right")

#     # 设置标题
#     ax.set_title('3D Brain Visualization (Desikan-Killiany Atlas)', fontsize=16)

#     # 设置坐标轴范围
#     ax.set_xlim(-100, 100)
#     ax.set_ylim(-100, 100)
#     ax.set_zlim(-100, 100)

#     # 设置视角
#     ax.view_init(elev=20, azim=135)

#     # 移除坐标轴刻度标签
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])

#     # 创建自定义图例
#     handles, labels = ax.get_legend_handles_labels()
#     left_handles = [h for h, l in zip(handles, labels) if l.startswith('Left')]
#     right_handles = [h for h, l in zip(handles, labels) if l.startswith('Right')]
#     left_labels = [l.split(' ', 1)[1] for l in labels if l.startswith('Left')]
#     right_labels = [l.split(' ', 1)[1] for l in labels if l.startswith('Right')]

#     # 添加两列图例
#     fig.legend(left_handles + right_handles, left_labels + right_labels, 
#                loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='xx-small',
#                title='Left Brain          Right Brain', title_fontsize='small')

#     # 调整布局并保存
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=1)
#     plt.close()

#     print(f"3D brain visualization saved as {filename}")

# def main():
#     # 加载左脑数据
#     data_arr_lh = np.loadtxt("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", skiprows=2)
#     left_coordinates = data_arr_lh[:, 1:4]
    
#     # 加载右脑数据
#     data_arr_rh = np.loadtxt("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", skiprows=2)
#     right_coordinates = data_arr_rh[:, 1:4]

#     # 加载和处理左脑DK值
#     annot_path_lh = 'lh.aparc.annot'
#     ID_per_half_lh = np.load('ij_id_lh.npy', allow_pickle=True)
#     ID_per_half_lh = ID_per_half_lh.astype(int)
#     _, _, left_dk_values, _ = color_map_DK(annot_path_lh, ID_per_half_lh)

#     # 加载和处理右脑DK值
#     annot_path_rh = 'rh.aparc.annot'
#     ID_per_half_rh = np.load('ij_id_rh.npy', allow_pickle=True)
#     ID_per_half_rh = ID_per_half_rh.astype(int)
#     _, _, right_dk_values, _ = color_map_DK(annot_path_rh, ID_per_half_rh)


#     # 打印调试信息
#     print("Debug: left_dk_values length:", len(left_dk_values))
#     print("Debug: right_dk_values length:", len(right_dk_values))
#     print("Debug: left_coordinates shape:", left_coordinates.shape)
#     print("Debug: right_coordinates shape:", right_coordinates.shape)

#     create_separate_brain_3d(left_coordinates, right_coordinates, left_dk_values, right_dk_values, "static_brain_3d_dk_atlas_1.png")

# if __name__ == "__main__":
#     main()
# show 3d to 2d projection process
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from longitude_transform import xyz_to_longtitudinal, get_ij_from_sphere, get_longitudinal_map_each, sphere_to_grid_each, color_map_DK
# # 
# Load raw data
# a = open("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
# # data is the raw data 
# data = a.read().splitlines()
# # data_truc is the raw data without the header
# data_truc = data[2:]
# res = list(map(str.strip, data_truc[0].split()))
# haha = xyz_to_longtitudinal(res)
# hehe = get_ij_from_sphere(haha, 100)
# haha_per = get_longitudinal_map_each(data_truc)
# origin_ij_list, origin_ij_grid = sphere_to_grid_each(haha_per,100)
# i = origin_ij_grid[:,1]
# j = origin_ij_grid[:,2]
# haha_per = np.array(haha_per)
# phi = haha_per[:,1]
# theta = haha_per[:,2]
# # Convert to spherical coordinates
# # phi, theta = xyz_to_longtitudinal(x, y, z)
# #
# data_arr_lh = np.loadtxt("GenR_mri/lh.fsaverage.sphere.cortex.mask.label", skiprows=2)
# left_coordinates = data_arr_lh[:, 1:4]
# x = left_coordinates[:,0]
# y = left_coordinates[:,1]
# z = left_coordinates[:,2]
# filename = "3d_to_2d_projection_process.png"
# #
# # # Plotting
# # fig = plt.figure(figsize=(15, 5))
# # filename = "3d_to_2d_projection_process.png"
# # # Original 3D coordinates
# # ax1 = fig.add_subplot(131, projection='3d')
# # ax1.scatter(x, y, z)
# # ax1.set_title('Original XYZ')

# # # Spherical coordinates
# # ax2 = fig.add_subplot(132)
# # ax2.scatter(phi, theta)
# # ax2.set_title('Spherical Coordinates (phi, theta)')

# # # IJ coordinates
# # ax3 = fig.add_subplot(133)
# # ax3.scatter(i, j)
# # ax3.set_title('IJ Coordinates')

# # plt.tight_layout()
# # plt.show()

# # Plotting
# fig = plt.figure(figsize=(15, 5))

# # Original 3D coordinates
# ax1 = fig.add_subplot(131, projection='3d')
# ax1.scatter(x, y, z, s=1, alpha=0.1)
# ax1.set_title('Simulated Brain Surface (XYZ)')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # Spherical coordinates
# ax2 = fig.add_subplot(132)
# ax2.scatter(phi, theta, s=1, alpha=0.1)
# ax2.set_title('Spherical Coordinates (phi, theta)')
# ax2.set_xlabel('Phi')
# ax2.set_ylabel('Theta')
# ax2.set_xlim(0, 2*np.pi)
# ax2.set_ylim(0, np.pi)

# # IJ coordinates
# ax3 = fig.add_subplot(133)
# sc = ax3.scatter(i, j, s=1, alpha=0.1, c=z, cmap='viridis')
# ax3.set_title('IJ Coordinates')
# ax3.set_xlabel('I')
# ax3.set_ylabel('J')
# plt.colorbar(sc, ax=ax3, label='Z value')

# plt.tight_layout()
# plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=1)
# plt.close()

# print(f"3D brain visualization saved as {filename}")
# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import seaborn as sns
import nibabel as nib
from longitude_transform import xyz_to_longtitudinal, get_longitudinal_map_each, get_ij_from_sphere, sphere_to_grid_each, color_map_DK

def create_color_mapping(color_group_id, color_vertices):
    unique_ids = set(color_group_id)
    color_dict = {}
    for id in unique_ids:
        index = color_group_id.index(id)
        color_dict[id] = color_vertices[index]
    return color_dict

def plot_left_brain_visualizations(data_arr_lh, color_vertices, color_group_id, color_group_name, filename):
    fig = plt.figure(figsize=(20, 15))
    
    # Create color mapping
    color_dict = create_color_mapping(color_group_id, color_vertices)
    
    # 3D Brain Surface
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_brain(ax1, data_arr_lh, color_vertices)
    
    # Spherical Coordinates
    ax2 = fig.add_subplot(222)
    plot_spherical_coordinates(ax2, data_arr_lh, color_vertices)
    
    # IJ Coordinates
    ax3 = fig.add_subplot(223)
    plot_ij_coordinates(ax3, data_arr_lh, color_vertices)
    
    # DK Atlas Legend
    ax4 = fig.add_subplot(224)
    plot_dk_atlas_legend(ax4, color_dict, color_group_name, color_group_id)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=1)
    plt.close()
    print(f"Left brain visualizations saved as {filename}")

def plot_3d_brain(ax, data_arr_lh, color_vertices):
    data_arr_lh_ = np.loadtxt(data_arr_lh, skiprows=2)
    # left_coordinates = data_arr_lh[:, 1:4]
    # x = left_coordinates[:,0]
    # y = left_coordinates[:,1]
    # z = left_coordinates[:,2]
    
    ax.scatter(data_arr_lh_[:, 1], data_arr_lh_[:, 2], data_arr_lh_[:, 3], c=color_vertices, s=1, alpha=0.5)
    ax.set_title('3D Left Brain Surface (DK Atlas)', fontsize=12)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def plot_spherical_coordinates(ax, data_arr_lh, color_vertices):
    data = open(data_arr_lh, "r")
    data_lh = data.read().splitlines()
    data_truc_lh = data_lh[2:]
    print(f"Debug: First few lines of data_truc_lh: {data_truc_lh[:5]}")
    
    longitudinal_data = get_longitudinal_map_each(data_truc_lh)
    print(f"Debug: Shape of longitudinal_data: {np.array(longitudinal_data).shape}")
    print(f"Debug: First few rows of longitudinal_data: {longitudinal_data[:5]}")
    
    longitudinal_data = np.array(longitudinal_data, dtype=float)  # Convert to float array
    phi = longitudinal_data[:,1]
    theta = longitudinal_data[:,2]
    print(f"Debug: First few values of phi: {phi[:5]}")
    print(f"Debug: First few values of theta: {theta[:5]}")
    print(f"Debug: Shape of color_vertices: {np.array(color_vertices).shape}")

    # Convert phi to the range [-π, π]
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)

    scatter = ax.scatter(phi, theta, c=color_vertices, s=1, alpha=0.5)
    ax.set_title('Spherical Coordinates (DK Atlas)', fontsize=12)
    ax.set_xlabel('Phi')
    ax.set_ylabel('Theta')
    ax.set_xlim(-np.pi, np.pi)  # phi: [-π, π]
    ax.set_ylim(0, np.pi)      # theta: [0, π]
    # plt.colorbar(scatter, ax=ax)

def plot_ij_coordinates(ax, data_arr_lh, color_vertices):
    data = open(data_arr_lh, "r")
    data_lh = data.read().splitlines()
    data_truc_lh = data_lh[2:]
    longitudinal_data = get_longitudinal_map_each(data_truc_lh)
    _, grid_each_half = sphere_to_grid_each(longitudinal_data, 100)
    i = grid_each_half[:, 1].astype(float)
    j = grid_each_half[:, 2].astype(float)
    ax.scatter(i, j, c=color_vertices, s=1, alpha=0.5)
    ax.set_title('IJ Coordinates (DK Atlas)', fontsize=12)
    ax.set_xlabel('I')
    ax.set_ylabel('J')

def plot_dk_atlas_legend(ax, color_dict, color_group_name, color_group_id):
    print(f"Debug: color_group_name length: {len(color_group_name)}")
    print(f"Debug: color_group_id length: {len(color_group_id)}")
    print(f"Debug: color_dict keys: {color_dict.keys()}")

    # Count occurrences of each name
    name_counter = Counter(color_group_name)
    print(f"Debug: Unique names and their counts: {name_counter}")

    if len(name_counter) == 0:
        print("Warning: No unique names found in color_group_name")
        ax.text(0.5, 0.5, "No DK Atlas data available", ha='center', va='center')
    else:
        for name, count in name_counter.items():
            # Find the first occurrence of this name
            index = color_group_name.index(name)
            id = color_group_id[index]
            color = color_dict.get(id, 'gray')  # Use 'gray' as fallback color
            ax.bar(0, 0, color=color, label=f"{name} ({count})")

    ax.set_axis_off()
    if len(name_counter) > 0:
        ax.legend(loc='center', ncol=2, fontsize='xx-small', title='DK Atlas Regions')
    ax.set_title('DK Atlas Legend', fontsize=12)

    
def main():
    # Load left brain data
    data_arr_lh = "GenR_mri/lh.fsaverage.sphere.cortex.mask.label"
    # Load DK atlas values for left brain
    annot_path_lh = 'lh.aparc.annot'
    ID_per_half_lh = np.load('ij_id_lh.npy', allow_pickle=True).astype(int)
    color_vertices, color_group_id, color_group_name, _ = color_map_DK(annot_path_lh, ID_per_half_lh)

    plot_left_brain_visualizations(data_arr_lh, color_vertices, color_group_id, color_group_name, "left_brain_3d_to_2d_peocess.png")
# 
if __name__ == "__main__":
    main()


# 
