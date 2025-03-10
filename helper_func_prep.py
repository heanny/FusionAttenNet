"""
This file saves all helper functions we used.
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each

# Load SOI data and maintain the array of 769*195 for each SOI
# We maintain the array of 769*195 of SOI (signals of interests, thickness/surface area/volume.)
def SOI_array_per_right(ID_semi, SOI_path):
    """
    return the reshaped array for each SOI data for each person given the ID of 149955 vertices (left, right: 149926) of semi sphere.
    """
    SOI_load = nib.load(SOI_path)
    SOI_raw = SOI_load.get_fdata() 
    # Get SOI data of given ID for per person (hemi-sphere)
    SOI_array_raw = SOI_raw[ID_semi]
    SOI_array_reshape = SOI_array_raw[:,0,0].reshape((14, 10709)) #left (769, 195), right (14, 10709)
    return SOI_array_reshape

def SOI_array_per_left(ID_semi, SOI_path):
    """
    return the reshaped array for each SOI data for each person given the ID of 149955 vertices (left, right: 149926) of semi sphere.
    """
    SOI_load = nib.load(SOI_path)
    SOI_raw = SOI_load.get_fdata() 
    # Get SOI data of given ID for per person (hemi-sphere)
    SOI_array_raw = SOI_raw[ID_semi]
    SOI_array_reshape = SOI_array_raw[:,0,0].reshape((769, 195)) #left (769, 195), right (14, 10709)
    return SOI_array_reshape

def SOI_array_per_fs5(ID_semi, SOI_path):
    """
    return the reshaped array for each SOI data for each person given the ID of 149955 vertices of semi sphere.
    """
    SOI_load = nib.load(SOI_path)
    SOI_raw = SOI_load.get_fdata() 
    # Get SOI data of given ID for per person (hemi-sphere)
    SOI_array_raw = SOI_raw[ID_semi]
    SOI_array_reshape = SOI_array_raw[:10242,0,0].reshape((569, 18))
    return SOI_array_reshape


# stacked SOI (769, 195, 4) using min-max norm and plot it
def min_max_normalize(matrix):
    """
    function of min_max normalize for SOI data
    """
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    return normalized_matrix


def robust_scale_normalize(matrix):
    """
    Function to normalize data using RobustScaler for SOI data
    """
    # create RobustScaler scaler
    scaler = RobustScaler()
    # get the oroginal shape of the matrix, and make it into 2D matrix
    original_shape = matrix.shape
    matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    
    normalized_matrix = scaler.fit_transform(matrix_2d)
    
    # transform the normalized matrix to the original shape
    normalized_matrix = normalized_matrix.reshape(original_shape)
    
    return normalized_matrix

def robust_scale_normalize_display(matrix, is_right_brain=False):
    """
    Function to normalize data using RobustScaler for SOI data
    
    Args:
        matrix: Input data matrix
        is_right_brain: Boolean indicating if the data is from right hemisphere
    """
    # Create RobustScaler
    scaler = RobustScaler()
    
    # Flatten the matrix first to ensure consistent scaling
    flattened = matrix.flatten()
    
    # Reshape to 2D array for sklearn
    data_2d = flattened.reshape(-1, 1)
    
    # Apply scaling
    normalized = scaler.fit_transform(data_2d)
    
    # Reshape back to original shape
    if is_right_brain:
        return normalized.reshape(14, 10709)
    else:
        return normalized.reshape(769, 195)


def quantile_normalize(matrix):
    """
    Function to normalize data using QuantileTransformer for SOI data
    """
    # create RobustScaler scaler
    scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
    # get the oroginal shape of the matrix, and make it into 2D matrix
    original_shape = matrix.shape
    matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    

    normalized_matrix = scaler.fit_transform(matrix_2d)
    
    # transform the normalized matrix to the original shape
    normalized_matrix = normalized_matrix.reshape(original_shape)
    
    return normalized_matrix


def log_transform_and_scale(x):
    # log transformation
    x_log = np.log1p(x)  # log1p is log(x + 1)
    # Min-Max 
    return min_max_normalize(x_log)

def plot_SOI(SOI_mx_minmax):
    """
    function of plot the SOI data
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # plot this 3d array of shape (769, 195, 4)
    for i in range(4):
        ax = axes[i]
        im = ax.imshow(SOI_mx_minmax[:, :, i], cmap='viridis')
        ax.set_title(f'Matrix {i + 1}')
        ax.axis('off')

        # Add color bar to each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value')

    plt.tight_layout()
    plt.show()

def plot_SOI_model(SOI_mx_minmax):
    """
    function of plot the SOI data
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # plot this 3d array of shape (769, 195, 4)
    for i in range(4):
        ax = axes[i]
        im = ax.imshow(SOI_mx_minmax[i, :, :], cmap='viridis')
        ax.set_title(f'Matrix {i + 1}')
        ax.axis('off')

        # Add color bar to each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value')

    plt.tight_layout()
    plt.show()


def brain_SOI_matrix_left(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
    # load the cortex label file, which is used for select the vertices.
    # left half brain: 149955, right half brain: 149926
    ij_id = np.load('ij_id_lh.npy', allow_pickle=True)
    ID_per_half = ij_id.astype('int')

    # load MRI files 
    thickness_array_re = SOI_array_per_left(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
    volume_array_re = SOI_array_per_left(ID_per_half, volume_path)   # volume in [0, 5.9636817]
    SA_array_re = SOI_array_per_left(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
    w_g_array_re = SOI_array_per_left(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

    # min-max normalize the SOI data
    # thickness_mx_norm = min_max_normalize(thickness_array_re)
    # volume_mx_norm = min_max_normalize(volume_array_re)
    # SA_mx_norm = min_max_normalize(SA_array_re)
    # w_g_ar_norm = w_g_array_re/100
    
    # robust scaler normalization the SOI data
    thickness_mx_norm = robust_scale_normalize(thickness_array_re)
    volume_mx_norm = robust_scale_normalize(volume_array_re)
    SA_mx_norm = robust_scale_normalize(SA_array_re)
    w_g_ar_norm = robust_scale_normalize(w_g_array_re)

    # quantile normalization the SOI data
    # thickness_mx_norm = quantile_normalize(thickness_array_re)
    # volume_mx_norm = quantile_normalize(volume_array_re)
    # SA_mx_norm = quantile_normalize(SA_array_re)
    # w_g_ar_norm = quantile_normalize(w_g_array_re)

    # stack them as a matrix
    SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)

    return SOI_mx_minmax

def brain_SOI_matrix_right(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
    # load the cortex label file, which is used for select the vertices.
    # left half brain: 149955, right half brain: 149926
    # This is the Longitude transform for one vertex of a person 
    ij_id = np.load('ij_id_rh.npy', allow_pickle=True)
    ID_per_half = ij_id.astype('int')

    # load MRI files 
    thickness_array_re = SOI_array_per_right(ID_per_half, thick_path)  # 
    volume_array_re = SOI_array_per_right(ID_per_half, volume_path)   # 
    SA_array_re = SOI_array_per_right(ID_per_half, SA_path)   # 
    w_g_array_re = SOI_array_per_right(ID_per_half, w_g_pct_path) # 

    # min-max normalize the SOI data
    # thickness_mx_norm = min_max_normalize(thickness_array_re)
    # volume_mx_norm = min_max_normalize(volume_array_re)
    # SA_mx_norm = min_max_normalize(SA_array_re)
    # w_g_ar_norm = w_g_array_re/100
    
    # robust scaler normalization the SOI data
    thickness_mx_norm = robust_scale_normalize(thickness_array_re)
    volume_mx_norm = robust_scale_normalize(volume_array_re)
    SA_mx_norm = robust_scale_normalize(SA_array_re)
    w_g_ar_norm = robust_scale_normalize(w_g_array_re)

    # quantile normalization the SOI data
    # thickness_mx_norm = quantile_normalize(thickness_array_re)
    # volume_mx_norm = quantile_normalize(volume_array_re)
    # SA_mx_norm = quantile_normalize(SA_array_re)
    # w_g_ar_norm = quantile_normalize(w_g_array_re)

    # stack them as a matrix
    SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)

    return SOI_mx_minmax