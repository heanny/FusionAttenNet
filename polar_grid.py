# %% IMPORTS

import numpy as np
from sklearn import neighbors
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# %%
def get_polar_grid(params={"height": 512, "width": 256, "radius": 100, "max_y": 2 * np.pi, "min_y": 0, "max_x": np.pi, "min_x": 0}):
    """
    Function to create polar grid and map it to cartesian.
    Params contains all important variables (height of image,
    width of image, sphere radius, maximum and minimum x-value and y-value)

    :param dict params: Dictionary containing all important variables:
                         - height: height of image (number rows, i.e. 256)
                         - width: width of image (number columns, i.e. 512)
                         - radius: spherical radius (freesurfer default = 100)
                         - max_x: maximum x-value (e.g. 2 pi)
                         - min_x: minimum x-value (e.g. 0)
                         - max_y: maximum y-value (e.g. 1 pi)
                         - min_y: minimum y-Value (e.g. 0 pi)

    :return np.arrays: corresponding x,y,z cartesian coordinates
    """
    polar_grid = grid_setup(params["height"], params["width"], params["min_x"],
                            params["max_x"], params["min_y"], params["max_y"])

    xyz_grid = transform_polar_to_xyz(polar_grid, params["radius"])
    return polar_grid, xyz_grid


# %%
def grid_setup(height, width, min_x, max_x, min_y, max_y):
    """
    Function to generate a two dimensional image of given
    height and width. A delta is defined based on the min
    and max values in x and y direction. To avoid sampling
    at the poles, this delta is shifted by 1/2.
    :return:
    """
    # Determine delta based on min/max and total size of grid
    delta_x = (max_x - min_x)/width

    # Shift corner conditions by 0.5 delta to avoid singularity issues
    max_x -= 0.5 * delta_x
    min_x += 0.5 * delta_x

    # Get polar coordinates image grid (theta, phi) and add z
    image_grid = np.mgrid[min_y:max_y:complex(height), min_x:max_x:complex(width)].reshape(2, -1)
    # image_grid = np.mgrid[min_x:max_x:complex(width), min_y:max_y:complex(height)].reshape(2, -1)

    return image_grid


# %%
def transform_polar_to_xyz(polar_coords, r):
    """
    Function to transform a given matrix in polar (spherical) coordinates (r, theta, phi)
    to cartesian coordinates (x, y, z). The calculations are as follows:

        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)

    with r = radius, theta = azimuthal angle (long) and phi = polar angle (lat)

    Certain conditions have to be met:
    r >= 0
    0 <= theta <= 2*pi (360 deg, 0 - 360)
    0 <= phi <= 1*pi (180 deg, 0 - 180)

    :param np.ndarray polar_coords: polar coords, dim = (2, v) with theta, phi
    :param float r: radius of the sphere (e.g. 100 for freesurfer)
    :return np.ndarray: cartesian coordinates, dim = (v, 3) with x, y, z
    """

    theta = polar_coords[0, :] #0-2pi
    phi = polar_coords[1, :]  #0-1pi

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    cartesian_coords = np.stack((x, y, z), axis=-1)
    return cartesian_coords


# %% load surface data
a = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
# data is the raw data 
data = a.read().splitlines()
# data_truc is the raw data without the header
data_truc = data[2:]
polar_grid, xyz_grid = get_polar_grid(params={"height": 512, "width": 256, "radius": 100, "max_y": 2 * np.pi, "min_y": 0, "max_x": np.pi, "min_x": 0})

# %% plot grid
import matplotlib.pyplot as plt
plt.plot(polar_grid[0,:],polar_grid[1,:], 'b.')
plt.show
# %% color_map_DK(annot_path)
import nibabel as nib
def color_map_DK(annot_path, origin_ij):
    """
    getting color map of Desikan-Killiany Atlas given annotation path and (original) ij_id (array)
    return color_vertices(149955), color_group_id(149955), color_group_name(149955), and the annotation file.
    myannot[0] # vertex data (ID is the same with 'label' data, but categorised in 36 types due to DK atlas)
    myannot[1] # RGB color table for all vertices, in total 36
    Return color_vertices, color_group_id, color_group_name, myannot
    """
    myannot = nib.freesurfer.io.read_annot(annot_path, orig_ids=False)

    # the id of DK_annot_RBG array is the id we saved before
    id_per_half = origin_ij.astype('int')
    # create the colored_grid for the original map. (Half sphere, per person)
    colored_grid_list_norm = []
    color_group_id = []
    color_group_name = []

    for m in range(len(id_per_half)):
        myvertex_color_id = myannot[0][id_per_half[m]]
        color_group_id.append(myvertex_color_id)
        color_name = myannot[2][myvertex_color_id]
        color_group_name.append(color_name)
        # No hemi sphere has nonzero transparancy value. check(myannot[1][:][:][3]==0), is all 0.
        mycolor = myannot[1][myvertex_color_id][:3]/255.0
        colored_grid_list_norm.append(mycolor)

    colored_grid_array_norm = np.asarray(colored_grid_list_norm)
    color_vertices = colored_grid_array_norm

    color_keys = color_group_id
    color_values = color_vertices
    # color_dict = dict(zip(color_keys, color_values)) # len = 35, no key = 4, corpuscallosum)
    # color_name_dict =  dict(zip(color_keys, color_group_name))
    return color_vertices, color_group_id, color_group_name, myannot
    # return color_dict, color_name_dict, color_keys, myannot
annot_path = 'rh.aparc.annot'
# %% plot Desikan-Killiany Atlas for the original (i,j) grid
# plt.scatter(i,j, c=C, marker='.') # this is without labels
c_vertices, c_group_id, c_group_name, myannot = color_map_DK(annot_path, ij_id)
scatter_x = polar_grid[0,:]
scatter_y = polar_grid[1,:]
c_dict = dict(zip(c_group_id, c_vertices)) # len = 35, no key = 4, corpuscallosum)
c_name_dict =  dict(zip(c_group_id, c_group_name))
fig, ax = plt.subplots()
for g in np.unique(c_group_id):
    ix = np.where(c_group_id == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = c_dict[g], label=c_name_dict[g], marker='.')
leg = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), title="DK_atlas_name")
ax.add_artist(leg)
c_name_sorted = list(c_name_dict.keys())
c_name_sorted.sort()
for idx, x in enumerate(c_name_sorted):
    x = str(x)
plt.legend(labels=c_name_sorted,loc='center right', bbox_to_anchor=(1.7, 0.5), title="DK_atlas_id")
plt.show()
# %% plot this xyz in 3d
import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x=xyz_grid[:,0], y=xyz_grid[:,1], z=xyz_grid[:,2],opacity=0.9, color=xyz_grid[:,0])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
# %%
# %% load surface data
a = open("lh.fsaverage.sphere.cortex.mask.label", "r")
# data is the raw data 
data = a.read().splitlines()
# data_truc is the raw data without the header
data_truc = data[2:]
print(len(data_truc))
# %%
