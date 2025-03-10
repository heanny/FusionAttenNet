"""
This class is used to transform the (x,y,z) coordinates in spherical surface 
to grid (i,j) 2-D image coordinate system. The main step for it is from paper
https://deep-mi.org/static/pub/henschel_2020b.pdf, Equation (1),

First, we use Equation (1) of Longitude/colatitudesphericalparameterization
to transfer (x,y,z) to (phi, theta) space (with radius r=100), and we sample
(phi, theta) to (i, j) 2D grid, and the only special case is at the pole point,
the theta uses the half of width of the grid. The (i, j) 2D map is 
filled with "thickness" value. The size is 769*195 because that
149955 vertices on the original hemi-sphere.

"""
# %% import
import numpy as np
from math import degrees, atan2, sqrt
import torch
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# TODO: Before using the big dataset, check whether the unknown ID are the same

# %% load surface data
a = open("GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
# data is the raw data 
data = a.read().splitlines()
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
data_arr = np.array(data_lines)

# %% easy check
# (label, x,y,z,d), starting from data[2], id=0
print(data[2])
# %% easy check
print(data_truc[0])
print(len(data_truc)) # 149955 = 3*5*13*769 = 195*769 (left brain)
# 149926 = 14 * 10709 (right brain)


# %% xyz_to_longtitudinal
# the function of (x,y,z) to (phi, theta), phi in (0, 2pi), polar angle theta in (0,pi), r = 100
"""
according to the equation (1) of the original paper, compute phi and theta given (x,y,z) with radius.

Check this page for equation:
https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.07%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).

return: (id, phi,theta) (phi and theta are radian values)
"""
def xyz_to_longtitudinal(xyz_data_id):
    """
    according to the equation (1) of the original paper, compute phi and theta given (x,y,z) with radius.

    Check this page for equation:
    https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.07%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).

    EQUATION (1) in paper
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi) (?? is that correct??)

    they said that they use colatitude, the complementary angle of latitude (phi)

    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    https://www.sciencedirect.com/topics/mathematics/spherical-polar-coordinate#:~:text=In%20spherical%20polar%20coordinates%2C%20the,the%20azimuthal%20angle%20(longitude).
    MY THOUGHTS: if theta is the polar angle, then: 
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    theta is the angle from the polar direction (on the Earth, colatitude, which is 90°-latitude)
    phi is longitude.

    r = sqrt(x**2+y**2+z**2)
    phi = sgn(y)*arccos(x/sqrt(x**2+y**2)) = atan2(y,x)
    theta = arccos(z/sqrt(x * x + y * y + z * z)) = arctan(y/x) or theta = arctan(sqrt(x * x + y * y)/ z) = atan2(qrt(x * x + y * y)/ z)

    return: (id, phi,theta) (phi and theta are radian values)
    id ranges from 0 to 163841, there are 149955 for left brain in total.
"""
    id = xyz_data_id[0]
    x = float(xyz_data_id[1])
    y = float(xyz_data_id[2])
    z = float(xyz_data_id[3])

    # represented in degrees. 
    # phi = degrees(atan2(y,x))
    # theta = degrees(atan2(sqrt(x * x + y * y), z))

    # represnted in radians
    phi = atan2(y,x)
    theta = atan2(sqrt(x * x + y * y), z)

    # theta_2 = np.arccos(z/sqrt(x * x + y * y + z * z))

    return [id, phi, theta] #original: [id, phi, theta]

# %% test for xyz_to_longtitudinal (for one vertex of a person)
# This is the test for one vertex of a person    
res = list(map(str.strip, data_truc[0].split()))
print("the xyz", res)
haha = xyz_to_longtitudinal(res)
haha

# %% get_longitudinal_map_each person
def get_longitudinal_map_each(each_xyz_data_id):
    """
    get the hemi-surface of each person, make the transformation of xyz to
    Longitude/colatitude space. Ruture the list of 149955 longitudinal/colatitunal
    representation of per person. 

    Return all_vertex_each (datatype: list)
    """
    all_vertex_each = [] 
    for i in range(len(each_xyz_data_id)):
        data_split = list(map(str.strip, each_xyz_data_id[i].split()))
        temp = xyz_to_longtitudinal(data_split)
        all_vertex_each.append(temp)

    return all_vertex_each

# %% test get_longitudinal_map_each
# This is the test for whole points 
haha_per = get_longitudinal_map_each(data_truc)
len(haha_per)




# %% get_ij_from_sphere for one vertex of per person
def get_ij_from_sphere(sphere_data_id, radius):
    """
    the function of sampling (phi, theta) to (i, j) grid
    Given a "mapping sphere" of radius R,
    the Mercator projection (x,y) of a given latitude and longitude is:
    i = R * longitude
    j = R * log( tan( (latitude + pi/2)/2 ) )
    theta is co-latitude (90° - latitude), phi is longitude.
    Return [id, i, j] in 2d-grid format.
    """
    id = sphere_data_id[0]
    theta = sphere_data_id[2]
    phi = sphere_data_id[1]

    i = radius * phi
    # j = radius * np.log(np.tan(((1.5708-theta) + (np.pi/2))/2)) 

    # at the pole point, to avoid the singularity, we set theta at half of the grid width (769*195): 195/2
    
    if theta == 0:
        j = 544 #FIXME: 313.9042675697948(this is the half of max(i)-min(i)) for left; 313.8991636636057 for right brain
    elif theta == np.pi:
        j = -544
    else:
        j = radius * np.log(np.tan(((1.5708-theta) + (np.pi/2))/2)) 
        # j = radius * np.log(np.tan(((theta) + (np.pi/2))/2)) 

    return [id, i, j]


hehe = get_ij_from_sphere(haha, 100)
hehe

# %% sphere_to_grid_each person (x,y,z)->(phi,theta)->(i,j)
"""
    the function of sampling (phi, theta) to (i, j) grid for per person of hemi-sphere
    Given sphere_cooridinates data
    Return the 769*195 grid
"""
def sphere_to_grid_each(longitudinal_each_person, radius):
    list_each_half = []
    for i in range(len(longitudinal_each_person)):
        list_each_half.append(get_ij_from_sphere(longitudinal_each_person[i], radius))

    grid_each_half = np.array(list_each_half, dtype="O")
    print(grid_each_half.shape)
    return list_each_half, grid_each_half


# %% get original (id,i,j) for each person (hemi), and matrix of original_ij(id,i,j)
origin_ij_list, origin_ij_grid = sphere_to_grid_each(haha_per,100)
origin_ij_list[:10]
# maintain an indexing array ij_id
ij_id = origin_ij_grid[:,0]
print(ij_id[:10])
i = origin_ij_grid[:,1]
j = origin_ij_grid[:,2]
print(i[:10], j[:10])
# %% i and j matrices with shape of 769*195 of each.
# make the ij grid to matrix with shape of 769*195 for i and j, respectively.
# i_re = i.reshape((769, 195))
# j_re = j.reshape((769, 195))
i_mx = np.asmatrix(origin_ij_grid[:,1].astype('float').reshape((14, 10709)))#shape(769, 195), (14, 10709)
j_mx = np.asmatrix(origin_ij_grid[:,2].astype('float').reshape((14, 10709)))#shape(769, 195), (14, 10709)
# i_mx = np.asmatrix(origin_ij_grid[:,1].astype('float'))#shape(769, 195), (14, 10709)
# j_mx = np.asmatrix(origin_ij_grid[:,2].astype('float'))#shape(769, 195), (14, 10709)

print(i_mx.shape,j_mx.shape)

# %%
# XXX: skip it when running!
# check the linear relationship of id, and appreantly they are not, 
# ID range in [0, 163841]
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# check_id = sampling_test_grid[:,0]
# x = []
# for i in range(len(check_id)):
#     x.append(int(check_id[i]))

# x_array = np.array(x)
# y = np.array([i for i in range(149956)])
# data = pd.DataFrame([x_array, y]).T
# data.columns = ['x', 'y']

# sns.lmplot(x="x", y="y", data=data, order=1)
# plt.ylabel('Target')
# plt.xlabel('Independent variable')

# %%
# XXX: slower, can skip!
# # plot the (i,j) grid!!
# i = origin_ij_grid[:,1]
# j = origin_ij_grid[:,2]
# data = pd.DataFrame([i, j]).T
# data.columns = ['x', 'y']

# sns.lmplot(x="x", y="y", data=data)
# plt.ylabel('j')
# plt.xlabel('i')


# %% plot the original uncolored 2D-ij-grid
# plot the original 2D-ij-grid
print(np.min(i_mx), np.max(i_mx))
print(np.min(j_mx), np.max(j_mx))

plt.plot(i_mx,j_mx, 'b.')
plt.show

# %% color_map_DK(annot_path)
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
annot_path = 'lh.aparc.annot'
# c_vertices, c_group_id, c_group_name, myannot = color_map_DK(annot_path, ij_id)

# %% XXX: wrong example: plot filtered i and j (keep j<= 300)
# filtered_id = []
# for a in range(len(origin_ij_grid)):
#     if np.abs(origin_ij_grid[a,2]) <= 300 and np.abs(origin_ij_grid[a,1]) <= 300:
#         filtered_id.append(a)
# filter_ij_grid = origin_ij_grid[filtered_id]
# filtered_id = np.asarray(filtered_id)
# c_vertices, c_group_id, c_group_name, myannot = color_map_DK(annot_path, filtered_id)
# scatter_x = filter_ij_grid[:,1]
# scatter_y = filter_ij_grid[:,2]
# c_dict = dict(zip(c_group_id, c_vertices)) # len = 35, no key = 4, corpuscallosum)
# c_name_dict =  dict(zip(c_group_id, c_group_name))
# fig, ax = plt.subplots()
# for g in np.unique(c_group_id):
#     ix = np.where(c_group_id == g)
#     ax.scatter(scatter_x[ix], scatter_y[ix], c = c_dict[g], label=c_name_dict[g], marker='.')
#     # ax2.scatter(scatter_x[ix], scatter_y[ix], c = c_dict[g], label=g, marker='.')
# leg = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), title="DK_atlas_name")
# ax.add_artist(leg)
# c_name_sorted = list(c_name_dict.keys())
# c_name_sorted.sort()
# for idx, x in enumerate(c_name_sorted):
#     x = str(x)
# plt.legend(labels=c_name_sorted,loc='center right', bbox_to_anchor=(1.7, 0.5), title="DK_atlas_id")
# plt.show()


# %% plot Desikan-Killiany Atlas for the original (i,j) grid
# plt.scatter(i,j, c=C, marker='.') # this is without labels
c_vertices, c_group_id, c_group_name, myannot = color_map_DK(annot_path, ij_id)
scatter_x = i
scatter_y = j
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



# %% Load SOI data and maintain the array of 769*195 for each SOI
# We maintain the array of 769*195 of SOI (signals of interests, thickness/surface area/volume.)
def SOI_array_per(ID_semi, SOI_path):
    """
    return the reshaped array for each SOI data for each person given the ID of 149955 vertices of semi sphere.
    """
    SOI_load = nib.load(SOI_path)
    SOI_raw = SOI_load.get_fdata() 
    # Get SOI data of given ID for per person (hemi-sphere)
    SOI_array_raw = SOI_raw[ID_semi]
    SOI_array_fs5 = SOI_array_raw[:10242]
    SOI_array_reshape = SOI_array_raw[:,0,0].reshape((769, 195))
    SOI_array_fs5_reshape = SOI_array_fs5[:,0,0].reshape((569, 18))
    return SOI_array_reshape, SOI_array_fs5_reshape

thick_path = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.thickness.fwhm10.fsaverage.mgh'
volume_path = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.volume.fwhm10.fsaverage.mgh'
SA_path = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf/lh.area.fwhm10.fsaverage.mgh'
# For ABCD, it could be the one with "-"
w_g_pct_path = '/home/jouyang/GenR_mri/sub-1_ses-F09/surf//lh.w-g.pct.mgh.fwhm10.fsaverage.mgh'

# ID array that can be indexed by range(149955)
ID_per_half = ij_id.astype('int')
thickness_array_re, thickness_array_re_fs5 = SOI_array_per(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
volume_array_re, volume_array_re_fs5 = SOI_array_per(ID_per_half, volume_path)   # volume in [0, 5.9636817]
SA_array_re, SA_array_re_fs5 = SOI_array_per(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
w_g_array_re, w_g_array_re_fs5 = SOI_array_per(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

print(thickness_array_re_fs5.shape)




# %% TODO: to delet after make the class. save ij_id for the model 
np.save('ij_id_rh.npy', ij_id)
id_loaded = np.load('ij_id_rh.npy', allow_pickle=True)
print(len(id_loaded))


# %% XXX: DONT run, this old method doesnt make sense, right?
# no w_g_per in SOI_array because it's in range [0,1]
SOI_array = np.concatenate((thickness_array_re, volume_array_re, SA_array_re), axis=-1)
SOI_norm = preprocessing.normalize(SOI_array, norm='l2') 
SOI_mx = SOI_norm.reshape((769, 780))
# SOI_mx = np.concatenate((thickness_mx, volume_mx, SA_mx, w_g_mx), axis=-1) # shape(769, 780)
# SOI_mx = np.vstack((thickness_mx, volume_mx, SA_mx, w_g_mx)) # = equal to concat (axis=0)
# SOI_mx = np.hstack((thickness_mx, volume_mx, SA_mx, w_g_mx)) # = equal to concat (axis=1)

# 
plt.imshow(SOI_mx, cmap='Greys') #FIXME: or vmin=0, vmax=1, to replace normlization??
plt.colorbar()
plt.show()

# %% stacked SOI (769, 195, 4) using min-max norm and plot it
def min_max_normalize(matrix):
    """
    function of min_max normalize for SOI data
    """
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    return normalized_matrix

# Perform min-max normalization
thickness_mx_norm = min_max_normalize(thickness_array_re)
volume_mx_norm = min_max_normalize(volume_array_re)
SA_mx_norm = min_max_normalize(SA_array_re)
w_g_ar_norm = w_g_array_re/100

# Perform min-max normalization for fs5 data too
thickness_mx_norm_fs5 = min_max_normalize(thickness_array_re_fs5)
volume_mx_norm_fs5 = min_max_normalize(volume_array_re_fs5)
SA_mx_norm_fs5 = min_max_normalize(SA_array_re_fs5)
w_g_ar_norm_fs5 = w_g_array_re_fs5/100

# Matrix_4 (white/grey matter ratio) with shape (769, 195), already in the range [0, 1].
# Stack the four normalized matrices along the third dimension
SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)
SOI_mx_minmax_fs5 = np.stack([thickness_mx_norm_fs5 , volume_mx_norm_fs5 , SA_mx_norm_fs5 , w_g_ar_norm_fs5 ], axis=-1)

# The resulting shape will be (769, 195, 4)
print(SOI_mx_minmax.shape)
print(SOI_mx_minmax_fs5.shape)

# print(normalized_matrices)
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# plot this 3d array of shape (769, 195, 4) normalized Value for fs7 data
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

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# plot normalized Value for fs5 data
for i in range(4):
    ax = axes[i]
    im = ax.imshow(SOI_mx_minmax_fs5[:, :, i], cmap='viridis')
    ax.set_title(f'Matrix {i + 1}')
    ax.axis('off')

    # Add color bar to each subplot
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value for fs5 data')

plt.tight_layout()
plt.show()

# %% XXX: plot 3D views for thickness of each vertex
import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x=scatter_x, y=scatter_y, z=thickness_array[:,0,0],
                    color=thickness_array[:,0,0],opacity=0.7)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

# I think matrix, but 769*195, and save thickness values in each point. 
# But ij coordinates range does not correspond to 769*195. Should we save by ID, or by ij coordinates??
# If using ij coordinates, how to round it??

# %% XXX: wrong example of sampling the 149955 vertices and plotting it
# sampling the 149955 vertices and plotting it
ij = origin_ij_grid[:,1:]
index = np.random.choice(149955, 149955)
sampling_ij = ij[index]
sampling_ij = sampling_ij.astype('float')
X = sampling_ij[:,0]
Y = sampling_ij[:,1]
plt.scatter(X,Y, c=c_vertices,marker='.')



# %% XXX: wrong sampling data from 35 regions
# XXX: sampling data from 35 regions (excl.: corpuscallosum, idx = 4 in myannot, myannot[2][4])
grouped_vertices_id = {}
prop_dict = {}
sampling_ij_region = {}
ij = origin_ij_grid[:,1:]
for g in np.unique(c_group_id):
    ix = np.where(c_group_id == g)
    grouped_vertices_id[g] = ix
for key, value in grouped_vertices_id.items():
    prop_dict[key] = len(list(filter(None, value[0])))
    index = np.random.choice(prop_dict[key], prop_dict[key])
    sampling_ij_region[key] = ij[index].astype('float')

fig1, ax1 = plt.subplots()
for key, value in sampling_ij_region.items():
    scatter_x_key = sampling_ij_region[key][:,0]
    scatter_y_key = sampling_ij_region[key][:,1]
    ax1.scatter(scatter_x_key, scatter_y_key, c = c_dict[key], label=c_name_dict[key], marker='.')
    # ax2.scatter(scatter_x[ix], scatter_y[ix], c = c_dict[g], label=g, marker='.')
leg = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), title="DK_atlas_name")
ax1.add_artist(leg)
c_name_sorted = list(c_name_dict.keys())
c_name_sorted.sort()
for idx, x in enumerate(c_name_sorted):
    x = str(x)
plt.legend(labels=c_name_sorted,loc='center right', bbox_to_anchor=(1.7, 0.5), title="DK_atlas_id")
plt.show()

# %% FIXME: This is the unfinished sampling function, wrong sample with plot...rm 
fig2, ax2 = plt.subplots()
scatter_x_key = sampling_ij_region[5][:,0]
scatter_y_key = sampling_ij_region[5][:,1]
ax2.scatter(scatter_x_key, scatter_y_key, c = c_dict[5], label=c_name_dict[5], marker='.')
leg = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), title="DK_atlas_name")
ax2.add_artist(leg)
c_name_sorted = list(c_name_dict.keys())
c_name_sorted.sort()
for idx, x in enumerate(c_name_sorted):
    x = str(x)
plt.legend(labels=c_name_sorted,loc='center right', bbox_to_anchor=(1.7, 0.5), title="DK_atlas_id")
plt.show()
# grouped_ij_id_dict = dict(zip(c_group, grouped_vertices_id))
# XXX: I found that the propotion can only be used within the range/surface_area of that part of brain, 
# otherwise, how do we sample?? 
def sampling_via_region():
    """
    This is the function used for sampling 149955 vertices in total. 
    
    """

    return