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
# import
import numpy as np
from math import degrees, atan2, sqrt
import nibabel as nib
import matplotlib.pyplot as plt

# TODO: Before using the big dataset, check whether the unknown ID are the same


# xyz_to_longtitudinal
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

    EQUATION (1) in paper:
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi) (?? is that correct??)

    they said that they use colatitude, the complementary angle of latitude (phi)

    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    https://www.sciencedirect.com/topics/mathematics/spherical-polar-coordinate#:~:text=In%20spherical%20polar%20coordinates%2C%20the,the%20azimuthal%20angle%20(longitude).
    MY THOUGHTS: if theta is the polar angle, then: theta [0, pi], phi [0, 2pi)
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    theta is the angle from the polar direction (on the Earth, colatitude, which is 90°-latitude), polar angle.
    phi is longitude, azimuth angle.

    r = sqrt(x**2+y**2+z**2)
    phi = sgn(y)*arccos(x/sqrt(x**2+y**2)) = atan2(y,x)
    theta = arccos(z/sqrt(x * x + y * y + z * z)) = arctan(y/x) or theta = arctan(sqrt(x * x + y * y)/ z) = atan2(qrt(x * x + y * y)/ z)

    return: (id, phi,theta) (phi and theta are radian values)
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

    return [id, phi, theta]

# get_longitudinal_map_each person
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

#  get_ij_from_sphere for one vertex of per person
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
    
    if theta == 0 or theta == np.pi:
        j = 195/2 #FIXME: 313.9042675697948(this is the half of max(i)-min(i)), or 195/2 (half of the ij-grid width)
    else:
        j = radius * np.log(np.tan(((1.5708-theta) + (np.pi/2))/2)) 

    return [id, i, j]

#  sphere_to_grid_each person (x,y,z)->(phi,theta)->(i,j)
def sphere_to_grid_each(longitudinal_each_person, radius):
    """
    the function of sampling (phi, theta) to (i, j) grid for per person of hemi-sphere
    Given sphere_cooridinates data
    Return the 769*195 grid
    """
    list_each_half = []
    for i in range(len(longitudinal_each_person)):
        list_each_half.append(get_ij_from_sphere(longitudinal_each_person[i], radius))

    grid_each_half = np.array(list_each_half, dtype="O")
    print(grid_each_half.shape)
    return list_each_half, grid_each_half


# color_map_DK(annot_path)
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
        # TODO: Do other hemi spheres has nonzero transparancy value? check(myannot[1][:][3]==0), now is all 0
        mycolor = myannot[1][myvertex_color_id][:3]/255.0
        colored_grid_list_norm.append(mycolor)

    colored_grid_array_norm = np.asarray(colored_grid_list_norm)
    color_vertices = colored_grid_array_norm

    return color_vertices, color_group_id, color_group_name, myannot

# get original (id,i,j) for each person (hemi), and matrix of original_ij(id,i,j)
def plot_original(origin_ij_grid):
    #FIXME: the reshape needs changes on right half brain
    i_mx = np.asmatrix(origin_ij_grid[:,1].astype('float').reshape((769, 195)))#shape(769, 195)
    j_mx = np.asmatrix(origin_ij_grid[:,2].astype('float').reshape((769, 195)))#shape(769, 195)

    print(np.min(i_mx), np.max(i_mx))
    print(np.min(j_mx), np.max(j_mx))

    plt.plot(i_mx,j_mx, 'b.')
    plt.show

# plot Desikan-Killiany Atlas for the original (i,j) grid
def plot_DK_map(c_vertices, c_group_id, c_group_name, i, j):
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
