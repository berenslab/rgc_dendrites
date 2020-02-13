import numpy as np
import pandas as pd

def unique_row(a):

    """
    Returns an array of the ordered, unique set of rows for input array a.

    Parameters
    ----------
    a: array
        an array with replicated rows.

    returns
    -------
    unique_a: array
        an ordered array without replicated rows.

    example
    -------
    >>> a = np.array([[9,9],
                      [8,8],
                      [1,1],
                      [9,9]])
    >>> unique_row(a)
    >>> array([[1, 1],
               [8, 8],
               [9, 9]])
    """

    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]

    return unique_a

def get_path_real_length(path):

    """
    Get the dendritic length of a path, which is the sum of the distance between each consecutive points.

    Parameters
    ----------
    path: array
        a coordinate array with dim=(n, 3)

    Returns
    -------
    the dendritic length of this path: float

    """

    return np.sum(np.sqrt(np.sum((path[1:] - path[:-1])**2, 1)))


def get_path_euclidean_length(path):
    """
    get the euclidean length of a path, which is the distance between the first and last points.

    Parameters
    ----------
    path: array
        a coordinate array with dim=(n, 3)

    Returns
    -------
    the euclidean length of this path: float

    """
    return np.sqrt(np.sum((path[0] - path[-1]) ** 2))


def get_outer_terminals(all_terminals):

    """
    Get terminal points which form the convex hull of the cell.

    Parameters
    ----------
    all_terminals: array
        The array contains all terminal points from terminal paths (no other paths connected to them)

    Returns
    -------
    outer_terminals_3d: array
        The array contains all terminal points which found the convex hull of the cell.

    """

    from scipy.spatial import ConvexHull
    hull = ConvexHull(all_terminals[:,:2])
    outer_terminals_3d = all_terminals[hull.vertices]
    outer_terminals_3d = np.vstack([outer_terminals_3d, outer_terminals_3d[0]])

    return outer_terminals_3d

def get_angle(v0, v1):

    """
    Get angle (in both radian and degree) between two vectors.

    Parameters
    ----------
    v0: array
        vector zero.
    v1: array
        vector one.

    Returns
    -------
    Return a tuple, (angle in radian, angle in degree).

    """
    v0 = np.array(v0)
    v1 = np.array(v1)

    if not v0.any() or not v1.any():
        return 0, 0

    c = np.dot(v0, v1) / np.linalg.norm(v0) / np.linalg.norm(v1)
    return np.arccos(np.clip(c, -1, 1)), np.degrees(np.arccos(np.clip(c, -1, 1)))


def get_remote_vector(path):

    """
    Get vector of certain path between the first and the last point.

    Parameters
    ----------
    df_paths: pandas.DataFrame

    path_id: int

    Returns
    -------
    normalized v: array
        returned a normalized vector.
    """

    s = path[0]
    e = path[-1]
    v= e-s

    if (v == 0).all():
        return np.zeros(3)
    else:
        return v/np.linalg.norm(v)

def get_local_vector(path):

    """
    Get vector of certain path between the first and the second point.

    Parameters
    ----------
    df_paths: pandas.DataFrame

    path_id: int

    Returns
    -------
    normalized v: array
        returned a normalized vector.
    """

    s = path[0]
    e = path[1]
    v= e-s

    if (v == 0).all():
        return np.zeros(3)
    else:
        return v/np.linalg.norm(v)

def get_average_angles(df_paths):
    """
    a helper function to get the average of all kinds of angles.

    Parameters
    ----------
    df_paths: pandas.DataFrame

    Returns
    -------
    average_nodal_angle_deg
    average_nodal_angle_rad
    average_local_angle_deg
    average_local_angle_rad

    """

    nodal_angles_deg = {}
    nodal_angles_rad = {}

    local_angles_deg = {}
    local_angles_rad = {}

    n = 0
    for i in np.unique(df_paths.connect_to):

        path_ids = df_paths[df_paths.connect_to == i].index.tolist()

        if len(path_ids) >= 2:

            from itertools import combinations
            path_id_combs = combinations(path_ids, 2)
            
            for path_id_pair in path_id_combs:

                p0 = df_paths.loc[path_id_pair[0]].path
                p1 = df_paths.loc[path_id_pair[1]].path

                v00 = get_remote_vector(p0)
                v01 = get_remote_vector(p1)
                nodal_angles_rad[n], nodal_angles_deg[n] = get_angle(v00, v01)

                v10 = get_local_vector(p0)
                v11 = get_local_vector(p1)
                local_angles_rad[n], local_angles_deg[n] = get_angle(v10, v11)

                n+=1
        else:
            continue

    average_nodal_angle_deg = np.nanmean(list(nodal_angles_deg.values()))
    average_nodal_angle_rad = np.nanmean(list(nodal_angles_rad.values()))

    average_local_angle_deg = np.nanmean(list(local_angles_deg.values()))
    average_local_angle_rad = np.nanmean(list(local_angles_rad.values()))

    return average_nodal_angle_deg, average_nodal_angle_rad, average_local_angle_deg, average_local_angle_rad

def get_summary_of_type(df_paths):

    """
    A helper function to gather all summarized infomation

    Parameters
    ----------
    df_paths: pandas.DataFrame

    Return
    ------
    a list of all summarized information.

    """

    if len(df_paths) < 1:
        return None

    branchpoints = np.vstack(df_paths.connect_to_at)
    branchpoints = unique_row(branchpoints)
    num_branchpoints = len(branchpoints)

    max_branch_order = max(df_paths.branch_order)

    terminalpaths = df_paths.path[df_paths.connected_by.apply(len) == 0].values
    terminalpoints = np.vstack([p[-1] for p in terminalpaths])
    num_terminalpoints = len(terminalpoints)

    # outerterminals = get_outer_terminals(terminalpoints)

    num_irreducible_nodes = num_branchpoints + num_terminalpoints

    num_dendritic_segments = len(df_paths)

    # path length

    reallength = df_paths['real_length']
    reallength_sum = reallength.sum()
    reallength_mean = reallength.mean()
    reallength_median = reallength.median()
    reallength_min = reallength.min()
    reallength_max = reallength.max()

    euclidean = df_paths['euclidean_length']
    euclidean_sum = euclidean.sum()
    euclidean_mean = euclidean.mean()
    euclidean_median = euclidean.median()
    euclidean_min = euclidean.min()
    euclidean_max = euclidean.max()

    tortuosity = reallength / euclidean
    average_tortuosity = np.mean(tortuosity)

    # node angles
    average_nodal_angle_deg, average_nodal_angle_rad, average_local_angle_deg, average_local_angle_rad = get_average_angles(df_paths)

    if df_paths.iloc[0].type == 2:
        t = 'axon'
    elif df_paths.iloc[0].type == 3:
        t = 'basal_dendrites'
    elif df_paths.iloc[0].type == 4:
        t = 'apical_dendrites'
    else:
        t = 'undefined'

    return (t,int(num_dendritic_segments),
            int(num_branchpoints),
            int(num_irreducible_nodes),
            int(max_branch_order),
            average_nodal_angle_deg,
            average_nodal_angle_rad,
            average_local_angle_deg,
            average_local_angle_rad,
            average_tortuosity,
            reallength_sum,
            reallength_mean,
            reallength_median,
            reallength_min,
            reallength_max,
            euclidean_sum,
            euclidean_mean,
            euclidean_median,
            euclidean_min,
            euclidean_max,)

def get_summary_data(df_paths):

    """
    The summary of the cell morphology.
    """

    df_paths = df_paths.copy()

    soma = df_paths[df_paths.type == 1]
    axon = df_paths[df_paths.type == 2]
    dend_basal = df_paths[df_paths.type == 3]
    dend_apical = df_paths[df_paths.type == 4]

    axon_summary = get_summary_of_type(axon)
    dend_basal_summary = get_summary_of_type(dend_basal)
    dend_apical_summary = get_summary_of_type(dend_apical)

    labels = [
            'type',
            'num_path_segments',
            'num_branchpoints',
            'num_irreducible_nodes',
            'max_branch_order',
            'average_nodal_angle_deg',
            'average_nodal_angle_rad',
            'average_local_angle_deg',
            'average_local_angle_rad',
            'average_tortuosity',
            'real_length_sum',
            'real_length_mean',
            'real_length_median',
            'real_length_min',
            'real_length_max',
            'euclidean_length_sum',
            'euclidean_length_mean',
            'euclidean_length_median',
            'euclidean_length_min',
            'euclidean_length_max',
            ]

    neurites = [axon_summary,dend_basal_summary,dend_apical_summary]
    df_summary = pd.DataFrame.from_records([n for n in neurites if n is not None], columns=labels)

    return df_summary
    
def get_path_statistics(df_paths):
    """

    Add path statistics (e.g. real/euclidean length of each path, ordering, index of paths back to soma...)

    Parameters
    ==========
    df_paths

    Returns
    =======
    a updated df_paths
    """


    df_paths = df_paths.copy()

    all_keys = df_paths.index

    real_length_dict = {}
    euclidean_length_dict = {}
    back_to_soma_dict = {}
    branch_order_dict = {}

    for path_id in all_keys:

        path = df_paths.loc[path_id].path

        real_length_dict[path_id] = get_path_real_length(path)
        euclidean_length_dict[path_id] = get_path_euclidean_length(path)
        branch_order_dict[path_id] = len(df_paths.loc[path_id].back_to_soma) - 1

    df_paths['real_length'] = pd.Series(real_length_dict)
    df_paths['euclidean_length'] = pd.Series(euclidean_length_dict)
    df_paths['branch_order'] = pd.Series(branch_order_dict)

    return df_paths

def get_density_data_of_type(neurites, soma):

    """
    A helper function to gether all summarized infomation

    Parameters
    ----------
    neurites: pandas.DataFrame
    soma: pandas.DataFrame

    Returns
    -------
    result tuple: 
        (type, asymmetry, neurites_radius, neurites_size)

    Z: numpy.array
        the density map of neurites, a (100, 100) matrix
        
    """
    
    if len(neurites) < 2:
        return None, None
    
    import cv2
    from scipy.stats import gaussian_kde
    from scipy.spatial import ConvexHull
    from scipy.ndimage.measurements import center_of_mass
    
    soma_coords = soma.path.values[0].flatten()
    xy = (np.vstack(neurites.path)[:, :2] - soma_coords[:2]).T
    kernel = gaussian_kde(xy, bw_method='silverman')

    lim_max = int(np.ceil((xy.T).max() / 20) * 20)
    lim_min = int(np.floor((xy.T).min() / 20) * 20)
    lim = max(abs(lim_max), abs(lim_min))
    X, Y = np.mgrid[-lim:lim:100j, -lim:lim:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.flipud(np.rot90(np.reshape(kernel(positions).T, X.shape)))

    density_center = np.array(center_of_mass(Z))
    density_center = density_center * (2 * lim) / Z.shape[0] - lim
    
    asymmetry = np.sqrt(np.sum(density_center ** 2))

    hull = ConvexHull(xy.T)
    outer_terminals = xy.T[hull.vertices]
    outer_terminals = np.vstack([outer_terminals, outer_terminals[0]])
    neurites_radius = np.mean(np.sqrt(np.sum((outer_terminals - density_center)**2, 1)))
    neurites_size = cv2.contourArea(outer_terminals.astype(np.float32))

    if neurites.iloc[0].type == 2:
        t = 'axon'
    elif neurites.iloc[0].type == 3:
        t = 'basal_dendrites'
    elif neurites.iloc[0].type == 4:
        t = 'apical_dendrites'
    else:
        t = 'undefined'

    return (t, asymmetry, neurites_radius, neurites_size), Z

def get_density_data(df_paths):

    """
    A helper function to gether all summarized infomation

    Parameters
    ----------
    df_paths: pandas.DataFrame

    Returns
    -------
    df_density: pandas.DataFrame

    density_maps: numpy.array
        a (3, 100, 100) matrix, each layer is a density map of one neurites type 
        (0: axon; 1: basal dendrites; 2: apical dendrites)

    """

    df_paths = df_paths.copy()

    soma = df_paths[df_paths.type == 1]
    axon = df_paths[df_paths.type == 2]
    dend_basal = df_paths[df_paths.type == 3]
    dend_apical = df_paths[df_paths.type == 4]

    axon_density_summary, axon_density_map = get_density_data_of_type(axon, soma)
    dend_basal_density_data, dend_basal_density_map = get_density_data_of_type(dend_basal, soma)
    dend_apical_density_data, dend_apical_density_map = get_density_data_of_type(dend_apical, soma)

    density_maps = np.zeros([3, 100, 100])
    density_maps[0] = axon_density_map
    density_maps[1] = dend_basal_density_map
    density_maps[2] = dend_apical_density_map

    labels = ['type', 'asymmetry', 'radius', 'size']
    neurites = [axon_density_summary,dend_basal_density_data,dend_apical_density_data]
    df_density = pd.DataFrame.from_records([n for n in neurites if n is not None], columns=labels)

    return df_density, density_maps
    