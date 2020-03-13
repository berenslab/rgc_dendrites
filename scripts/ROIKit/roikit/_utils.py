import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.transform import resize, rotate
from skimage.feature import match_template

from shapely.geometry import Polygon

from rfest import splineLG, build_design_matrix, get_spatial_and_temporal_filters
import cv2
import pandas as pd

def get_scale_factor(rec_pixel_size, stack_pixel_sizes):
    
    return rec_pixel_size / stack_pixel_sizes[0]


def resize_roi(data_scan, data_stack):
    
    output_shape = np.ceil(np.asarray(data_scan['ROIs'].shape) * get_scale_factor(data_scan['pixel_size'], data_stack['pixel_size'])).astype(int)
    
    return resize(data_scan['ROIs'], output_shape=output_shape, order=0, mode='constant')

def resize_rec(data_scan, data_stack):
    
    reci = data_scan['scan']
    reci[:4, :] = reci.mean() - 0.5*reci.std()
    
    output_shape = np.ceil(np.asarray(reci.shape) * get_scale_factor(data_scan['pixel_size'], data_stack['pixel_size'])).astype(int)

    return resize(reci, output_shape=output_shape, order=1, mode='constant')

def rotate_rec(data_scan, data_stack, angle_adjust=0):
    
    ang_deg = data_scan['angle'] + angle_adjust# ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratote angle (radian)
    
    rec_rec = resize_rec(data_scan, data_stack)
    rec_rot = rotate(rec_rec, ang_deg, resize=True, order=1, cval=rec_rec.min())
    
    (shift_x, shift_y) = 0.5 * (np.array(rec_rot.shape) - np.array(rec_rec.shape))
    (cx, cy) = 0.5 * np.array(rec_rec.shape)
    
    px, py = (0, 0) # origin
    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)    
    
    # the shifted origin after rotation
    
    return rec_rot, (xn, yn)

def rotate_roi(data_scan, data_stack, angle_adjust=0):
    
    ang_deg = data_scan['angle'] + angle_adjust # ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rois = resize_roi(data_scan, data_stack)
    rec_rois_rot = rotate(rec_rois, ang_deg, cval=1, order=0, resize=True)
    rec_rois_rot = np.ma.masked_where(rec_rois_rot == 1, rec_rois_rot)

    (shift_x, shift_y) = 0.5 * (np.array(rec_rois_rot.shape) - np.array(rec_rois.shape))
    (cx, cy) = 0.5 * np.array(rec_rois.shape)
    
    labels = np.unique(rec_rois)[:-1][::-1]

    px = [np.vstack(np.where(rec_rois == i)).T[:, 0].mean() for i in labels] 
    py = [np.vstack(np.where(rec_rois == i)).T[:, 1].mean() for i in labels]

    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)
    
    return rec_rois_rot, np.vstack([xn, yn]).T

def get_linestack(df_paths, stack_shape):
    coords = np.vstack(df_paths.path_stack)
    
    linestack = np.zeros(stack_shape)
    for c in coords:
        linestack[tuple(c)] = 1
        
    return linestack

def get_linestack_xy(linestack):
        
    linestack_xy = linestack.mean(2)
    linestack_xy[linestack_xy != 0] = 1
        
    return linestack_xy

def rel_position_um(ref, d):
    
    """
    Relative position between dendrites and soma in um.
    
    original order: (y,x,z)
    return order: (x,y,z)
    
    """
    
    relative_posistion = ref['position'] - d['position']
    
    return relative_posistion[[1, 0, 2]]

def roi_matching(image, template):
    
    result = match_template(image, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    
    return np.array(ij)

def get_matching_results(data_scan, data_stack, data_paths, data_scans, linestack_xy, **kwargs):
            
    offset = kwargs['offset'] if 'offset' in kwargs.keys()else [0, 0]
    pad_more = kwargs['pad_more'] if 'pad_more' in kwargs.keys() else 0
    angle_adjust = kwargs['angle_adjust'] if 'angle_adjust' in kwargs.keys()else 0
    
    d_rec = resize_rec(data_scan, data_stack)
    d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(data_scan, data_stack, angle_adjust)
    d_rois_rot, roi_coords_rot = rotate_roi(data_scan, data_stack, angle_adjust)   

    d_rel_cx, d_rel_cy, _ = rel_position_um(data_scans[data_scans['rec_id'] == 0].squeeze(), 
                                data_scan) if 0 in data_scans['rec_id'].values else rel_position_um(data_stack, 
                                data_scan)
    
    (stack_soma_cx, stack_soma_cy, _) = data_paths.loc[0]['path_stack'][0] if 0 in data_scans['rec_id'].values else [None, None, None]
    
    d_stack_cx = int(linestack_xy.shape[0]/2+d_rel_cx) if stack_soma_cx is None else int(stack_soma_cx+d_rel_cx)
    d_stack_cy = int(linestack_xy.shape[0]/2+d_rel_cy) if stack_soma_cy is None else int(stack_soma_cy+d_rel_cy) 

    padding = int(max(d_rec_rot.shape) + 25) + pad_more

    crop_x0, crop_x1 = np.maximum(0, d_stack_cx-padding), np.minimum(d_stack_cx+padding, linestack_xy.shape[0]-1)
    crop_y0, crop_y1 = np.maximum(0, d_stack_cy-padding), np.minimum(d_stack_cy+padding, linestack_xy.shape[0]-1)

    
    crop_x0 = np.maximum(crop_x0+offset[0], 0)
    crop_x1 = np.maximum(crop_x1+offset[0], 0)
    
    crop_y0 = np.maximum(crop_y0-offset[1], 0)
    crop_y1 = np.maximum(crop_y1-offset[1], 0)
    
    crop = linestack_xy[crop_x0:crop_x1, crop_y0:crop_y1]

    
    if data_scan['rec_id'] == 0:
        
        rec_center_stack_xy = data_paths.loc[0]['path_stack'][0][:-1]
        roi_coords_stack_xy = data_paths.loc[0]['path_stack'][0][:-1]

        rec_center_crop = rec_center_stack_xy - np.array([crop_x0, crop_y0])
        roi_coords_crop = (roi_coords_stack_xy - np.array([crop_x0, crop_y0])).reshape(1,2)
        
        d_rec_rot_x0, d_rec_rot_y0 = (rec_center_crop - np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2])).astype(int)
        roi_coords_rot = (roi_coords_crop - np.array([d_rec_rot_x0, d_rec_rot_y0])).reshape(1,2).astype(int)

    else:
    
        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot)
        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
    
    d_rec_rot_x0 = np.maximum(d_rec_rot_x0, 0)
    d_rec_rot_y0 = np.maximum(d_rec_rot_y0, 0)    
    
    d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=1)
    d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 1, d_rois_rot_crop)

    rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

    roi_coords_stack_xy = roi_coords_crop + np.array([crop_x0, crop_y0])

    d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((crop_x0, 0), (crop_y0, 0)), 
                                                  mode='constant', constant_values=1)
    d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 1, d_rois_rot_stack_xy)

    rec_center_stack_xy = rec_center_crop + np.array([crop_x0,crop_y0])

    d_coords_xy = np.round(roi_coords_stack_xy).astype(int)
        
    # plot

    fig = plt.figure(figsize=((8.27, 11.69)))

    ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
    ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

    # recording region
    ax1.imshow(d_rec_rot, origin='lower', cmap=plt.cm.binary)
    ax1.imshow(d_rois_rot, origin='lower', cmap=plt.cm.viridis)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange')
    for point_id, point in enumerate(roi_coords_rot):
        ax1.annotate(point_id+1, xy=point[::-1], xytext=point[::-1]-np.array([0, 2]), color='red')

    ax1.set_title('Recording Region', fontsize=12)

    # crop region
    ax2.imshow(crop, origin='lower', cmap=plt.cm.binary)
    h_d_rec_rot, w_d_rec_rot = d_rec.shape
    rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
    tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -data_scan['angle']) + ax2.transData
    rect_d_rec_rot.set_transform(tmp2)
    ax2.add_patch(rect_d_rec_rot)
    ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
    ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], color='orange')
    ax2.set_title('Cropped Region', fontsize=12)
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # whole region
    ax3.imshow(linestack_xy, origin='lower', cmap=plt.cm.binary)
    # ax3.scatter(self.soma[1]/self.pixel_sizes_stack[1], self.soma[0]/self.pixel_sizes_stack[0], s=120, marker='x')
    hd, wd = crop.shape
    rect_crop = mpl.patches.Rectangle((crop_y0, crop_x0), wd, hd, edgecolor='r', facecolor='none', linewidth=2)

    h_d_rec_rot, w_d_rec_rot = d_rec.shape
    rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + crop_y0+origin_shift_y, d_rec_rot_x0 + crop_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
    tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ crop_y0+origin_shift_y, d_rec_rot_x0+crop_x0+origin_shift_x, -data_scan['angle']) + ax3.transData
    rect_crop_d_rec.set_transform(tmp3)

    ax3.add_patch(rect_crop_d_rec)
    ax3.add_patch(rect_crop)
    ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
    ax3.scatter(roi_coords_crop[:, 1]+crop_y0, roi_coords_crop[:, 0]+crop_x0, color='orange')
    # ax3.annotate(dname, xy=(d_rec_rot_y0 + crop_y0-10, d_rec_rot_x0 + crop_x0-10), color='white')
    ax3.set_title('ROIs on Cell Morpholoy', fontsize=12)
    ax3.grid(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0,linestack_xy.shape[0])
    ax3.set_ylim(0,linestack_xy.shape[0])

    cell_id , rec_id = data_scan['cell_id'], data_scan['rec_id']

    plt.suptitle(f'{cell_id}: d{rec_id:02d}', fontsize=18)

    plt.close()
    
    return rec_center_stack_xy, d_coords_xy, fig


def calibrate_one_roi(coords_xy, linestack):
    
    x_o = coords_xy[0] # roi_x_original
    y_o = coords_xy[1] # roi_y_original

    search_scope = 0
    offset = np.where(linestack[x_o:x_o+1, y_o:y_o+1] == 1)

    while offset[2].size == 0:
        search_scope +=1
        offset = np.where(linestack[x_o-search_scope:x_o+search_scope+1, y_o-search_scope:y_o+search_scope+1] == 1)

    z_o = np.mean(offset[2]).astype(int)  # roi_z_original, this is a guess

    x_c = np.arange(x_o-search_scope,x_o+search_scope+1)[offset[0]]
    y_c = np.arange(y_o-search_scope,y_o+search_scope+1)[offset[1]]
    z_c = offset[2]

    candidates = np.array([np.array([x_c[i], y_c[i], z_c[i]]) for i in range(len(x_c))])
    origins = np.array([x_o, y_o, z_o])

    x, y, z = candidates[np.argmin(np.sum((candidates - origins) ** 2, 1))]

    return np.array([x,y,z])

def on_which_path(df_paths, point):
    
    result_path = df_paths[df_paths.path_stack.apply(lambda x: (x == point).all(1).any())]
    path_id = result_path.index[0]
    
    return path_id

def get_loc_on_path_stack(df_paths, point):
    
    loc = [i for i in df_paths.path_stack.apply(lambda x: np.where((x == point).all(1))[0]) if len(i) != 0][0]
    
    return loc[0]

def get_segment_length(arr):
    
    return np.sum(np.sqrt(np.sum((arr[1:] - arr[:-1])**2, 1)))

def get_dendritic_distance_to_soma(df_paths, path_id, loc_on_path):
    
    length_all_paths = sum(df_paths.loc[df_paths.loc[path_id]['back_to_soma']]['real_length'])
    length_to_reduce = get_segment_length(df_paths.loc[path_id].path[loc_on_path:])
    
    return length_all_paths-length_to_reduce

def get_euclidean_distance_to_one_point(roi_pos, point_pos):
    
    return np.sqrt(np.sum((roi_pos - point_pos) ** 2))

def get_density_center(df_paths, soma, Z):    
    
    from scipy.ndimage.measurements import center_of_mass
    def change_coordinate(coords, origin_size, new_size):

        coords_new = np.array(coords) * (max(new_size) - min(new_size)) / max(origin_size) - max(new_size) 

        return coords_new

    xy = np.vstack(df_paths.path)[:, :2] - soma[:2]
    lim_max = int(np.ceil((xy.T).max() / 20) * 20)
    lim_min = int(np.floor((xy.T).min() / 20) * 20)
    lim = max(abs(lim_max), abs(lim_min))

    density_center = center_of_mass(Z)[::-1]
    density_center = change_coordinate(density_center, (0, max(Z.shape)), (-lim, lim))
    density_center += soma[:2]
    
    return density_center

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_btw_node(roi_0_pos, roi_1_pos, node):
    
    v0 = unit_vector(roi_0_pos - node)
    v1 = unit_vector(roi_1_pos - node)
    
    return np.degrees(np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0)))

def get_cntr_interception(roi_0_cntr, roi_1_cntr):
    
    inner_cntr_list_all = []
    overlap_index_all = []
    for cntr0 in roi_0_cntr:
        for cntr1 in roi_1_cntr:
            if cntr0.shape[0] > cntr1.shape[0]:
                sCntr = cntr1.copy()
                bCntr = cntr0.copy()
            else:
                sCntr = cntr0.copy()
                bCntr = cntr1.copy()

            sCntrPoly = Polygon(sCntr)
            bCntrPoly = Polygon(bCntr)

            sCntr_area = sCntrPoly.area
            bCntr_area = bCntrPoly.area

            check_intercept =  sCntrPoly.intersects(bCntrPoly)

            if check_intercept:

                CntrInpt = sCntrPoly.intersection(bCntrPoly)

                if CntrInpt.type == 'Polygon':
                    inner_cntr_list = [np.asarray(CntrInpt.boundary.coords)]
                    overlap_area = CntrInpt.area
                elif CntrInpt.type == 'MultiPolygon':
                    inner_cntr_list = []
                    overlap_area = 0
                    for i in np.arange(len(CntrInpt.geoms)):
                        inner_cntr_list.append(np.asarray(CntrInpt.geoms[i].boundary.coords))
                        overlap_area += CntrInpt.geoms[i].area
                        
            else:
                inner_cntr_list = [np.nan]
                overlap_area = 0
                
            overlap_index_all.append(overlap_area/sCntr_area)
            inner_cntr_list_all.append(inner_cntr_list)
            
    return inner_cntr_list_all, np.mean(overlap_index_all)


def znorm(data):
    """
    Normalizing raw trace.
    """
    return (data - data.mean())/data.std()

def interpolate_weights(tracetime, traces_znorm, triggers):
    
    """
    Align the stimulus time and triggertime, 
    aka downsampling the raw trace to the same length as the stimulus.
    """
    
    from scipy import interpolate
    data_interp = interpolate.interp1d(
        tracetime.flatten(), 
        traces_znorm.flatten(),
        kind = 'linear'
    ) (triggers)
    
    return znorm(data_interp)

def get_rf(*data, X, dims):
    #     (rec_id, roi_id, tracetime, triggertime, traces_raw) = data
    rec_id = data[0]
    roi_id = data[1]
    tracetime = data[2]
    triggertime = data[3]
    traces_raw = data[4]
        
    y = interpolate_weights(tracetime, 
                                  znorm(traces_raw.flatten()), 
                                  triggertime)


    y = y[:1500]
    y = np.gradient(y) # take the derivative of the calcium trace
    X = X[:len(X), :]    
    
    spl = splineLG(X, y, dims=dims, df=7)
    sRF, tRF = get_spatial_and_temporal_filters(spl.w_spl, dims)
        
    return [sRF, tRF, spl.w_spl]

def upsample_rf(rf, rf_pixel_size, stack_pixel_size):
    from skimage.transform import resize
    scale_factor = rf_pixel_size/stack_pixel_size
    output_shape = (np.array([15, 20]) * scale_factor).astype(int)
    
    return resize(rf, output_shape=output_shape, mode='constant')

def rescale_data(data):
    return (data - data.min()) / (data.max() - data.min())

def get_contour(data, stack_pixel_size, 
                rf_pixel_size=30):

    data = rescale_data(data)
    levels = np.arange(60, 75, 5)/100
    
    CS = plt.contour(data, levels=levels)
    plt.clf()
    plt.close()

    res = 0
    for i in range(len(levels)):

        ps = CS.collections[i].get_paths()
        
        all_cntrs = [p.vertices for p in ps]

        cntrs_size = [cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 for cntr in all_cntrs]

        # if i == 0:
        #     res = pd.Series([all_cntrs, cntrs_size])
        # else:
        #     tmp = pd.Series([all_cntrs, cntrs_size])
        #     res = res.append(tmp)
        
        good_cntrs = [cntr[:, ::-1] for cntr in all_cntrs if (cntr[0] == cntr[-1]).all() and cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 > 2.5]
        good_cntrs_size = [cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 for cntr in good_cntrs]

        if i == 0:
            res = pd.Series([good_cntrs, good_cntrs_size])
        else:
            tmp = pd.Series([good_cntrs, good_cntrs_size])
            res = res.append(tmp)

    res.index = np.arange(len(levels) * 2)
    return res


def get_irregular_index(cnts):
    irregular_index = []
    
    if len(cnts) == 0: return 1
    
    for j, cnt in enumerate(cnts):
        hull = cv2.convexHull(cnt.astype(np.float32)).flatten().reshape(-1, 2)
        hull = np.vstack([hull, hull[0]])
        
        RFarea = cv2.contourArea(cnt.astype(np.float32))
        CHarea = cv2.contourArea(hull.astype(np.float32))

        irregular_index.append((CHarea - RFarea) / CHarea)
            
    return np.max(irregular_index)