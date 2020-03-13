import glob
import h5py
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import morphkit as mk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar

import cv2

from ._utils import *

__all__ = ['ROIs']

class ROIs:
    
    def __init__(self, cell_id, datapath='../data/'):
                
        self.cell_id = cell_id
            
        datapath_scans_all = datapath + 'raw/data_scans.pickle'
        with open(datapath_scans_all, 'rb') as f:
            data_scans_all = pickle.load(f)
            data_scans = data_scans_all[data_scans_all['cell_id'] == cell_id]
            data_scans.index = np.arange(len(data_scans))
                        
        datapath_stacks_all = datapath + 'raw/data_stacks_info.pickle'
        with open(datapath_stacks_all, 'rb') as f:
            data_stacks_all = pickle.load(f)
            data_stack = data_stacks_all[data_stacks_all['cell_id'] == cell_id].squeeze() 
            
        datapath_raw_all = datapath + 'raw/data_raw.pickle'
        with open(datapath_raw_all, 'rb') as f:
            data_rawtraces_all = pickle.load(f)
            data_rawtraces = data_rawtraces_all[data_rawtraces_all['expdate'] == cell_id]
            data_rawtraces.index = np.arange(len(data_rawtraces))
            
        datapath_morph_all = datapath + 'raw/data_morph.pickle'
        with open(datapath_morph_all, 'rb') as f:
            data_morph_all = pickle.load(f)
            data_paths = data_morph_all[data_morph_all['expdate'] == cell_id]
            data_paths.index = np.arange(len(data_paths))  
            
        datapath_rois_all = datapath + 'raw/data_roi.pickle'
        with open(datapath_rois_all, 'rb') as f:
            data_rois_all = pickle.load(f)
            data_rois = data_rois_all[data_rois_all['expdate'] == cell_id]
            data_rois.index = np.arange(len(data_rois))        
                         
        datapath_cntr_all = datapath + 'raw/data_cntr.pickle'
        with open(datapath_cntr_all, 'rb') as f:
            data_cntr_all = pickle.load(f)
            data_cntr = data_cntr_all[data_cntr_all['expdate'] == cell_id]
            data_cntr.index = np.arange(len(data_cntr))        
                         
                
        linestack = get_linestack(data_paths, data_stack['shape'])
        linestack_xy = get_linestack_xy(linestack)
       
        self.data_rawtraces = data_rawtraces
        self.data_stack = data_stack
        self.data_paths = data_paths
        self.data_scans = data_scans
        self.data_rois = data_rois
        self.data_cntr = data_cntr
        self.linestack = linestack
        self.linestack_xy  = linestack_xy
        
        soma_row = self.data_paths[self.data_paths.type == 1]
        if len(soma_row) == 0:
            self.soma = self.data_paths.loc[0].path[0].flatten()
        else:
            self.soma = soma_row.path[0].flatten()
            
        

    def compute_data_morph(self):
      
        datapath_morph = datapath + f'/morphologies/{self.cell_id}.swc'
        m = mk.Morph(datapath_morph)
        m.df_paths = m.df_paths.assign(path_stack=m.df_paths.path.apply(lambda x: np.round(x/data_stack['pixel_size']).astype(int)))
        data_paths = m.df_paths
        linestack = get_linestack(data_paths, data_stack['shape'])
        linestack_xy = get_linestack_xy(linestack)
                
        self.density_maps = m.density_maps
        
    def compute_scan_matching(self):
        
        res = data_scans.apply(lambda x: get_matching_results(x, data_stack, data_paths, data_scans, linestack_xy), axis=1)
        data_scans = data_scans.assign(scan_center=res.apply(lambda x: x[0]),
                                       ROI_pos_stack=res.apply(lambda x: x[1]),
                                       figure_matched=res.apply(lambda x: x[2]))
        
        self.data_scans = data_scans
            
    def finetune_scan_matching(self, roi_id, pad_more=0, offset=[0, 0], angle_adjust=0):
    
        res = self.data_scans.loc[[roi_id]].apply(lambda x: get_matching_results(x, self.data_stack, self.data_paths, self.data_scans, self.linestack_xy, pad_more=pad_more, offset=offset, angle_adjust=angle_adjust), axis=1)
        self.data_scans.loc[[roi_id]] = self.data_scans.assign(scan_center=res.apply(lambda x: x[0]),
                                       ROI_pos_stack=res.apply(lambda x: x[1]),
                                       figure_matched=res.apply(lambda x: x[2]))
        
    def save_pdf(self, output_path='./export/ROIs_on_morph/'):
    
        cell_id = self.cell_id 
    
        with PdfPages(output_path + f'ROIs_on_morph_{cell_id}.pdf') as pp:
            for fig in self.data_scans['figure_matched'].values:
                pp.savefig(fig)
                
    def compute_data_rois(self):
        
        data_scans = self.data_scans
        data_paths = self.data_paths
        linestack = self.linestack
        
        dict_cell_id = {}
        dict_rec_id = {}
        dict_roi_id = {}
        
        dict_scan_center = {}
        dict_roi_pos_stack = {}
        
        counter = 0
        for row in data_scans.iterrows():

            cell_id = row[1]['cell_id']
            rec_id = row[1]['rec_id']
            scan_center = row[1]['scan_center']

            ROI_pos_stack = row[1]['ROI_pos_stack']

            for ii in range(len(ROI_pos_stack)):

                pos = ROI_pos_stack[ii]

                dict_cell_id[counter] = cell_id
                dict_rec_id[counter] = rec_id
                dict_roi_id[counter] = ii
                
                dict_scan_center[counter] = scan_center
                dict_roi_pos_stack[counter] = pos

                counter += 1

        data_rois = pd.DataFrame()
        data_rois['cell_id'] = pd.Series(dict_cell_id)
        data_rois['rec_id'] = pd.Series(dict_rec_id)
        data_rois['roi_id'] = pd.Series(dict_roi_id)
        data_rois['scan_center'] = pd.Series(dict_scan_center)
        data_rois['ROI_pos_stack'] = pd.Series(dict_roi_pos_stack)    
        
        
        # Find ROI position on pixel coordinate stack (x, y, z)
        data_rois = data_rois.assign(
                ROI_pos_stack=data_rois['ROI_pos_stack'].apply(
                lambda x: calibrate_one_roi(x, linestack)))

        # Find the path each ROI is on
        data_rois = data_rois.assign(
                path_id=data_rois['ROI_pos_stack'].apply(
                lambda x: on_which_path(data_paths, x)))

        # Find the location of each ROI on its coresponding path 
        data_rois = data_rois.assign(
                loc_on_path=data_rois['ROI_pos_stack'].apply(
                lambda x: get_loc_on_path_stack(data_paths, x)))

        # Find ROI pos in real length coordinate. 
        data_rois = data_rois.assign(ROI_pos=pd.Series(
                    {
                        row[0]: data_paths.path.loc[row[1]['path_id']][row[1]['loc_on_path']] 
                        for row in data_rois.iterrows()
                    }
                    )
                )

        # Get dendritic distance from ROI to soma
        data_rois = data_rois.assign(dendritic_distance_to_soma=pd.Series(
                { 
                        row[0]: get_dendritic_distance_to_soma(
                            data_paths, row[1]['path_id'],row[1]['loc_on_path'])
                        for row in data_rois.iterrows()
                }
                )
            )

        # Get euclidean distance from ROI to soma
        data_rois = data_rois.assign(
                euclidean_distance_to_soma=data_rois['ROI_pos'].apply(
                lambda x: get_euclidean_distance_to_one_point(x, data_rois.loc[0]['ROI_pos'])))

        # Get euclidean distance from ROI to dendritic density center

        density_center = get_density_center(data_paths, data_rois.loc[0]['ROI_pos'], self.density_maps[1])
        data_rois = data_rois.assign(
                euclidean_distance_to_density_center=data_rois['ROI_pos'].apply(
                lambda x: get_euclidean_distance_to_one_point(x[:2], density_center)))        

        # Get number of branchpoints from ROI to soma
        data_rois = data_rois.assign(
                branches_to_soma=data_rois['path_id'].apply(
                lambda x: np.array(data_paths.loc[x]['back_to_soma']).astype(int)))

        self.data_rois = data_rois
        
        
    def compute_rf(self, recompute=False):
        
        if hasattr(self, 'data_rf') and recompute is False:
            print('`data_rf` is already computed. To recompute, set `recompute=True`.')
        
        if recompute:
        
            noise_columns = ['rec_id', 'roi_id', 'Tracetimes0_noise',
                           'Triggertimes_noise','Traces0_raw_noise']

            stim = h5py.File('../data/raw/noise.h5', 'r')['k'][:].T
            dims = [5, 20, 15]
            X = build_design_matrix(stim, dims[0])

            data_rf = pd.DataFrame()
            data_rf[['recording_id', 'roi_id']] = self.data_rawtraces[['rec_id', 'roi_id']]
            rf_all =self.data_rawtraces[noise_columns].apply(lambda x: get_rf(*x, X=X, dims=dims), axis=1)
            data_rf['RF'] = rf_all.apply(lambda x:x[2])
            data_rf['sRF'] = rf_all.apply(lambda x:x[0])
            data_rf['tRF'] = rf_all.apply(lambda x:x[1])
            data_rf['RF_upsampled'] = data_rf['sRF'].apply(lambda x: upsample_rf(x, 30, self.data_stack['pixel_size'][0]))

            self.data_rf = data_rf
    
    def compute_cntr(self, recompute=False):
        raise NotImplementedError('This method needs to be updated.')
        
#         if hasattr(self, 'data_cntr') and recompute is False:
#             print('`data_cntr` is already computed. To recompute, set `recompute=True`.')
            
#         if recompute:
            
#             import itertools

#             data_cntr = pd.DataFrame()
#             data_cntr[['recording_id', 'roi_id']] = self.data_rawtraces[['rec_id', 'roi_id']]

#             # levels = np.linspace(0, 1, 41)[::2][10:-6]
#             # levels = np.arange(0.6, 0.72, 0.025)
#             levels = np.arange(60, 75, 5)/ 100
#             labels = [['RF_cntr_upsampled_{0}'.format(int(lev * 100)), 'sRF_asd_upsampled_cntr_size_{0}'.format(int(lev * 100)) ] for lev in levels]
#             labels = list(itertools.chain(*labels))

#             data_cntr[labels] = self.data_rf['RF_upsampled'].apply(lambda x: get_contour(x, self.data_stack['pixel_size'][0], 30))   

#             rfcenter =np.array([15,20]) * int(30) * 0.5
#             padding = self.data_rois.recording_center.apply(
#                 lambda x: (rfcenter-np.array(x) * self.data_stack['pixel_size'][0]).astype(int)
#             )
            

#             for lev in np.arange(60, 75, 5):

#                 data_cntr['cntr_irregularity_{}'.format(lev)] = data_cntr['RF_cntr_upsampled_{}'.format(lev)].apply(lambda x: get_irregular_index(x))
#                 data_cntr['cntr_counts_{}'.format(lev)] = data_cntr['RF_cntr_upsampled_{}'.format(lev)].apply(lambda x: len(x))
#                 data_cntr['cntr_quality_{}'.format(lev)] = np.logical_and(data_cntr['cntr_counts_{}'.format(lev)] < 2, 
#                                                                         data_cntr['cntr_irregularity_{}'.format(lev)] < 0.1)

#                 res = []
#                 for j, roi_contours in enumerate(data_cntr['RF_cntr_upsampled_{}'.format(lev)]):
#                     res.append([x * self.data_stack['pixel_size'][0] - padding[j] for x in roi_contours]) 
#                 data_cntr['RF_cntr_tree_upsampled_{}'.format(lev)] = pd.Series(res) 

#                 quality = data_cntr['cntr_quality_{}'.format(lev)].copy()
#                 all_cntrs = data_cntr['RF_cntr_tree_upsampled_{}'.format(lev)][quality]
#                 all_cntrs_center = all_cntrs.apply(lambda x: [np.mean(y,0) for y in x][0])
#                 rois_pos = np.vstack(self.data_rois.roi_pos)[:, :2][quality]
#                 rois_offsets = np.vstack(all_cntrs_center) - rois_pos
#                 rois_offset = rois_offsets.mean(0)
#                 cntrs_calibrate_to_rois = all_cntrs.apply(lambda x: [y - rois_offset for y in x])

#                 data_cntr['RF_cntr_calibrated_{}'.format(lev)] = cntrs_calibrate_to_rois

#                 data_cntr['cntrs_offset_{}'.format(lev)] = data_cntr['RF_cntr_calibrated_{}'.format(lev)][data_cntr['RF_cntr_calibrated_{}'.format(lev)].notnull()].apply(lambda x: np.array([y.mean(0) for y in x])) - self.data_rois.roi_pos.apply(lambda x:x[:2])
#                 data_cntr['distance_from_RF_center_to_soma_{}'.format(lev)] = data_cntr['RF_cntr_tree_upsampled_{}'.format(lev)][data_cntr['RF_cntr_tree_upsampled_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum((y.mean(0) - self.soma[:2])**2)) for y in x]))
#                 data_cntr['distance_from_RF_center_to_ROI_{}'.format(lev)] = data_cntr['cntrs_offset_{}'.format(lev)][data_cntr['cntrs_offset_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum(y**2)) for y in x]))    



#             data_cntr['RF_cntr_upsampled'] = data_cntr['RF_cntr_upsampled_65'] 
#             data_cntr['cntrs_offset_'] = data_cntr['cntrs_offset_65'] 
#             data_cntr['distance_from_RF_center_to_soma'] = data_cntr['distance_from_RF_center_to_soma_65'] 
#             data_cntr['distance_from_RF_center_to_ROI'] = data_cntr['distance_from_RF_center_to_ROI_65'] 

#             data_cntr['cntr_quality'] = data_cntr[[
#                                   'cntr_quality_60', 
#                                   'cntr_quality_65', 
#                                   'cntr_quality_70'
#                                   ]].all(1)     

#             self.data_cntr = data_cntr
        

    def compute_pairwise(self):
        
        from itertools import combinations
        
        cntr_quality = self.data_cntr['cntr_quality'].loc[1:].values # exclude soma
        
        print('\nStart calculating pairwise data.\n')
        
        df_rois = self.data_rois.loc[cntr_quality]
        df_cntr = self.data_cntr.loc[1:].loc[cntr_quality] 
        df_paths = self.data_paths
          
        
        soma = self.soma
        
        total_num_pairs = np.sum(np.arange(len(df_rois)))
        print('  {} pairs of ROIs are being processing.\n'.format(total_num_pairs))
        
        pair_ids = combinations(df_cntr.index - 1, 2)
        
        column_labels = ('pair_id', 'euclidian_distance_between_rois', 'dendritic_distance_between_rois',
                             'euclidian_distance_to_soma_sum', 'dendritic_distance_to_soma_sum',
                             'cbpt_angle_between_rois_deg', 'soma_angle_between_rois_deg',
                             'overlap_cntr','overlap_index')  
        
        df_pairs = pd.DataFrame(columns=column_labels)
        
        for pair_row_id, (roi_0, roi_1) in enumerate(pair_ids):

            # logging.info('  {}: {} {}'.format(pair_row_id, roi_0, roi_1))
            every_ten = int(total_num_pairs / 10)
            if every_ten == 0:
                print(' ({:04d}/{:04d}) Processing pair ({} {})...'.format(pair_row_id, total_num_pairs,roi_0, roi_1))     
            elif pair_row_id % every_ten == 0:
                print(' ({:04d}/{:04d}) Processing pair ({} {})...'.format(pair_row_id, total_num_pairs,roi_0, roi_1))            

            roi_0_pos = df_rois.loc[roi_0].roi_pos
            roi_1_pos = df_rois.loc[roi_1].roi_pos

            roi_0_branches = set(df_rois.loc[roi_0].branches_to_soma)
            roi_1_branches = set(df_rois.loc[roi_1].branches_to_soma)

            roi_0_dend_dist = df_rois.loc[roi_0].dendritic_distance_to_soma
            roi_1_dend_dist = df_rois.loc[roi_1].dendritic_distance_to_soma

            roi_0_eucl_dist = df_rois.loc[roi_0].euclidean_distance_to_soma
            roi_1_eucl_dist = df_rois.loc[roi_1].euclidean_distance_to_soma

            roi_0_cntr = df_cntr.loc[roi_0+1]['RF_cntr_upsampled']
            roi_1_cntr = df_cntr.loc[roi_1+1]['RF_cntr_upsampled']

            # paths interection and nonoverlap
            interection = roi_0_branches & roi_1_branches
            nonintercet = roi_0_branches ^ roi_1_branches

            dist_overlap = np.sum(df_paths.loc[interection].real_length)

            # dendritic distance between rois
            if roi_0_branches <= roi_1_branches or roi_1_branches <= roi_0_branches:

                dendritic_distance_btw = abs(roi_0_dend_dist - roi_1_dend_dist)

                list_branches = [roi_0_branches,roi_1_branches]
                shorter = list_branches[np.argmin(list(map(len, list_branches)))]
                cbpt = df_paths.loc[np.argmax(df_paths.loc[shorter].back_to_soma.apply(len))].path[-1]
                cbpt_angle = angle_btw_node(roi_0_pos, roi_1_pos, cbpt)

            else:

                dendritic_distance_btw = roi_0_dend_dist + roi_1_dend_dist - 2*dist_overlap

                if len(interection)>0:
                    cbpt = df_paths.loc[np.argmax(df_paths.loc[interection].back_to_soma.apply(len))].path[-1]
                else:
                    cbpt = soma

                cbpt_angle = angle_btw_node(roi_0_pos, roi_1_pos, cbpt)

            # euclidean distance bwetween rois

            euclidean_distance_btw = np.linalg.norm(roi_0_pos - roi_1_pos)

            # sum euclidian distance to soma 
            euclidean_distance_to_soma = roi_0_eucl_dist + roi_1_eucl_dist

            # sum dendritic distance to soma
            dendritic_distance_to_soma = roi_0_dend_dist + roi_1_dend_dist

            # angle between via soma
            soma_angle = angle_btw_node(roi_0_pos, roi_1_pos, soma)

            # rf overlap

            # inner_cntr_list, sCntr_area, bCntr_area, overlap_area, overlap_index = get_cntr_interception(roi_0_cntr, roi_1_cntr)
            inner_cntr_list, overlap_index = get_cntr_interception(roi_0_cntr, roi_1_cntr)
            # store restults to dataframe
            df_pairs.loc[pair_row_id] = [(roi_0, roi_1), euclidean_distance_btw, dendritic_distance_btw,
                                     euclidean_distance_to_soma, dendritic_distance_to_soma,
                                    cbpt_angle, soma_angle,
                                    inner_cntr_list, overlap_index]

        print('  Done.\n')
        
        self.data_pairs = df_pairs
        
        
    def plot_rois(self, roi_max_distance=300):
        
        fig = plt.figure(figsize=(8.27,8.27))

        ax1 = plt.subplot2grid((4,4), (0,1), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid((4,4), (3,1), rowspan=1, colspan=3)
        ax4 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=1)
        
        soma_pos = self.soma
        dendrites = self.data_paths[self.data_paths.type == 3]
        
        stack_pixel_size = self.data_stack['pixel_size']
        stack_shape = self.data_stack['shape'] 
        maxlim0, _, maxlim1 = stack_shape * stack_pixel_size
        
        for row in dendrites.iterrows():

            path_id = row[0]
            path = row[1]['path']
            ax1.plot(path[:, 0], path[:, 1], color='black')
            ax2.plot(path[:, 2], path[:, 1], color='black')
            ax3.plot(path[:, 0], path[:, 2], color='black')   
            
        rois_pos = np.vstack(self.data_rois.roi_pos)
        rois_dis = self.data_rois.dendritic_distance_to_soma.values

        # soma
        ax1.scatter(soma_pos[0], soma_pos[1], c='grey', s=160, zorder=10)
        ax2.scatter(soma_pos[2], soma_pos[1], c='grey', s=160, zorder=10)
        ax3.scatter(soma_pos[0], soma_pos[2], c='grey', s=160, zorder=10)

        sc = ax1.scatter(rois_pos[:, 0], rois_pos[:, 1], c=rois_dis, s=40, 
                         cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
        cbar = plt.colorbar(sc, ax=ax1, fraction=0.02, pad=.01 )
        cbar.outline.set_visible(False)

        ax2.scatter(rois_pos[:, 2], rois_pos[:, 1], c=rois_dis, s=40 * 0.8, 
                    cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
        ax3.scatter(rois_pos[:, 0], rois_pos[:, 2], c=rois_dis, s=40 * 0.8, 
                    cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)

        ax1.set_xlim(0, 350)
        ax1.set_ylim(0, 350)
        
        ax2.set_xlim(0, maxlim1)
        ax2.set_ylim(0, 350)
        
        ax3.set_xlim(0, 350)
        ax3.set_ylim(0, maxlim1)
        
        ax1.invert_yaxis()
        ax2.invert_yaxis()
#         ax3.invert_yaxis()
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        ax1.axis('off')
        scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=4)
        ax1.add_artist(scalebar)

        plt.suptitle('ROIs on morph')
        
    def plot_cntr(self, roi_max_distance=300, padding=50):
        
        soma_pos = self.soma
        dendrites = self.data_paths[self.data_paths.type == 3]   

        rois_pos = np.vstack(self.data_rois.roi_pos)
        rois_dis = self.data_rois.dendritic_distance_to_soma.values

        colors = np.vstack(plt.cm.viridis((rois_dis / roi_max_distance * 255).astype(int)))[:, :3]

        fig, ax = plt.subplots(figsize=(8,8))
        
        quality = self.data_cntr['cntr_quality'].values
                
        for row in dendrites.iterrows():

            path = row[1]['path']
            ax.plot(path[:, 0], path[:, 1], color='gray')
    
        for row in self.data_cntr.iterrows():

            idx = row[0]
            idx -= 1

            if idx < 0: continue
            
            distance = self.data_rois.loc[idx]['dendritic_distance_to_soma']
    
            if row[1]['cntr_quality']:

                cntr = row[1]['RF_cntr_upsampled'][0]
                ax.plot(cntr[:, 0], cntr[:, 1], color=colors[idx])
                ax.scatter(rois_pos[idx, 0], rois_pos[idx, 1], color=colors[idx], zorder=10)

                
        stack_pixel_size = self.data_stack['pixel_size']
        stack_shape = self.data_stack['shape']         
        max_lim = (stack_shape * stack_pixel_size)[0]
        max_lim = np.maximum(max_lim, 350)
        ax.set_xlim(-padding, max_lim)
        ax.set_ylim(-padding, max_lim)
        
        scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=0)
        ax.add_artist(scalebar)        
        
        ax.invert_yaxis()
        ax.axis('off')
        ax.axis('equal')