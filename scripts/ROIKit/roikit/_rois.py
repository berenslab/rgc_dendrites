import glob
import h5py
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import morphkit as mk
from matplotlib.backends.backend_pdf import PdfPages

from ._utils import *

__all__ = ['ROIs']

class ROIs:
    
    def __init__(self, cell_id, datapath='./export/manuscript/'):
                
        self.cell_id = cell_id
            
        datapath_scans_all = datapath + 'data_scans.pickle'
        with open(datapath_scans_all, 'rb') as f:
            data_scans_all = pickle.load(f)
            data_scans = data_scans_all[data_scans_all['cell_id'] == cell_id]
            data_scans.index = np.arange(len(data_scans))
                        
        datapath_stacks = datapath + 'data_stacks_info.pickle'
        with open(datapath_stacks, 'rb') as f:
            data_stacks = pickle.load(f)
            data_stack = data_stacks[data_stacks['cell_id'] == cell_id].squeeze() 
                        
        datapath_morph = datapath + f'morphologies/{cell_id}.swc'
        m = mk.Morph(datapath_morph)
        m.df_paths = m.df_paths.assign(path_stack=m.df_paths.path.apply(lambda x: np.round(x/data_stack['pixel_size']).astype(int)))
        data_paths = m.df_paths
        linestack = get_linestack(data_paths, data_stack['shape'])
        linestack_xy = get_linestack_xy(linestack)
        
        res = data_scans.apply(lambda x: get_matching_results(x, data_stack, data_paths, data_scans, linestack_xy), axis=1)
        data_scans = data_scans.assign(scan_center=res.apply(lambda x: x[0]),
                                       ROI_pos_stack=res.apply(lambda x: x[1]),
                                       figure_matched=res.apply(lambda x: x[2]))
                
        (self.data_stack, 
         self.data_paths, 
         self.data_scans, 
         self.linestack,
         self.linestack_xy) = data_stack, data_paths, data_scans, linestack, linestack_xy
        
        self.density_maps = m.density_maps
                
    def finetune(self, roi_id, pad_more=0, offset=[0, 0], angle_adjust=0):
    
        res = self.data_scans.loc[[roi_id]].apply(lambda x: get_matching_results(x, self.data_stack, self.data_paths, self.data_scans, self.linestack_xy, pad_more=pad_more, offset=offset, angle_adjust=angle_adjust), axis=1)
        self.data_scans.loc[[roi_id]] = self.data_scans.assign(scan_center=res.apply(lambda x: x[0]),
                                       ROI_pos_stack=res.apply(lambda x: x[1]),
                                       figure_matched=res.apply(lambda x: x[2]))
        
    def save_pdf(self, output_path='./export/ROIs_on_morph/'):
    
        cell_id = self.cell_id 
    
        with PdfPages(output_path + f'ROIs_on_morph_{cell_id}.pdf') as pp:
            for fig in self.data_scans['figure_matched'].values:
                pp.savefig(fig)
                
    def get_data_rois(self):
        
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