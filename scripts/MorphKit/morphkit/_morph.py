from ._utils import *
from ._summarize import *
from ._visualize import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['Morph']

class Morph:
    
    def __init__(self, filepath, verbal=0):
        
        self.unit = 'um'
        self.filename = filepath.split('/')[-1].split('.')[0].lower()
        self.filetype = filepath.split('/')[-1].split('.')[-1].lower()        
  
        df_paths = data_preprocessing(filepath)
        
        if verbal: print('  Calculating path statistics (e.g. real length, branching order...)')
        df_paths = get_path_statistics(df_paths)
        self.df_paths = df_paths
 
        if verbal: print('  Calculating density data...')
        df_density, density_maps = get_density_data(self.df_paths)
        self.density_maps = density_maps
        
        if verbal: print('  Calculating summary data...')
        self.df_summary = get_summary_data(self.df_paths)
        
        self.df_summary = pd.concat([self.df_summary, df_density[['asymmetry', 'radius', 'size']]], axis=1)
        
        
    def show_summary(self):

        """
        Print out summary statistics of the cell.

        Parameters
        ----------
        summary: pd.DataFrame
            a pandas DataFrame that contains summary of one type of neurites of the cell.
        """

        print('  Summary of the cell')
        print('  ======================\n')

        summary = self.df_summary.to_dict()
        # density = self.df_density.to_dict()

        for n in range(len(summary['type'])):

            neurite_type = summary['type'][n]
            num_path_segments = summary['num_path_segments'][n]
            num_branchpoints = summary['num_branchpoints'][n]
            num_irreducible_nodes = summary['num_irreducible_nodes'][n]
            max_branch_order = summary['max_branch_order'][n]
            average_nodal_angle_deg = summary['average_nodal_angle_deg'][n]
            average_nodal_angle_rad = summary['average_nodal_angle_rad'][n]
            average_local_angle_deg = summary['average_local_angle_deg'][n]
            average_local_angle_rad = summary['average_local_angle_rad'][n]
            average_tortuosity = summary['average_tortuosity'][n]
            real_length_sum = summary['real_length_sum'][n]
            real_length_mean = summary['real_length_mean'][n]
            real_length_median = summary['real_length_median'][n]
            real_length_min = summary['real_length_min'][n]
            real_length_max = summary['real_length_max'][n]
            euclidean_length_sum = summary['euclidean_length_sum'][n]
            euclidean_length_mean = summary['euclidean_length_mean'][n]
            euclidean_length_median = summary['euclidean_length_median'][n]
            euclidean_length_min = summary['euclidean_length_min'][n]
            euclidean_length_max = summary['euclidean_length_max'][n]



            print('  {}\n'.format(neurite_type).upper())
            print('    Number of arbor segments: {}'.format(num_path_segments))
            print('    Number of branch points: {}'.format(num_branchpoints))
            print('    Number of irreducible nodes: {}'.format(num_irreducible_nodes))
            print('    Max branching order: {}\n'.format(max_branch_order))


            asymmetry = summary['asymmetry'][n]
            radius = summary['radius'][n]
            fieldsize = summary['size'][n]

            print('    Asymmetry: {:.3f}'.format(asymmetry))
            print('    Radius: {:.3f}'.format(radius))
            print('    Field Area: {:.3f} ×10\u00b3 um\u00b2\n'.format(fieldsize / 1000))


            print('  ## Angle \n')
            print('    Average nodal angle in degree: {:.3f}'.format(average_nodal_angle_deg))
            print('    Average nodal angle in radian: {:.3f}'.format(average_nodal_angle_rad))
            print('    Average local angle in degree: {:.3f}'.format(average_local_angle_deg))
            print('    Average local angle in radian: {:.3f} \n'.format(average_local_angle_rad))

            print('  ## Average tortuosity: {:.3f}\n'.format(average_tortuosity))

            print('  ## Real length (μm)\n')

            print('    Sum: {:.3f}'.format(real_length_sum))
            print('    Mean: {:.3f}'.format(real_length_mean))
            print('    Median: {:.3f}'.format(real_length_median))
            print('    Min: {:.3f}'.format(real_length_min))
            print('    Max: {:.3f}\n'.format(real_length_max))

            print('  ## Euclidean length (μm)\n')

            print('    Sum: {:.3f}'.format(euclidean_length_sum))
            print('    Mean: {:.3f}'.format(euclidean_length_mean))
            print('    Median: {:.3f}'.format(euclidean_length_median))
            print('    Min: {:.3f}'.format(euclidean_length_min))
            print('    Max: {:.3f}\n'.format(euclidean_length_max))

            print('  ======================\n')


    ##################
    #### Plotting ####
    ##################

    def show_morph(self, view='xy', plot_axon=True, plot_basal_dendrites=True, plot_apical_dendrites=True, savefig=False, save_to='./output/img/'):

        """
        Plot cell morphology in one view.

        Parameters
        ----------
        view: str
            * top view: 'xy'
            * front view: 'xz'
            * side view: 'yz'
        plot_axon: bool
        plot_basal_dendrites: bool
        plot_apical_dendrites: bool
        save_fig: str or None
            If None, no figure is saved. 
            Otherwiese, figure is saved to the specified path.
        """

        df_paths = self.df_paths.copy()
        fig, ax = plt.subplots(1, 1, figsize=(12,12))
        ax = plot_morph(ax, df_paths, view, plot_axon, plot_basal_dendrites, plot_apical_dendrites)
        
        if savefig:
            figname = '{}_oneview.png'.format(self.filename)
            logging.info('  Saving {} to {}'.format(figname, save_to))
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            fig.savefig(save_to + figname)

        return fig, ax

    def show_threeviews(self, plot_axon=True, plot_basal_dendrites=True, plot_apical_dendrites=True, savefig=False, save_to='./output/img/'):
        """
        Plot cell morphology in three views.

        Parameters
        ----------
        plot_axon: bool
        plot_basal_dendrites: bool
        plot_apical_dendrites: bool
        save_fig: str or None
            If None, no figure is saved. 
            Otherwiese, figure is saved to the specified path.
        """

        df_paths = self.df_paths.copy()

        fig, ax = plt.subplots(1, 3, figsize=(18,6))
        # fig, ax = plt.subplots(1, 3, figsize=(10,3.5))

        ax0 = plot_morph(ax[0], df_paths, 'xy', plot_axon, plot_basal_dendrites, plot_apical_dendrites)
        ax1 = plot_morph(ax[1], df_paths, 'xz', plot_axon, plot_basal_dendrites, plot_apical_dendrites)
        ax2 = plot_morph(ax[2], df_paths, 'yz', plot_axon, plot_basal_dendrites, plot_apical_dendrites)
        
        if savefig:
            figname = '{}_threeviews.pdf'.format(self.filename)
            logging.info('  Saving {} to {}'.format(figname, save_to))
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            fig.savefig(save_to + figname)

        return fig, ax
        

        