# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:44:47 2019

@author: Eichenbaum lab
"""
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import params
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.base.rois import register_multisession, extract_active_components
import cv2
import numpy as np
import os
import tifffile as TIF
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csc_matrix
import logging
from skimage import measure


def load_results(directory):
    """
    If Caiman has already been run, load results.

    Parameters
    ---
    directory: str
        Path to folder containing caiman_outputs.pkl.

    """

    with open(os.path.join(directory, 'caiman_outputs.pkl'), 'rb') as file:
        data = pickle.load(file)
        
    return data

class CaimanWrapper:
    """
    CaimanWrapper is a class designed to operate around the CaImAn package, 
    allowing fast and efficient utilization of the available tools.

    Parameters
    ---
    raw_dir: str
        Folder containing raw tif files.

    destination: str
        Folder where you want to copy cropped movies and caiman results.

    spatial_downsample: int, default: 2
        Factor to spatially downsample.

    do_crop: boolean, default: True
        Flag to crop movies.

    is_inscopix: boolean, default: True
        Flag whether to do Inscopix file-specific ordering changes.

    do_motion_correct: boolean, default: True
        Flag whether to perform motion correction.

    skip_file_transfer: boolean, default: False
        Flag whether to copy movie files to destination. Typically only used if movies are moved to destination
        beforehand. Note that crop coordinates will not be properly assigned in this case.

    """
    def __init__(self, raw_dir, destination,
                 spatial_downsample=2,
                 do_crop=True,
                 is_inscopix=True,
                 do_motion_correct=True,
                 skip_file_transfer=False):
        self.raw_dir = raw_dir
        self.destination = destination
        self.spatial_downsample = spatial_downsample
        self.do_crop = do_crop
        self.is_inscopix = is_inscopix
        self.do_motion_correct = do_motion_correct
        self.skip_file_transfer = skip_file_transfer
        
        # This is necessary for some reason.
        try:
            cv2.setNumThreads(0)
        except:
            pass
        
        self.get_std_map()
        plt.pause(5)        # Necessary because IDEs are dumb.
        
        # Crop and spatially downsample files.
        if not self.skip_file_transfer:
            self.fnames, self.crop_coords = self.crop_and_downsample()
        else:
            self.fnames = [os.path.join(self.destination, f) 
                           for f in os.listdir(self.destination) 
                           if f.endswith('.tif')]
            
            if self.do_crop: 
                self.crop_coords = (0, 
                                    self.std_map.shape[1], 
                                    0,
                                    self.std_map.shape[0])
            else:
                base_dir = os.path.dirname(self.destination)
                with open(os.path.join(base_dir, 'crop_coords.pkl'), 'rb') as f:
                    self.crop_coords = pickle.load(f)
            
            if self.is_inscopix:
                self.fnames.insert(0, self.fnames.pop())
                
    def quick_run(self):
        """
        Runs both motion correct and CNMF-E back to back with default settings.
        """
        
        self.run_phase1()
        self.run_phase2()
    
    #%%
    def run_phase1(self, skip_inspection=False, **kwargs):
        """
        Sets parameters, runs motion correction, and displays summary images

        Parameters
        ---
        skip_inspection: boolean, default: False
            Skips inspect_summary() call. Not recommended, since this gives you the cn_filter image.

        **kwargs: key, value pairings
            Runs phase 1 with these parameters changed from default. See init_mc_params().

        """
        # Set parameters. 
        self.opts = self.init_mc_params(**kwargs)
        
        # Set up cluster.
        self.c, self.dview, self.n_processes = self.setup_cluster()
        
        # Do motion correction and save memory mapped file. 
        if self.do_motion_correct:
            print('Performing motion correction. This may take a while.')
            self.motion_correct()
        else:  # if no motion correction just memory map the file
            print('Skipping motion correction, getting memory mapped file.')
            self.fname_new = [os.path.join(self.destination, f)
                              for f in os.listdir(self.destination)
                              if (f.endswith('.mmap') and f.startswith('full_'))][0]
            self.bord_px = 0
            
        # Load the memory mapped file and plot the summary images. 
        self.load_memmap_file()
        
        self.init_cnm_params()
        if not skip_inspection:
            self.inspect_summary()
            print('Inspect summary images to select new min_corr or min_pnr if applicable')
        
    def run_phase2(self, **kwargs):
        """
        Run CNMF-E and eliminate bad neurons.

        Parameters
        ---
        **kwargs: key, value pairings
            Runs phase 2 with these parameters changed from defaults. See init_cnm_params().

        """
        # Change parameters based on plots. 
        self.change_params(**kwargs)
        
        # Run CNMF-E.
        self.fit_cnmfe()
        
        # Evaluate outputs. 
        self.eval_components()

        
    #%%
    def get_std_map(self):
        """
        Plot the standard deviation map to get crop coordinates.
        """
        # Get all tifs.
        files = [os.path.join(self.raw_dir, f) 
                 for f in os.listdir(self.raw_dir) 
                 if (f.endswith('.tif') and f.startswith('recording'))]
        
        # Select one and plot the standard deviation map.
        tif = TIF.TiffFile(files[0]).asarray()
        self.std_map = np.std(tif[::2],0)
        plt.imshow(self.std_map)
        

    def crop_and_downsample(self):
        """
        Crop and spatially downsample the raw files before proceeding.
        """
        # Get raw file names. 
        files_to_process = [os.path.join(self.raw_dir, f) 
                            for f in os.listdir(self.raw_dir) 
                            if (f.endswith('.tif') and f.startswith('recording'))]
        
        # First, check if past sessions saved a crop coordinate file. 
        # It is important to have the same coordinates for each session so that
        # registration works. 
        base_dir = os.path.dirname(self.destination)
        try:
            with open(os.path.join(base_dir, 'crop_coords.pkl'), 'rb') as f:
                crop_coords = pickle.load(f)
                
                y0, y1, x0, x1 = crop_coords
                
            print('Previous crop coordinates found.')
            print('Cropping with ' + str(crop_coords))
        except:
            print('No previous crop coordinates located. Define new ones.')
            
            if self.do_crop:
                # Get coordinates to crop.
                y0 = int(input('Crop from here (left x position): '))
                y1 = int(input('Crop to here (right x position): '))
                x0 = int(input('Crop from here (top y position): '))
                x1 = int(input('Crop to here (bottom y position): '))
            else:
                y0 = 0
                y1 = self.std_map.shape[1]
                x0 = 0
                x1 = self.std_map.shape[0]
                
            crop_coords = (y0, y1, x0, x1)
            
            # Save crop coordinates.
            print('Saving crop coordinates...')
            with open(os.path.join(base_dir, 'crop_coords.pkl'), 'wb') as f:
                pickle.dump(crop_coords, f)
        
        print('Cropping...this may take a while.')
        # crop each file and save to destination.
        for file in files_to_process:
            tif = TIF.TiffFile(file).asarray()
            
            cropped = tif[:, 
                          x0:x1:self.spatial_downsample, 
                          y0:y1:self.spatial_downsample]
            
            tif_name = os.path.split(file)[-1]
            TIF.imsave(os.path.join(self.destination, 'cropped_'+tif_name), cropped)
        
        # Get the new cropped file names.
        fnames = [os.path.join(self.destination, f) 
                  for f in os.listdir(self.destination) 
                  if (f.endswith('.tif') and f.startswith('cropped'))]
        
        # Inscopix saves the first file without a numeric tag 
        # (e.g., 000, 001) so it gets sorted to last place. This line brings 
        # the first file to the front. 
        if self.is_inscopix:
            fnames.insert(0, fnames.pop())
            
        return fnames, crop_coords
    
    #%%
    def init_mc_params(self, **kwargs):
        """
        Define parameters.

        """
        #Build the dict. These are defaults.
        params_dict = {
                'fnames': self.fnames,      # file names
                'fr': 20,               # frame rate
                'decay_time': .4,       # transient decay time
                'pw_rigid': False,      # flag for performing piecewise-rigid motion correction (otherwise just rigid)
                'max_shifts': (5,5),    # maximum allowed rigid shift
                'gSig_filt': (3,3),     # size of high pass spatial filtering, used in 1p data
                'strides': (48,48),     # start a new patch for pw-rigid motion correction every x pixels
                'overlaps': (24,24),    # overlap between pathes (size of patch strides+overlaps)
                'max_deviation_rigid': 3, # maximum deviation allowed for patch with respect to rigid shifts
                'border_nan': 'copy',   # replicate values along the boundaries
                }
        
        # If any parameters were modified from default, this line reflects those changes.
        for key, value in kwargs.items():
            params_dict[key] = value
        
        opts = params.CNMFParams(params_dict = params_dict)
        
        return opts
    

    def init_cnm_params(self,**kwargs):
        """
        Initializes the CNMF parameters.

        """
        
        params_dict={
                'method_init': 'corr_pnr',
                'K': None,          # upper bound on number of components per patch, in general None
                'gSig': (3,3),      # gaussian width of a 2D gaussian kernel, which approximates a neuron
                'gSiz': (13,13),    # average diameter of a neuron, in general 4*gSig+1
                'merge_thr': .4,    # merging threshold, max correlation allowed
                'p': 1,             # order of the autoregressive system
                'tsub': 2,          # downsampling factor in time for initialization
                'ssub': 1,          # downsampling factor in space for initialization
                'rf': 40,           # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80 
                'stride': 12,       # amount of overlap between the patches in pixels
                                    # (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
                'only_init': True,  
                'nb': 0,            # number of background components (rank) if positive
                'nb_patch': 0,      # number of background components (rank) per patch if gnb>0,
                                    # else it is set automatically
                'method_deconvolution': 'oasis',
                'low_rank_background': None,    # None leaves background of each patch intact,
                                                # True performs global low-rank approximation if gnb>0
                'update_background_components': True,
                'min_corr': .8,     # min peak value from correlation image
                'min_pnr': 15,      # min peak to noise ration from PNR image
                'normalize_init': False,
                'center_psf': True, # True for 1-photon imaging. 
                'ssub_B': 2,        # additional downsampling factor in space for background
                'ring_size_factor': 1.4,    # radius of ring is gSiz*ring_size_factor
                'del_duplicates': True,
                'border_pix': self.bord_px}
        
        for key, value in kwargs.items():
            params_dict[key] = value
        
        self.opts.change_params(params_dict)


    def change_params(self, **kwargs):
        """
        Change specified parameters (e.g., session.change_params(merge_thr=0.3))

        """
        mod_params = {key: value for key, value in kwargs.items()}
        
        self.opts.change_params(mod_params)
            
        
    #%%
    def setup_cluster(self):
        """
        Set up the cluster for doing stuff fast.

        """
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)
        
        return c, dview, n_processes
    
    #%%
    def motion_correct(self):
        """
        Do motion correction.

        """
        # Set up motion correction object and load parameters. 
        mc = MotionCorrect(self.fnames, 
                           dview=self.dview, 
                           **self.opts.get_group('motion'))
        
        # Do motion correction. 
        mc.motion_correct(save_movie=True)
        
        # Plot motion correction statistics. 
        fname_mc = mc.fname_tot_els if self.opts.motion['pw_rigid'] else mc.fname_tot_rig
        if self.opts.motion['pw_rigid']:
            self.bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                              np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            self.bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
    
        # Save memory map. 
        self.bord_px = 0 if self.opts.motion['border_nan'] is 'copy' else self.bord_px
        self.fname_new = cm.save_memmap(fname_mc, base_name='full', order='C',
                                        border_to_0=self.bord_px)
    
    #%%
    def load_memmap_file(self):
        """
        Load the memory map file.

        """
        Yr, dims, T = cm.load_memmap(self.fname_new)
        self.images = Yr.T.reshape((T,) + dims, order = 'F')
            
    #%%
    def inspect_summary(self):
        """
        Plot and inspect the local correlations and peak to noise ratio. It's 
        not explicitly mentioned how to optimize these parameters, but I 
        believe you should be adjusting them until you see only neuron outlines.

        """
        self.cn_filter, pnr = cm.summary_images.correlation_pnr(self.images[::2], 
                                                                gSig=self.opts.init['gSig'][0],
                                                                swap_dim=False)
        inspect_correlation_pnr(self.cn_filter, pnr)
        
    #%%
    def fit_cnmfe(self):
        """
        Do CNMF-E

        """
        self.cnm = cnmf.CNMF(n_processes=self.n_processes,
                             dview=self.dview,
                             Ain=None,
                             params=self.opts)
        self.cnm.fit(self.images)
                
    #%%
    def eval_components(self, min_SNR=3, r_values_min=0.85):
        """
        Evaluate the outputs of CNMF-E. Eliminate non-neuron-like shapes.

        Parameters
        ---
        min_SNR: float, default: 3
            Minimum signal to noise ratio to be included in final component list.

        r_values_min: float, default: 0.85
            Minimum correlation to trace to be included in final component list.
            
        """
        self.cnm.params.set('quality', {'min_SNR': min_SNR,
                                        'rval_thr': r_values_min,
                                        'use_cnn': False})
        self.cnm.estimates.evaluate_components(self.images, self.cnm.params, dview=self.dview)
        
        # Print number of accepted components. 
        print(' ***** ')
        print('Number of total components: ', len(self.cnm.estimates.C))
        print('Number of accepted components: ', len(self.cnm.estimates.idx_components))

    #%%
    def plot_cells(self, display_numbers=False):
        """
        Plot the background plus detected cells.

        Parameters
        ---
        display_numbers: boolean, default: False
            Flag for plotting the numbers on top of the cell masks.

        """
        self.cnm.estimates.plot_contours(img=self.cn_filter, 
                                         idx=self.cnm.estimates.idx_components,
                                         display_numbers=display_numbers)
        
    #%% 
    def inspect_cells(self):
        """
        Inspect components on jupyter notebook.

        """
        self.cnm.estimates.nb_view_components(img=self.cn_filter, 
                                              idx=self.cnm.estimates.idx_components, 
                                              denoised_color='red', cmap='gray')
        
    #%%
    def save(self, save_name=None):
        """
        Save results.

        Parameters
        ---
        save_name: str, default: 'DESTINATION_PATH/caiman_outputs.pkl'
            FULL file name including the path where you want to save results.

        """
        save_name = os.path.join(self.destination, 
                                 'caiman_outputs.pkl') if save_name is None else save_name

        # Build dict.
        save_obj = {
                'files': self.fnames,
                'processed_file': self.fname_new,
                'crop_coords': self.crop_coords,
                'data': self.cnm.estimates,
                'cn_filter': self.cn_filter,
                'std_map': self.std_map,
                'opts': self.opts}

        # Save.
        with open(save_name, 'wb') as f:
            pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    #%% 
    def terminate(self):
        """
        Stops multiprocessing pools.

        """
        cm.stop_server(dview=self.dview)
        

#%%
class Registration:
    def __init__(self, paths):
        """
        Object for performing and inspecting across-day registration.

        Parameters
        ---
        paths: str
            Folders containing caiman_outputs.pkl files.

        """
        # Make a list of caiman_wrapper objects by loading pickled files.
        self.caiman_objs = []
        for folder in paths:
            with open(os.path.join(folder, 'caiman_outputs.pkl'), 'rb') as file:
                session = pickle.load(file)
            
            self.caiman_objs.append(session)
        
        # Get the cell outlines.
        A = [obj['data'].A for obj in self.caiman_objs]
        self.A = [csc_matrix(A1/A1.sum(0)) for A1 in A]

        # Get registration templates.
        self.templates = [obj['cn_filter'] for obj in self.caiman_objs]

        # Get dimensions.
        dims = [img.shape for img in self.templates]
        if all(x == dims[0] for x in dims):
            self.dims = dims[0]
        else:
            raise ValueError('Dimensions are not the same across sessions')

        # Get cell masks.
        self.masks = self.sparse_mat_to_mask(self.A)
        
        # Do registration. 
        self.A_union, self.assignments, self.matchings = self.register_sessions()

        # Get aligned cell masks
        self.aligned_masks = []
        self.aligned_masks.append(self.sparse_mat_to_mask([self.A[0]])[0]),
        self.aligned_masks.extend(self.sparse_mat_to_mask(self.A2))
        
    def register_sessions(self):
        """
        Do registration.

        """
        A_union, assignments, matchings = \
            register_multisession(self.A, self.dims, templates=self.templates)
            
        return A_union, assignments, matchings

    def plot_cells(self, sessions=None, only=False):
        """
        Plot cells across sessions.

        Parameters
        ---
        sessions: list, default: None
            Sessions to plot. If not specified, plots all sessions.

        only: boolean, default: False
            If True, plot cells that are active ONLY in these sessions and inactive in others. If False, components
            can be active in other sessions as well.

        """
        sessions = list(range(len(self.caiman_objs))) if sessions is None else sessions

        # Find activate components in the specified sessions.
        active_components = extract_active_components(self.assignments, sessions, only=only)
        sparsed_assignments = self.assignments[list(active_components)]
        assignments_list = sparsed_assignments.astype(int).T.tolist()

        # Plot contours of all cells.
        plt.figure()
        plt.imshow(self.templates[0], cmap='gray')
        for session in sessions:
            for mask in self.aligned_masks[session][assignments_list[session]]:
                plt.contour(mask)
                
    def find_cell_contours(self, masks, p_max_threshold=0.8):
        """
        Convert cell masks into contours. 
        """
        
        # Initialize full list. 
        contours = []
        
        # For each session, start a new list. 
        for session in masks:
            cells_this_session = []
            
            # Append each cell's contours onto this second-tier list. 
            for n, cell in enumerate(session):
                # Set a threshold for inclusion into contour region.
                threshold = p_max_threshold * cell.max()
                cell_contour = measure.find_contours(cell, threshold)
                    
                cells_this_session.append(cell_contour)
                
            # Append onto master list. 
            contours.append(cells_this_session)

        return contours

    def sparse_mat_to_mask(self, A):
        mask = [np.reshape(A_.toarray(), self.dims + (-1,),
                          order='F').transpose(2, 0, 1) for A_ in A]

        return mask
            

if __name__ == '__main__':
    s1 = 'L:\\CaImAn data folders\\BLA\\Mundilfari\\08_06_2018_Shock'
    s2 = 'L:\\CaImAn data folders\\BLA\\Mundilfari\\08_07_2018_Ext1a_Ctx1'
    s3 = 'L:\\CaImAn data folders\\BLA\\Mundilfari\\08_07_2018_Ext1b_Ctx2'
    from caiman_wrapper import Registration

    r = Registration([s1, s2, s3])

    pass