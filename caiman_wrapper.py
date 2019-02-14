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
import cv2
import numpy as np
import os
import tifffile as TIF
import matplotlib.pyplot as plt
import pickle

class CaimanWrapper:
    """
    CaimanWrapper is a class designed to operate around the CaImAn package, 
    allowing fast and efficient utilization of the available tools. 
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
        
        # Crop and spatially downsample files.
        if not self.skip_file_transfer:
            self.get_std_map()
            plt.pause(5)        # Necessary because IDEs are dumb.
            self.fnames = self.crop_and_downsample()
        else:
            self.fnames = [os.path.join(self.destination, f) 
                           for f in os.listdir(self.destination) 
                           if f.endswith('.tif')]
            self.fnames.insert(0, self.fnames.pop())
    
    #%%
    def run_phase1(self, skip_inspection=False):
        """
        Sets parameters, runs motion correction, and displays summary images
        """
        # Set parameters. 
        self.opts = self.init_mc_params()
        
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
                              if (f.endswith('.mmap') and f.startswith('memmap__'))][0]
            self.bord_px = 0
            
        # Load the memory mapped file and plot the summary images. 
        self.load_memmap_file()
        
        self.init_cnm_params()
        if not skip_inspection:
            self.inspect_summary()
        
        
    def run_phase2(self):
        """
        Enter new min_corr and min_pnr values, then run CNMF-E and eliminate 
        bad neurons.
        """
        min_corr = float(input('New min_corr value: '))
        min_pnr = float(input('New min_pnr value: '))
        
        # Change parameters based on plots. 
        self.change_params(min_corr=min_corr, min_pnr=min_pnr)
        
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
        
    #%%
    def crop_and_downsample(self):
        """
        Crop and spatially downsample the raw files before proceeding.
        """
        # Get raw file names. 
        files_to_process = [os.path.join(self.raw_dir, f) 
                            for f in os.listdir(self.raw_dir) 
                            if (f.endswith('.tif') and f.startswith('recording'))]
        
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
            
        return fnames
    
    #%%
    def init_mc_params(self,
                       motion_correct=True,
                       pw_rigid=False,
                       gSig_filter = (3,3),
                       max_shifts = (5,5),
                       strides = (48,48),
                       overlaps = (24,24),
                       max_deviation_rigid = 3,
                       border_nan = 'copy',
                       frame_rate = 20,
                       decay_time = .4,
                       ):
        """
        Define parameters
        """
        #Build the dict. 
        params_dict = {
                'fnames': self.fnames,      # file names
                'fr': frame_rate,           
                'decay_time': decay_time,   # transient decay time
                'pw_rigid': pw_rigid,       # flag for performing piecewise-rigid motion correction (otherwise just rigid)
                'max_shifts': max_shifts,   # maximum allowed rigid shift
                'gSig_filt': gSig_filter,   # size of high pass spatial filtering, used in 1p data
                'strides': strides,         # start a new patch for pw-rigid motion correction every x pixels
                'overlaps': overlaps,       # overlap between pathes (size of patch strides+overlaps)
                'max_deviation_rigid': max_deviation_rigid, # maximum deviation allowed for patch with respect to rigid shifts
                'border_nan': border_nan,   # replicate values along the boundaries
                }
        
        opts = params.CNMFParams(params_dict = params_dict)
        
        return opts
    
    #%%
    def init_cnm_params(self,
                        p = 1,               # order of the autoregressive system
                        K = None,           # upper bound on number of components per patch, in general None
                        gSig = (3, 3),       # gaussian width of a 2D gaussian kernel, which approximates a neuron
                        gSiz = None,         # average diameter of a neuron, in general 4*gSig+1
                        Ain = None,          # possibility to seed with predetermined binary masks
                        merge_thresh = .7,   # merging threshold, max correlation allowed
                        rf = 40,             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
                        stride_cnmf = None,  # amount of overlap between the patches in pixels
                        #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
                        tsub = 2,            # downsampling factor in time for initialization
                        ssub = 1,            # downsampling factor in space for initialization
                        low_rank_background = None,  # None leaves background of each patch intact,
                        #                     True performs global low-rank approximation if gnb>0
                        gnb = 0,             # number of background components (rank) if positive
                        nb_patch = 0,        # number of background components (rank) per patch if gnb>0,
                        #                     else it is set automatically
                        min_corr = .8,       # min peak value from correlation image
                        min_pnr = 15,        # min peak to noise ration from PNR image
                        ssub_B = 2,          # additional downsampling factor in space for background
                        ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
                        ):
        
        gSiz = (4*gSig[0]+1, 4*gSig[0]+1) if gSiz is None else gSiz
        stride_cnmf = 4*gSig[0] if stride_cnmf is None else stride_cnmf
        
        self.opts.change_params(params_dict={
                'method_init': 'corr_pnr',
                'K': K,
                'gSig': gSig,
                'gSize': gSiz,
                'merge_thresh': merge_thresh,
                'p': p,
                'tsub': tsub,
                'ssub': ssub,
                'rf': rf,
                'stride': stride_cnmf,
                'only_init': True,
                'nb': gnb, 
                'nb_patch': nb_patch,
                'method_deconvolution': 'oasis',
                'low_rank_background': low_rank_background,
                'update_background_components': True,
                'min_corr': min_corr,
                'min_pnr': min_pnr,
                'normalize_init': False,
                'center_psf': True,
                'ssub_B': ssub_B,
                'ring_size_factor': ring_size_factor,
                'del_duplicates': True,
                'border_pix': self.bord_px})
    
    #%% 
    def change_params(self, **kwargs):
        """
        Change specified parameters.
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
        self.fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
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
        self.cn_filter, pnr = cm.summary_images.correlation_pnr(self.images, 
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
    def plot_cells(self):
        """
        Plot the background plus detected cells. 
        """
        self.cnm.estimates.plot_contours(img=self.cn_filter, 
                                         idx=self.cnm.estimates.idx_components)
        
    #%% 
    def inspect_cells(self):
        self.cnm.estimates.nb_view_components(img=self.cn_filter, 
                                              idx=self.cnm.estimates.idx_components, 
                                              denoised_color='red', cmap='gray')
        
    #%%
    def save(self, save_name=None):
        """
        Save stuff.
        """
        save_name = os.path.join(self.destination, 
                                 'caiman_outputs.pkl') if save_name is None else save_name
                                 
        save_obj = {
                'files': self.fnames,
                'processed_file': self.fname_new,
                'data': self.cnm.estimates,
                'cn_filter': self.cn_filter,
                'std_map': self.std_map,
                'opts': self.opts}

        with open(save_name, 'wb') as f:
            pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    #%% 
    def terminate(self):
        cm.stop_server(dview=self.dview)