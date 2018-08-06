#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 22:07:12 2018

@author: felix

various different registration algorithms

# make sure to import the fio functions 

"""

from dipy.viz import regtools
from dipy.align.imaffine import transform_centers_of_mass, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D
    
# higher level import.
import Utility_Functions.file_io as fio
from Visualisation.imshowpair import imshowpair
import numpy as np 


#==============================================================================
#   First set = dipy translation registration algorithms. 
#==============================================================================

def setup_dipy_register(nbins=50, metric='mutualinfo', sampling_prop=None, level_iters=[10000, 1000, 100], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1]):
    
    static_grid2world = np.eye(4)
    moving_grid2world = np.eye(4)
    
    if metric == 'mutualinfo':
        metric = MutualInformationMetric(nbins, sampling_prop)
        
    affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
    
    return (static_grid2world, moving_grid2world), affreg

    
def dipy_register_translation(fixed, moving, reg_config=None, static_grid2world=None, moving_grid2world=None, affreg=None):
    
    if reg_config is not None:
        
        """
        setup affine registration object.  
        """
        (static_grid2world, moving_grid2world), affreg = setup_dipy_register(reg_config['nbins'], 
                                                                             reg_config['metric'], 
                                                                             reg_config['sampling_prop'].
                                                                             reg_config['level_iters'],
                                                                             reg_config['sigmas'],
                                                                             reg_config['factors'])
    
    transform = TranslationTransform3D()
    c_of_mass = transform_centers_of_mass(fixed, static_grid2world,
                                  moving, moving_grid2world)
    
    starting_affine = c_of_mass.affine
    params0 = None
    translation = affreg.optimize(fixed, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)
    
    registered = translation.transform(moving)
    
    return registered, translation
    

def dipy_register_translation_batch(dataset_files, in_folder, out_folder, reg_config, timer=True, debug=False):
    
    import pylab as plt 
    
    fixed = fio.read_multiimg_PIL(dataset_files[0]) # initialise the first as the reference. 
    
    # save this image into outfolder.
    fio.save_multipage_tiff(fixed, dataset_files[0].replace(in_folder, out_folder))
    
    """
    setup affine registration object.  
    """
    (static_grid2world, moving_grid2world), affreg = setup_dipy_register(reg_config['nbins'], 
                                                                         reg_config['metric'], 
                                                                         reg_config['sampling_prop'],
                                                                         reg_config['level_iters'],
                                                                         reg_config['sigmas'],
                                                                         reg_config['factors'])

    tforms = []
    
    if timer:
        from tqdm import tqdm
        for i in tqdm(range(len(dataset_files)-1)):
            
            # read in the next image
            moving = fio.read_multiimg_PIL(dataset_files[i+1])
            
            registered, translation = dipy_register_translation(fixed, moving, static_grid2world=static_grid2world, moving_grid2world=moving_grid2world, affreg=affreg)
        
            # replace this one. bug. 
            fio.save_multipage_tiff(registered, dataset_files[i+1].replace(in_folder, out_folder))
     
            if debug:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                imshowpair(ax[0], fixed[fixed.shape[0]//2], moving[moving.shape[0]//2])
                imshowpair(ax[1], fixed[fixed.shape[0]//2], registered[registered.shape[0]//2])
                plt.show()
            
            # update fixed. 
            fixed = registered.copy()
            tforms.append(translation)
            
    else:
        for i in range(len(dataset_files)-1):
            
            # read in the next image
            moving = fio.read_multiimg_PIL(dataset_files[i+1])
            
            registered, translation = dipy_register_translation(fixed, moving, static_grid2world=static_grid2world, moving_grid2world=moving_grid2world, affreg=affreg)
        
            # replace this one. bug. 
            fio.save_multipage_tiff(registered, dataset_files[i+1].replace(in_folder, out_folder))
     
            if debug:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                imshowpair(ax[0], fixed[fixed.shape[0]//2], moving[moving.shape[0]//2])
                imshowpair(ax[1], fixed[fixed.shape[0]//2], registered[registered.shape[0]//2])
                plt.show()
            
            # update fixed. 
            fixed = registered.copy()
            tforms.append(translation)
        
    return tforms


#==============================================================================
#   Matlab scripts for similarity transform (what of the dipy version? )
#==============================================================================

def matlab_register_similarity_batch(dataset_files, in_folder, out_folder, reg_config, timer=True, debug=False):
    
    """
    Registers the similarity transformation 
    """
    import matlab.engine
    import scipy.io as spio 
    import os
    import shutil
    import pylab as plt 
    
    eng = matlab.engine.start_matlab() 
    
    fixed = fio.read_multiimg_PIL(dataset_files[0]) # initialise the first as the reference. 
    
    tmp_folder = 'tmp'
    fio.mkdir(tmp_folder)
    
    spio.savemat(os.path.join(tmp_folder, 'im1.mat'), {'im1': fixed}) # save fixed temorary 
    
    # save this image into outfolder.
    fio.save_multipage_tiff(fixed, dataset_files[0].replace(in_folder, out_folder))
    
    tforms = []
    
    if timer:
        from tqdm import tqdm
        for i in tqdm(range(len(dataset_files)-1)):
        
            im2 = fio.read_multiimg_PIL(dataset_files[i+1])
            spio.savemat(os.path.join(tmp_folder, 'im2.mat'), {'im2': im2}) # save moving
            
            registered, transform = eng.register3D_rigid(os.path.join(tmp_folder, 'im1.mat'), os.path.join(tmp_folder, 'im2.mat'), 'tform.mat', 0, reg_config['downsample'], reg_config['modality'], reg_config['max_iter'], reg_config['type'], nargout=2)        
            
            registered = np.asarray(registered, dtype=np.uint8) # this data type conversion is critical. 
            transform = np.asarray(transform)
            
            # save this out 
            fio.save_multipage_tiff(registered, dataset_files[i+1].replace(in_folder, out_folder))
        
            if debug:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                imshowpair(ax[0], fixed[fixed.shape[0]//2], im2[im2.shape[0]//2])
                imshowpair(ax[1], fixed[fixed.shape[0]//2], registered[registered.shape[0]//2])
                plt.show()
            
            
            fixed = registered.copy()
            
            # replace the previous temp
            spio.savemat(os.path.join(tmp_folder,'im1.mat'), {'im1': fixed}) # save fixed
            tforms.append(transform)
            
    else:
        for i in range(len(dataset_files)-1):
        
            im2 = fio.read_multiimg_PIL(dataset_files[i+1])
            spio.savemat(os.path.join(tmp_folder, 'im2.mat'), {'im2': im2}) # save moving
            
            registered, transform = eng.register3D_rigid(os.path.join(tmp_folder, 'im1.mat'), os.path.join(tmp_folder, 'im2.mat'), 'tform.mat', 0, reg_config['downsample'], reg_config['modality'], reg_config['max_iter'], reg_config['type'], nargout=2)        
            
            registered = np.asarray(registered, dtype=np.uint8) # this data type conversion is critical. 
            transform = np.asarray(transform)
            
            # save this out 
            fio.save_multipage_tiff(registered, dataset_files[i+1].replace(in_folder, out_folder))
        
            if debug:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                imshowpair(ax[0], fixed[fixed.shape[0]//2], im2[im2.shape[0]//2])
                imshowpair(ax[1], fixed[fixed.shape[0]//2], registered[registered.shape[0]//2])
                plt.show()
            
            fixed = registered.copy()
            
            # replace the previous temp
            spio.savemat(os.path.join(tmp_folder,'im1.mat'), {'im1': fixed}) # save fixed
            tforms.append(transform)
            
            
    # remove temp folder
    shutil.rmtree(tmp_folder)
            
    return tforms