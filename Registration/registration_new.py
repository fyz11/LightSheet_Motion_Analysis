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
from Geometry import transforms as tf
from transforms3d.affines import decompose44, compose
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

# this contains bugs in windows? 
def dipy_register_translation(fixed, moving, reg_config=None, static_grid2world=None, moving_grid2world=None, affreg=None):
    
    if reg_config is not None:
        
        """
        setup affine registration object.  
        """
        (static_grid2world, moving_grid2world), affreg = setup_dipy_register(reg_config['nbins'], 
                                                                             reg_config['metric'], 
                                                                             reg_config['sampling_prop'],
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

def matlab_register(fixed_file, moving_file, save_file, reg_config):
    
    import matlab.engine
    import os 
    import scipy.io as spio 
    eng = matlab.engine.start_matlab() 
    
    if reg_config['view'] is not None:
        print('using initialisation')
        # use initialisation. 
        initial_tf = reg_config['initial_tf']
        spio.savemat('tform.mat', {'tform':initial_tf.T}) # transpose for matlab 
        
        transform = eng.register3D_rigid_faster(fixed_file, moving_file, save_file, 
                                                 'tform.mat', 1, reg_config['downsample'], 
                                                 reg_config['modality'], reg_config['max_iter'], 
                                                 reg_config['type'], 
                                                 reg_config['return_img'], 
                                                 nargout=1) 
    else:
        transform = eng.register3D_rigid_faster(fixed_file, moving_file, save_file, 
                                                 'tform.mat', 0, reg_config['downsample'], 
                                                 reg_config['modality'], reg_config['max_iter'], 
                                                 reg_config['type'], 
                                                 reg_config['return_img'], 
                                                 nargout=1)     
    transform = np.asarray(transform)
    
    if reg_config['return_img']!= 1:
        im1 = fio.read_multiimg_PIL(fixed_file)
        im2 = fio.read_multiimg_PIL(moving_file)
        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), np.array(im1.shape)[[2,1,0]]))
        im2_ = im2_.transpose(2,1,0) # flip back
    
        return transform, im2_
    else:
        return transform 
        

def matlab_register_batch(dataset_files, in_folder, out_folder, reg_config, debug=False):
    
    """
    Registers the similarity transformation 
    """
    import matlab.engine
    import scipy.io as spio 
    import os
    import shutil
    import pylab as plt 
    from tqdm import tqdm
    
    eng = matlab.engine.start_matlab() 
        
    tforms = []
    fixed_file = dataset_files[0]
    
    # save the reference image. 
    im1 = fio.read_multiimg_PIL(fixed_file)
    fio.save_multipage_tiff(im1, fixed_file.replace(in_folder, out_folder))

    for i in tqdm(range(len(dataset_files)-1)):
    
        moving_file = dataset_files[i+1]
        save_file = moving_file.replace(in_folder, out_folder)
        
        transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
                                         'tform.mat', 0, reg_config['downsample'], 
                                         reg_config['modality'], reg_config['max_iter'], 
                                         reg_config['type'], 
                                         reg_config['return_img'], 
                                         nargout=1)        
        
        transform = np.asarray(transform) # (z,y,x) 
        
#        if reg_config['return_img'] != 1: # this is too slow and disabled. 
        im1 = fio.read_multiimg_PIL(fixed_file)
        im2 = fio.read_multiimg_PIL(moving_file)
        
        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), np.array(im1.shape)[[2,1,0]]))
        im2_ = im2_.transpose(2,1,0) # flip back

        fio.save_multipage_tiff(im2_, save_file)

        if debug:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            imshowpair(ax[0], im1[im1.shape[0]//2], im2[im2.shape[0]//2]) 
            imshowpair(ax[1], im1[im1.shape[0]//2], im2_[im2_.shape[0]//2])
            
            fig, ax = plt.subplots(nrows=1, ncols=2)
            imshowpair(ax[0], im1[:,im1.shape[1]//2], im2[:,im2.shape[1]//2])
            imshowpair(ax[1], im1[:,im1.shape[1]//2], im2_[:,im2_.shape[1]//2])

            plt.show()
        
        fixed_file = dataset_files[i+1].replace(in_folder, out_folder)
        tforms.append(transform)
            
    return tforms
    
    
# for fast coarse translation adjustment using dipy, but appears problematic?
def align_centers(im1, im2):
    
    """
    im1: volume img
    im2: volume img
    """
    import numpy as np 
    from dipy.align.imaffine import transform_centers_of_mass
    
    static_grid2world = np.eye(4) 
    moving_grid2world = np.eye(4)
    
    c_of_mass = transform_centers_of_mass(im1, static_grid2world, im2, moving_grid2world)

    im2_ = c_of_mass.transform(im2)

    return im2_, c_of_mass.affine

    
def COM_2d(im1, im2):
    
    """
    grab center of mass of 2d images based on threholding. 
    """
    from scipy.ndimage.measurements import center_of_mass
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.filters import threshold_otsu
    
    mask1 = binary_fill_holes(im1>=threshold_otsu(im1))
    mask2 = binary_fill_holes(im2>=threshold_otsu(im2))

    return center_of_mass(mask1), center_of_mass(mask2)
    
    
# function to join two volume stacks 
def simple_join(stack1, stack2, cut_off=None, blend=True, offset=10, weights=[0.7,0.3]):
    
    """
    stack1 and stack2 are assumed to be the same size. 
    """
    
    if blend:
        combined_stack = np.zeros_like(stack1)
        combined_stack[:cut_off-offset] = stack1[:cut_off-offset]
        combined_stack[cut_off-offset:cut_off] = weights[0]*stack1[cut_off-offset:cut_off]+weights[1]*stack2[cut_off-offset:cut_off]
        combined_stack[cut_off:cut_off+offset] = weights[1]*stack1[cut_off:cut_off+offset]+weights[0]*stack2[cut_off:cut_off+offset]
        combined_stack[cut_off+offset:] = stack2[cut_off+offset:]         
    else:
        combined_stack = np.zeros_like(stack1)
        combined_stack[:cut_off] = stack1[:cut_off] 
        combined_stack[cut_off:] = stack2[cut_off:] 
    
    return combined_stack


#==============================================================================
#   Wrapper for 3D sift registration for registering aligning multi-view and sequential datasets. 
#==============================================================================
#==============================================================================
#   Wrapper for 3D sift registration for registering aligning multi-view and sequential datasets. 
#==============================================================================
def register3D_SIFT_wrapper(dataset_files, in_folder, out_folder, reg_config, reg_config_rigid=None):
    
    """
    Registers the similarity transformation exploiting the sift3D library.
    https://github.com/bbrister/SIFT3D
    
    returns transforms for each time point which subsequently applied to each image?  
    """
    import matlab.engine
    import scipy.io as spio 
    import os
    import shutil
    import pylab as plt 
    from tqdm import tqdm 
    import time 
    import pylab as plt 
    import Visualisation.imshowpair as imshowpair
    from skimage.exposure import rescale_intensity
    import sys
    
#     start the python matlab engine.
    eng = matlab.engine.start_matlab() 
    fio.mkdir(out_folder) # check that the output folder exists, create if does not exist.  
    
    if reg_config['mode'] == 1:
        
        print ('running sequential registration')
        
        tforms = []
        translate_matrixs = []
        
        datasetsave_files = np.hstack([dataset_files[2*i+1].replace(in_folder, out_folder) for i in range(len(dataset_files)//2)])
        
        for i in tqdm(range(len(dataset_files)//2)):
            
            im1file = dataset_files[2*i]
            im2file = dataset_files[2*i+1]
            
            tmatrix = eng.register3D_SIFT_wrapper(str(im1file), str(im2file), str(datasetsave_files[i]), 
                                              reg_config['downsample'], reg_config['lib_path'], reg_config['return_img'])
            tmatrix = np.asarray(tmatrix)
            tforms.append(tmatrix)
            
            """
            if matlab doesn't save matrix then we use python to do so.
            """
            if reg_config['return_img']!=1:
                
                im1 = fio.read_multiimg_PIL(im1file)
                im2 = fio.read_multiimg_PIL(im2file)
                                
                
                affine = np.zeros((4,4))
                affine[:-1,:] = tmatrix.copy()
                affine[-1] = np.array([0,0,0,1])
                
                """
                Allow different types of transformation. 
                """
                # decomposition (only if affine is not needed.)
                T,R,Z,S = decompose44(affine) # S is shear!
                
                if reg_config['type'] == 'similarity':
                    affine = compose(T, R, Z, np.zeros(3))
                if reg_config['type'] == 'rigid':
                    affine = compose(T, R, np.ones(3), np.zeros(3))
                if reg_config['type'] == 'translation':
                    affine = compose(T, np.eye(3), np.ones(3), np.zeros(3))

                im2_ = np.uint8(tf.apply_affine_tform(im2, np.linalg.inv(affine), np.array(im1.shape)))
                fio.save_multipage_tiff(im2_, datasetsave_files[i])
                
#                """
#                Translation correction.
#                """                
                translate_matrix, im2_ = matlab_register(str(im1file), str(datasetsave_files[i]), str(datasetsave_files[i]), reg_config_rigid)
                translate_matrixs.append(translate_matrix)
                
                fio.save_multipage_tiff(im2_, datasetsave_files[i])
                
                fig, ax = plt.subplots()
                imshowpair.imshowpair(ax, im1[im1.shape[0]//2], im2_[im2_.shape[0]//2])                
                fig, ax = plt.subplots()
                imshowpair.imshowpair(ax, im1[:,im1.shape[1]//2], im2_[:,im2_.shape[1]//2])
                plt.show()

            
    if reg_config['mode'] == 2:
        
        print ('running sequential registration')
            
        datasetsave_files = np.hstack([f.replace(in_folder, out_folder) for f in dataset_files])
        translate_matrixs = []

        if reg_config['return_img'] == 1:
            tforms = eng.register3D_SIFT_wrapper_batch(dataset_files, datasetsave_files, reg_config['downsample'], reg_config['lib_path'], reg_config['mode'])
        else:
            
            tforms = []
            
            # set the fixed image. 
            fixed = fio.read_multiimg_PIL(dataset_files[0])
            fio.save_multipage_tiff(fixed, datasetsave_files[0])            
            fixed_file = datasetsave_files[0]

            for i in tqdm(range(len(dataset_files)-1)):
                
                """
                we will first figure out all the sequential transforms. 
                """
                moving_file = dataset_files[i+1]
                
#                print('registering SIFT')
                tmatrix = eng.register3D_SIFT_wrapper(str(fixed_file), str(moving_file), str(datasetsave_files[i+1]), 
                                              reg_config['downsample'], reg_config['lib_path'], reg_config['return_img'])
                tmatrix = np.asarray(tmatrix)

                """
                Apply the transforms.
                """
                im1 = fio.read_multiimg_PIL(fixed_file)
                im2 = fio.read_multiimg_PIL(moving_file)
                
                affine = np.zeros((4,4))
                affine[:-1,:] = tmatrix.copy()
                affine[-1] = np.array([0,0,0,1])
                
                # decomposition (only if affine is not needed.)
                T,R,Z,S = decompose44(affine) # S is shear!
                
                if reg_config['type'] == 'similarity':
                    affine = compose(T, R, Z, np.zeros(3))
                if reg_config['type'] == 'rigid':
                    affine = compose(T, R, np.ones(3), np.zeros(3))
                if reg_config['type'] == 'translation':
                    affine = compose(T, np.eye(3), np.ones(3), np.zeros(3))
                
                tforms.append(affine)
                im2 = np.uint8(tf.apply_affine_tform(im2, np.linalg.inv(affine), np.array(im1.shape)))
                translate_matrixs.append(affine)

                fio.save_multipage_tiff(im2, datasetsave_files[i+1])
                
                
                fig, ax = plt.subplots()
                imshowpair.imshowpair(ax, im1[im1.shape[0]//2], im2[im2.shape[0]//2])                
                fig, ax = plt.subplots()
                imshowpair.imshowpair(ax, im1[:,im1.shape[1]//2], im2[:,im2.shape[1]//2])
                plt.show()
                
                # update the fixed file. 
                fixed_file = datasetsave_files[i+1]

    tforms = np.array(tforms)
    translate_matrixs = np.array(translate_matrixs)

    if reg_config['mode'] == 1:
        spio.savemat(os.path.join(out_folder, 'tforms_view_align.mat'), {'files': dataset_files,
                                                                        'config': reg_config,                                            'view_tforms': tforms,
                                                                        'translate_tforms':translate_matrixs})
    else:
        spio.savemat(os.path.join(out_folder, 'tforms_time_align.mat'), {'files': dataset_files,
                                                                        'config': reg_config,  
                                                                        'view_tforms': tforms,
                                                                        'translate_tforms':translate_matrixs})
    return (tforms, translate_matrixs)
    
    
    
    
    
    
    