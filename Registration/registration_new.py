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
#from Geometry import transforms as tf
#from Geometry import geometry as geom
import Geometry.transforms as tf
import Geometry.geometry as geom
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

def matlab_register(fixed_file, moving_file, save_file, reg_config, multiscale=False):
    
    import matlab.engine
    import scipy.io as spio 
    eng = matlab.engine.start_matlab() 
    
    if reg_config['view'] is not None:

        # use initialisation. 
        initial_tf = reg_config['initial_tf']

        # decompose the initialisation to comply with Matlab checks. # might be better ways to do this?
        T,R,Z,S = decompose44(initial_tf) # S is shear!
        
        if reg_config['type'] == 'similarity':
            # initialise with just rigid instead -> checks on this normally are too stringent.
            affine = compose(T, R, np.ones(3), np.zeros(3)) # the similarity criteria too difficult for matlab 
        if reg_config['type'] == 'rigid':
            affine = compose(T, R, np.ones(3), np.zeros(3))
        if reg_config['type'] == 'translation':
            affine = compose(T, np.eye(3), np.ones(3), np.zeros(3))
        if reg_config['type'] == 'affine':
            affine = initial_tf.copy()
        
        # save the tform as temporary for matlab to read. 
        spio.savemat('tform.mat', {'tform':affine}) # transpose for matlab 
        
        if multiscale == False:
            transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
                                                    'tform.mat', 1, reg_config['downsample'], 
                                                    reg_config['modality'], reg_config['max_iter'], 
                                                    reg_config['type'], 
                                                    reg_config['return_img'], 
                                                    nargout=1) 
        else:
            transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
                                                    'tform.mat', 1, reg_config['downsample'], 
                                                    reg_config['modality'], reg_config['max_iter'], 
                                                    reg_config['type'], 
                                                    reg_config['return_img'], 
                                                    nargout=1)
    else:
        if multiscale == False:
            transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
                                                    'tform.mat', 0, reg_config['downsample'], 
                                                    reg_config['modality'], reg_config['max_iter'], 
                                                    reg_config['type'], 
                                                    reg_config['return_img'], 
                                                    nargout=1)    
        else:
            print('multiscale')
            # convert to matlab arrays. 
            levels = matlab.double(reg_config['downsample'])
            warps = matlab.double(reg_config['max_iter'])
            transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
                                                    'tform.mat', 0, levels, 
                                                    reg_config['modality'], warps, 
                                                    reg_config['type'], 
                                                    reg_config['return_img'], 
                                                    nargout=1)
    transform = np.asarray(transform)
    
    if reg_config['return_img']!= 1:
        im1 = fio.read_multiimg_PIL(fixed_file)
        im2 = fio.read_multiimg_PIL(moving_file)
        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), np.array(im1.shape)[[2,1,0]]))
        im2_ = im2_.transpose(2,1,0) # flip back
    
        return transform, im2_, im1
    else:
        return transform 
        
# somewhat redundant. 
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
    fio.mkdir(out_folder)

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
        
        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), sampling_grid_shape=np.array(im1.shape)[[2,1,0]]))
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
        
    # save out tforms into a .mat file.
    tformfile = os.path.join(out_folder, 'tforms-matlab.mat')
    tforms = np.array(tforms)
    
    spio.savemat(tformfile, {'tforms':tforms,
                             'in_files':dataset_files,
                             'out_files':np.hstack([f.replace(in_folder, out_folder) for f in dataset_files])})
            
    return tforms

# add multiscale capability 
def matlab_group_register_batch(dataset_files, ref_file, in_folder, out_folder, reg_config, reset_ref_steps=0, multiscale=True, debug=False):
    
    """
    Registers the affine transformation temporally. ref_file is a 'derived' ref e.g. mean volume img. 

    Params:
    -------
        reset_ref_steps (int): if > 0, uses every nth registered file as the reference. default of 0 = fixed reference, for 1 = sequential    
    """
    import matlab.engine
    import scipy.io as spio 
    import os
    import shutil
    import pylab as plt 
    from tqdm import tqdm
    
    # start matlab engine. 
    eng = matlab.engine.start_matlab() 
    fio.mkdir(out_folder) # check output folder exists. 
    
    tforms = []
    ref_files = []
    
    # difference is now fixed_file is always the same one. 
    fixed_file = ref_file
    all_save_files = np.hstack([f.replace(in_folder, out_folder) for f in dataset_files])
    
    for i in tqdm(range(len(dataset_files))):
    
        moving_file = dataset_files[i]
        save_file = all_save_files[i]
        
        if multiscale:
            print('multiscale')
            levels = matlab.double(reg_config['downsample'])
            warps = matlab.double(reg_config['max_iter'])
            transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
                                                    'tform.mat', 0, levels, 
                                                    reg_config['modality'], warps, 
                                                    reg_config['type'], 
                                                    reg_config['return_img'], 
                                                    nargout=1)
        else:
            transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
                                            'tform.mat', 0, reg_config['downsample'], 
                                            reg_config['modality'], reg_config['max_iter'], 
                                            reg_config['type'], 
                                            reg_config['return_img'], 
                                            nargout=1)        
        
        transform = np.asarray(transform) # (z,y,x) 
        tforms.append(transform)
        ref_files.append(fixed_file) # which is the reference used. 

#        if reg_config['return_img'] != 1: # this is too slow and disabled. 
        im1 = fio.read_multiimg_PIL(fixed_file)
        im2 = fio.read_multiimg_PIL(moving_file)
        
        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), sampling_grid_shape=np.array(im1.shape)[[2,1,0]]))
        im2_ = im2_.transpose(2,1,0) # flip back

        fio.save_multipage_tiff(im2_, save_file)

        if debug:

            # visualize all three possible cross sections for reference and checking.
            fig, ax = plt.subplots(nrows=1, ncols=2)
            imshowpair(ax[0], im1[im1.shape[0]//2], im2[im2.shape[0]//2]) 
            imshowpair(ax[1], im1[im1.shape[0]//2], im2_[im2_.shape[0]//2])
            
            fig, ax = plt.subplots(nrows=1, ncols=2)
            imshowpair(ax[0], im1[:,im1.shape[1]//2], im2[:,im2.shape[1]//2])
            imshowpair(ax[1], im1[:,im1.shape[1]//2], im2_[:,im2_.shape[1]//2])

            fig, ax = plt.subplots(nrows=1, ncols=2)
            imshowpair(ax[0], im1[:,:,im1.shape[2]//2], im2[:,:,im2.shape[2]//2])
            imshowpair(ax[1], im1[:,:,im1.shape[2]//2], im2_[:,:,im2_.shape[2]//2])

            plt.show()
        
        if reset_ref_steps > 0:
            # change the ref file.
            if np.mod(i+1, reset_ref_steps) == 0: 
                fixed_file = save_file # change to the file you just saved. 
        
    # save out tforms into a .mat file.
    tformfile = os.path.join(out_folder, 'tforms-matlab.mat')
    tforms = np.array(tforms)
    ref_files = np.hstack(ref_files)
    
    spio.savemat(tformfile, {'tforms':tforms,
                             'ref_files':ref_files,
                             'in_files':dataset_files,
                             'out_files':all_save_files,
                             'reg_config':reg_config})
            
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


def sigmoid_join(stack1, stack2, cut_off=None, blend=True, gradient=200, shape=1, debug=False):
    
    """
    stack1 and stack2 are assumed to be the same size. 
    """
    def generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient):
        
        x = np.arange(0,stack1.shape[0])
        weights2 = 1./((1+np.exp(-grad*(x - cut_off)))**(1./shape))
        weights1 = 1. - weights2
        
        return weights1, weights2
    
    weights1, weights2 = generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient)
    if debug:
        import pylab as plt 
        plt.figure()
        plt.plot(weights1, label='1')
        plt.plot(weights2, label='2')
        plt.legend()
        plt.show()
        
    return stack2*weights1[:,None,None] + stack1*weights2[:,None,None]


def simple_recolor_join(stack1, stack2, cut_off=None, mode='global'):
    
    """
    if we are recoloring then not much need for blending?
    stack1 and stack2 are assumed to be the same dimensions. 
    
    Two options of global or local recoloring -> local uses each slice separately, global uses the mean average slice image. 
    
    How to speed things up? - next step. 
    """
    import Image_Functions.recolor as recolor
    
    if mode == 'global':
        mean1 = stack1.mean(axis=0)
        mean2 = stack2.mean(axis=0)
    
        # recolor works with RGB images. 
        ref_im1 = np.dstack([mean1, mean1, mean1])
        ref_im2 = np.dstack([mean2, mean2, mean2])
        
        # learn the mix_matrix. 
        _, mix_matrix = recolor.match_color(ref_im1/255., ref_im2/255., mode='chol', eps=1e-8, source_mask=None, target_mask=None, ret_matrix=True)
    
    combined_stack = np.zeros_like(stack1)
    combined_stack[:cut_off] = stack1[:cut_off] 
    
    # apply slice by slice -> how to parallelise? 
    for z in range(cut_off, stack1.shape[0],1):
        z_im1 = stack1[z]; z_im1 = np.dstack([z_im1, z_im1, z_im1])
        z_im2 = stack2[z]; z_im2 = np.dstack([z_im2, z_im2, z_im2])

        # should i be masking? -> why is there a discontinuity that  is neither? 
        if mode == 'local':
            im2_ = recolor.match_color(z_im1/255., z_im2/255., mode='chol', eps=1e-8, source_mask=None, target_mask=None)[:,:,0]
        if mode == 'global':
            im2_ = recolor.recolor_w_matrix(z_im1/255., z_im2/255., mix_matrix, source_mask=None, target_mask=None)[:,:,0]
        combined_stack[z] = np.uint8(255*im2_)
    
    """
    compute the intensity offset
    """
    offset_I = combined_stack[cut_off].mean() - combined_stack[cut_off-1].mean()    
    print(offset_I)
    combined_stack[cut_off:] = np.uint8(np.clip(combined_stack[cut_off:] + offset_I, 0, 255))
    
    return combined_stack
    

#==============================================================================
#   Wrapper for using MIND () in 3D to yield biologically plausible registrations of the surface, generically for any other non-rigid is also possible.
#==============================================================================
def nonregister_3D(infile1, infile2, savefile, savetransformfile, reg_config):
        
    """
    this registers once with SIFT and returns transformations. 
    """
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    print(reg_config['alpha'], reg_config['levels'], reg_config['warps'])
    
#    alpha = matlab.single(reg_config['alpha'])
    levels = matlab.double(reg_config['levels'])
    warps = matlab.double(reg_config['warps'])
    
    # add in additional options for modifying. 
    return_val = eng.nonrigid_register3D_MIND(str(infile1), str(infile2), str(savefile), str(savetransformfile),
                                      reg_config['alpha'], levels, warps)
        
    return return_val    

def nonregister_3D_demons(infile1, infile2, savefile, savetransformfile, reg_config):
        
    """
    this registers once with SIFT and returns transformations. 
    """
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    print(reg_config['level'], reg_config['warps'])
    
#    alpha = matlab.single(reg_config['alpha'])
#    level = matlab.double(reg_config['level'])
    level = float(reg_config['level'])
    warps = matlab.double(reg_config['warps'])
    smoothing = float(reg_config['alpha'])
    
    # add in additional options for modifying. 
    return_val = eng.nonrigid_register3D_demons(str(infile1), str(infile2), str(savefile), str(savetransformfile),
                                       level, warps, smoothing)
        
    return return_val  

# warping the image using matlab 
def warp_3D_demons_tfm(infile, savefile, transformfile, downsample, direction=1):
    """
    this warps the input image file according to the deformation field described in transformfile. If direction == 1 warp in the same direction else if direction == -1 in the reverse direction.
    """
    import matlab.engine
    eng = matlab.engine.start_matlab()

    return_val = eng.warp_3D_demons(str(infile), str(savefile), str(transformfile), int(downsample), direction)

    return return_val


# warping the image using python -> bypasses matlab easier for interfacing with python directly. 
def warp_3D_demons_matlab_tform_scipy(im, tform, direction='F'):
    
    import scipy.io as spio
    from skimage.transform import resize
    from scipy.ndimage import map_coordinates
    import Utility_Functions.file_io as fio
    
    dx,dy,dz = fio.read_demons_matlab_tform(tform, im.shape)
    im_interp = warp_3D_displacements_xyz(im, dx, dy, dz, direction=direction)
    
    return im_interp


def warp_3D_displacements_xyz(im, dx, dy, dz, direction='F'):
    
    from skimage.transform import resize
    from scipy.ndimage import map_coordinates
    
    XX, YY, ZZ = np.indices(im.shape) # set up the interpolation grid.
    
    if direction == 'F':
        XX = XX + dx
        YY = XX + dy 
        ZZ = ZZ + dz
    if direction == 'B':
        XX = XX - dx
        YY = YY - dy
        ZZ = ZZ - dz
    
    # needs more memory % does this actually work? 
    im_interp = map_coordinates(im, 
                                np.vstack([(XX).ravel().astype(np.float32), 
                                           (YY).ravel().astype(np.float32), 
                                           (ZZ).ravel().astype(np.float32)]), prefilter=False, order=1, mode='nearest')
    im_interp = im_interp.reshape(im.shape)
    
    return im_interp


def warp_3D_transforms_xyz(im, tmatrix, direction='F'):
    
    """
    This function is mainly to test how to combine the coordinates with transforms. 
    """
    from scipy.ndimage import map_coordinates
    XX, YY, ZZ = np.indices(im.shape) # set up the interpolation grid.
    
    xyz = np.vstack([(XX).ravel().astype(np.float32), 
                     (YY).ravel().astype(np.float32), 
                     (ZZ).ravel().astype(np.float32),
                     np.ones(len(ZZ.ravel()), dtype=np.float32)])
    
    if direction == 'F':
        xyz_ = tmatrix.dot(xyz)
    if direction == 'B':
        print('warp_inverse')
        print(np.linalg.inv(tmatrix))
        xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
    
    # needs more memory % does this actually work? 
    im_interp = map_coordinates(im, 
                                xyz_[:3], prefilter=False, order=1, mode='nearest')
    im_interp = im_interp.reshape(im.shape)
    
    return im_interp

#==============================================================================
#   Wrapper for 3D sift registration for registering aligning multi-view and sequential datasets. 
#==============================================================================
def register3D_SIFT(infile1, infile2, reg_config):
        
    """
    this registers once with SIFT and returns transformations. 
    """
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    # add in additional options for modifying. 
    tmatrix = eng.register3D_SIFT_wrapper(str(infile1), str(infile2), str('blank'), 
                                      reg_config['downsample'], reg_config['lib_path'], reg_config['return_img'], reg_config['nnthresh'])
    tmatrix = np.asarray(tmatrix)
    
    # return the matrix. 
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
        
    return affine


def register3D_SIFT_wrapper(dataset_files, in_folder, out_folder, reg_config, reg_config_rigid=None, swap_axes=None, debug=False):
    
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
    
    # start the python matlab engine.
    eng = matlab.engine.start_matlab() 
    fio.mkdir(out_folder) # check that the output folder exists, create if does not exist.  
    
    # check the reg_config? 
    if 'nnthresh' not in reg_config.keys():
        reg_config['nnthresh'] = 0.95
    if 'sigmaN' not in reg_config.keys():
        reg_config['sigmaN'] = 1.1 # good for high quality registration but may wish to increase to speed up 
    if 'numKpLevels' not in reg_config.keys():
        reg_config['numKpLevels'] = 3 # standard number of octaves for sift registration 
    if 'return_img' not in reg_config.keys():
        reg_config['return_img'] = 0 # don't return in matlab, let python handle all the grunt work.

    if reg_config['mode'] == 1:
        
        print ('running view alignment')

        tforms = []
        translate_matrixs = []
        
        # compile the elongated file lengths. 
        datasetsave_files = np.hstack([dataset_files[2*i+1].replace(in_folder, out_folder) for i in range(len(dataset_files)//2)])
        
        for i in tqdm(range(len(dataset_files)//2)):
            
            im1file = dataset_files[2*i]
            im2file = dataset_files[2*i+1]
            
            tmatrix = eng.register3D_SIFT_wrapper(str(im1file), str(im2file), str(datasetsave_files[i]), 
                                              reg_config['downsample'], reg_config['lib_path'], reg_config['return_img'], reg_config['nnthresh'], reg_config['sigmaN'], reg_config['numKpLevels'])

            tmatrix = np.array(tmatrix)
            """
            if matlab doesn't save matrix then we use python to do so.
            """
            if reg_config['return_img']!=1:

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
                if reg_config['type'] == 'affine':
                    pass

                tforms.append(affine) # use this matrix to save. 
                
                if reg_config_rigid is None:
                    # no downstream refinement. 
                    im1 = fio.read_multiimg_PIL(im1file)
                    im2 = fio.read_multiimg_PIL(im2file)
                                
                    if swap_axes is None:
                        im2_ = np.uint8(tf.apply_affine_tform(im2, np.linalg.inv(affine), sampling_grid_shape=np.array(im1.shape)))
                    else:
                        # transpose the components in the affine matrix 
                        new_affine = affine[:3].copy()
                        new_affine = new_affine[swap_axes,:] # flip rows first. (to flip the translation.)
                        new_affine[:,:3] = new_affine[:,:3][:,swap_axes] #flip columns (ignoring translations)
                        new_affine = np.vstack([new_affine, [0,0,0,1]]) # make it homogeneous 4x4 transformation. 
                        
                        im2_ = np.uint8(tf.apply_affine_tform(im2.transpose(swap_axes), np.linalg.inv(new_affine), sampling_grid_shape=np.array(im1.shape)[[2,1,0]]))
                        im2_ = im2_.transpose(np.argsort(swap_axes)) # reverse the swap_axes. 
                    
                    print('saving')
                    print(datasetsave_files[i])
                    fio.save_multipage_tiff(im2_, datasetsave_files[i])
                else:
                    """
                    Downstream correction is used therefore just pass the transform along. 
                    """                
                    affine_matlab = geom.shuffle_Tmatrix_axis_3D(affine, [2,1,0]) # don't need to shuffle at all ? 
                    print(affine_matlab)
                    # add these two options to the dict or modify to make sure this works. 
                    reg_config_rigid['view'] = None
                    reg_config_rigid['initial_tf'] = affine_matlab 
                    
                    # this allows reuse -> just pass the transform without having to interpolate first. 
                    translate_matrix, im2_, im1 = matlab_register(str(im1file), str(im2file), str(datasetsave_files[i]), reg_config_rigid)
                    translate_matrixs.append(translate_matrix)
                    
                # save the final registered ver. of image 2. 
                fio.save_multipage_tiff(im2_, datasetsave_files[i])
                
                if debug == True:
                    fig, ax = plt.subplots()
                    imshowpair.imshowpair(ax, im1[im1.shape[0]//2], im2_[im2_.shape[0]//2])                
                    fig, ax = plt.subplots()
                    imshowpair.imshowpair(ax, im1[:,im1.shape[1]//2], im2_[:,im2_.shape[1]//2])
                    fig, ax = plt.subplots()
                    imshowpair.imshowpair(ax, im1[:,:,im1.shape[2]//2], im2_[:,:,im2_.shape[2]//2])
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
                                              reg_config['downsample'], reg_config['lib_path'], reg_config['return_img'], reg_config['nnthresh'])
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
                im2 = np.uint8(tf.apply_affine_tform(im2, np.linalg.inv(affine), sampling_grid_shape=np.array(im1.shape)))
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

        if reg_config_rigid is None:
            translate_matrixs = 0

        spio.savemat(os.path.join(out_folder, 'tforms_view_align.mat'), {'files': dataset_files,
                                                                        'config': reg_config,                                            
                                                                        'view_tforms': tforms,
                                                                        'translate_tforms':translate_matrixs})
    else:

        if reg_config_rigid is None:
            translate_matrixs = 0
        spio.savemat(os.path.join(out_folder, 'tforms_time_align.mat'), {'files': dataset_files,
                                                                        'config': reg_config,  
                                                                        'view_tforms': tforms,
                                                                        'translate_tforms':translate_matrixs})
    return (tforms, translate_matrixs)
    
    
    
    
    
    
    