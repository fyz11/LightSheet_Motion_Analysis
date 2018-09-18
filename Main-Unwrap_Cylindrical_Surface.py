# -*- coding: utf-8 -*-
"""
Holly Hathrell
Main.py
Last edited 05/02/2018
"""
import os
import sys


import Utility_Functions.file_io as fio
import Unzipping.unzip_new as uzip # unzip new is the latest!. 
import numpy as np
import pylab as plt 
from skimage.exposure import rescale_intensity, equalize_adapthist
import Geometry.transforms as tf

from scipy.misc import imsave
from tqdm import tqdm 
import Geometry.meshtools as meshtools
import skimage.measure as measure 
"""
Read in Dataset. 
"""
# load in a set of files. 
dataset_folder = '/media/felix/Elements/Shankar LightSheet/Example Timelapse/test'
out_aligned_folder = os.path.join(dataset_folder, 'aligned2')
dataset_files = fio.load_dataset(out_aligned_folder, ext='.tif', split_key='TP_',split_position=1) # load in the just aligned files.


"""
Set the write out folder
"""
#out_folder = 'test_polar_unwrap_same';
#fio.mkdir(out_folder)

# test the parametrization approach again? 
for i in tqdm(range(len(dataset_files))[1:3]):
    
    im_array = fio.read_multiimg_PIL(dataset_files[i])
    im_array = im_array.transpose(1,0,2)
    """
    Tilt correction if need be.... 
    """
#    # uncomment if needs to do some tilt correction!. 
#    im_tilt_mask = np.zeros_like(im_array); im_tilt_mask[:,:150,:] = 1
#
#    # correct the axial tilt first. (z should be the last coordinate. )
#    tilt_tf, im_array = tf.correct_axial_tilt(im_array.transpose(0,2,1), I_thresh=15, ksize=3, mode='pca', pole='S', mask=im_tilt_mask.transpose(0,2,1), use_im_center=True, out_shape=None)
#    im_array = im_array.transpose(2,0,1) # transpose axes back. 
    
    """
    Segment the surface. 
    """
    # this surface is for sure non-manifold!. 
    embryo_mask, contour_mask = uzip.segment_contour_embryo(im_array, I_thresh=10, ksize=3, fast_flag=False) # this effectively finds the surface.
    
    """
    May need to erode the binary mask to fetch the surface intensities.  
    """
    # cut to the surface!. 
    embryo_mask = uzip.shrink_emb_mask(embryo_mask, pad=10)
    contour_mask = uzip.contour_seg_embryo(im_array, embryo_mask) # recompute the contour.
    
    
    # get the only 1 component in the contour mask. 
    """
    OPTIONAL: filter and retain the largest component.
    """
    contour_mask = uzip.keep_largest_component(contour_mask)
    


# =============================================================================
#   Main workload!. 
# =============================================================================

    # extract the point coordinates on the surface. 
    coords = np.array(np.where(contour_mask > 0)).T
    center = [np.mean(coords[:,0]), np.mean(coords[:,1]), np.mean(coords[:,2])]
    
    # resample the points !. 
#    coords = meshtools.create_clean_mesh_points(im_array, contour_coords, n_pts=100, kind='linear', min_pts=5, eps=1, alpha=[1000,1000])
    
# =============================================================================
#   Unwrapping now .    
# =============================================================================
    
    print('starting unwrapping')
    
    """
    Compute the cylindrical statistics (need to run this for each timepoint)
    """
    # compute statistics. 
    unwrap_params = uzip.compute_cylindrical_statistics(contour_mask, coords, smoothing=None)
    
# =============================================================================
#   Generate the mapping space (i,j) -> (x,y,z). This only needs to be done once for one TP. 
# =============================================================================
    
    if i == 1:
        print('learning mapping parameters')
        # generate ref space
        ref_space = uzip.build_mapping_space(unwrap_params['coords'], 
                                             ranges=unwrap_params['ranges'], 
                                             shape=unwrap_params['aspect_ratio'])
    
    
        mapped_coords = uzip.match_coords_to_ref_space(unwrap_params['coords'], 
                                                       ref_space[0,:,0], 
                                                       ref_space[:,0,1], map_index=[-2,0]) # mapping 3D->2D
        
        ref_coord_set = np.hstack([unwrap_params['coords'], mapped_coords])
        ref_map = uzip.gen_ref_map(im_array, ref_coord_set, ref_space, interp_method='cubic') # should be in the form of an image. 
        
        ref_map = ref_map.reshape(-1,3).astype(np.int)
        
# =============================================================================
#   Application of the map to new data.          
# =============================================================================
    # now we freely retrieve the corresponding points for all subsequent time points! without generating a map again!. 
    mapped_I = im_array[ref_map[:,0], ref_map[:,1], ref_map[:,2]].reshape(ref_space.shape[:-1])

    mapped_I_rows, mapped_I_cols = np.indices(mapped_I.shape)
#    mapped_I_dist = np.abs(ref_map[:,0] - unwrap_params['center'][0])
    
    unwrapped_img = uzip.map_intensities(np.vstack([mapped_I_rows.ravel(), mapped_I_cols.ravel()]).T, 
                                         mapped_I.ravel(), ref_space.shape[:-1], 
                                         interp=True,  min_I=5)
###       
    plt.figure(figsize=(10,10));plt.imshow(equalize_adapthist(unwrapped_img[0]/255., clip_limit=0.01))


#    from scipy.misc import imsave
#     
#    imsave('Results/unwrapped_corrected_L491.tif', np.uint8(255* equalize_adapthist(unwrapped_img[0]/255., clip_limit=0.01)))

