#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:22:56 2018

@author: felix
"""

if __name__=="__main__":
    
    """
    Script aims to implement a full pipeline based on the library. 
    """
    
    import Utility_Functions.file_io as fio
    import Utility_Functions.stack as stack_utils
    from Visualisation.imshowpair import imshowpair
    from skimage.exposure import rescale_intensity
#    import Registration.registration as registration
    import Registration.registration_new as registration
    import Optical_Flow.optflow as optflow
    import Tracking.supervoxel_tracks as svoxel_tracks
    import Visualisation.volume_img as vimgviz
    import os 
    import numpy as np 
    
    import time
    import pylab as plt 
    from tqdm import tqdm
    import scipy.io as spio
    import Utility_Functions.stack as stack
    
#==============================================================================
#     load in the dataset 
#==============================================================================

    dataset_folder = '/media/felix/Elements/Shankar LightSheet/Example Timelapse/test'
    
    """
    Create output folders.
    """
    out_pre_folder = os.path.join(dataset_folder, 'preprocess_full'); fio.mkdir(out_pre_folder)
#    out_pre_folder = os.path.join(dataset_folder, 'enhance_scaled'); fio.mkdir(out_pre_folder)
#    out_aligned_folder = os.path.join(dataset_folder, 'aligned_full'); fio.mkdir(out_aligned_folder)
    out_aligned_folder = os.path.join(dataset_folder, 'aligned_sift-enh-test-correct1'); fio.mkdir(out_aligned_folder)
#    out_aligned_folder2 = os.path.join(dataset_folder, 'aligned2_full'); fio.mkdir(out_aligned_folder2)
#    out_flow_folder = os.path.join(dataset_folder, 'optflow_full'); fio.mkdir(out_flow_folder)
#    out_track_folder = os.path.join(dataset_folder, 'tracks_full'); fio.mkdir(out_track_folder)
    
##==============================================================================
##   Preprocessing: Rescale and repad the frames in the dataset (for speed) 
##==============================================================================
#
#    dataset_files = fio.load_dataset(dataset_folder, ext='.tif', split_position=1)
##    downsample_factor = 5
#    pad_num = 11
#    
#    
#    for f in tqdm(dataset_files[:]):
#        vidstack = fio.read_multiimg_PIL(f)
#        sepstacks = stack_utils.split_stack(vidstack, n_splits=1)
#        
#        # convert to uint8
#        newsepstack = np.uint8(255*rescale_intensity(sepstacks[0]/255.))
#        newsepstack = stack.pad_z_stack_adj(newsepstack, pad_slices=pad_num, min_I=15)
#        
#        # save this out. 
#        basename = fio.get_basename(f, ext='.tif') # extract the basename (minus extension)
#        savefilename = os.path.join(out_pre_folder, basename+'.tif') # save into the prefolder. 
#        fio.save_multipage_tiff(newsepstack, savefilename)
        
        
##==============================================================================
###   Registration: Register dataset. (similarity transform) (calls matlab for now.), how to go to something else?
###==============================================================================
#    dataset_files = fio.load_dataset(out_pre_folder, ext='.tif', split_key='TP_',split_position=1)
#    
#    """
#    Declare similarity registration settings
#    """
#    reg_config_similarity = {'downsample' :4.0, 
#                             'modality' :'multimodal', 
#                             'max_iter':500,
#                             'type': 'similarity',
#                             'return_img':0}
#    
#    sim_tforms = registration.matlab_register_similarity_batch(dataset_files, out_pre_folder, out_aligned_folder, reg_config_similarity, timer=True, debug=True)
    
#==============================================================================
#   Registration: Register dataset. (similarity transform - no shear). we try using SIFT.
#==============================================================================
    dataset_files = fio.load_dataset(out_pre_folder, ext='.tif', split_key='TP_',split_position=1)
    
    """
    Declare similarity registration settings
    """
#    reg_config_similarity = {'downsample' :4.0, 
#                             'modality' :'multimodal', 
#                             'max_iter':500,
#                             'type': 'similarity'}
    # mode 2 = sequential registration mode. 
    reg_config = {'downsample': 4., #8  ant
                  'lib_path': '/home/felix/Documents/Software/SIFT3D/build/lib/wrappers/matlab', 
                  'mode':2,
                  'return_img':0}
    
#    sim_tforms = registration.matlab_register_similarity_batch(dataset_files, out_pre_folder, out_aligned_folder, reg_config_similarity, timer=False, debug=True)
    tforms = registration.register3D_SIFT_wrapper(dataset_files[5:7], out_pre_folder, out_aligned_folder, reg_config)


#    def combine_tforms(tform1, tform2):
#        
#        mat1 = tform1[:3,:3]
#        mat2 = tform2[:3,:3]
#        t1 = tform1[:3,-1]
#        t2 = tform2[:3,-1]
#        
#        mat = mat1.dot(mat2)
#        t = t1+t2
#        
#        tform = np.zeros((4,4))
#        tform[:3,:3] = mat
#        tform[:3,-1] = t
#        tform[-1] = [0,0,0,1]
#        
#        return tform
#        
#    
#    """
#    check the performance.
#    """
#    im1 = fio.read_multiimg_PIL(dataset_files[0])
##    im2 = fio.read_multiimg_PIL(dataset_files[1])
#    im3 = fio.read_multiimg_PIL(dataset_files[2])
#    plt.figure()
#    plt.imshow(im3[:,1000])
#    
#    t = [tforms[0][0], tforms[0][1]]
#    t_ = [tforms[1][0], tforms[1][1]]
#    
#    t1 = t[0].copy(); #t1[:3,-1] += t_[0][:3,-1]
#    t2 = t[1].copy(); #t2[:3,-1] += t_[1][:3,-1]
#    
##    tform = combine_tforms(tform1, tform2)
#    from Geometry import transforms as tf
#    
#    t1_ = np.zeros((4,4)); t1_[:3] = t1; t1_[-1] = [ 0, 0, 0, 1]
#    t2_ = np.zeros((4,4)); t2_[:3] = t2; t2_[-1] = [ 0, 0, 0, 1]
#    
#    tt = combine_tforms(t1_, t2_)
#    
#    
#    im2 = np.uint8(tf.apply_affine_tform(im3.transpose(1,2,0), tt, np.array(im1.shape)[[1,2,0]]))
#    im2 = im2.transpose(2,0,1)
#    
#    
#    test = np.dstack([im1[:,1000], im2[:,1000], np.zeros_like(im1[:,1000])])
#    test = np.dstack([im1[:,1000], im3[5:-5,1000], np.zeros_like(im1[:,1000])])
#    
    
    
    
#    dataset_files = fio.load_dataset(out_aligned_folder, ext='.tif', split_key='TP_',split_position=1)
#    
#    mid = []
#    
#    for data in dataset_files:
#        im = fio.read_multiimg_PIL(data)
#        sli = im[:,1000]
#        mid.append(sli)
#        
#        plt.figure()
#        plt.imshow(sli)
#        plt.show()
#        
#    mid = np.array(mid)
        
    
        
##==============================================================================
##   2nd Round of translation registration. 
##==============================================================================
#    dataset_files = fio.load_dataset(out_aligned_folder, ext='.tif', split_key='TP_',split_position=1) # load in the just aligned files. 
#    
#    """
#    Declare similarity registration settings
#    """
#    reg_config_similarity = {'nbins' : 50, 
#                             'metric' :'mutualinfo', 
#                             'sampling_prop': None,
#                             'level_iters': [10000,1000,100],
#                             'sigmas':[3.0, 1.0, 0.0], 
#                             'factors':[4, 2, 1]}
#    
#    translation_tforms = registration.dipy_register_translation_batch(dataset_files, out_aligned_folder, out_aligned_folder2, reg_config_similarity, timer=False, debug=True)
#    
#
###==============================================================================
###   Visual registration check
###==============================================================================
##  
#    dataset_files = fio.load_dataset(out_aligned_folder2, ext='.tif', split_key='TP_',split_position=1) # load in the just aligned files.
#    
#    mid_frames = []
#    # load everything and extract the central frames
#    for fil in dataset_files:
#        vid = fio.read_multiimg_PIL(fil)
#        mid_frames.append(vid[vid.shape[0]//2])
#    mid_frames = np.array(mid_frames)
#        
#    plt.figure()
#    plt.imshow(np.mean(mid_frames, axis=0))
#    plt.show()
#    
#
##==============================================================================
##   Optical Flow computation in 3D (calls matlab based RTTtracker scripts)
##==============================================================================
#
#    print 'performing optical flow registration'
#    flow_config = {'method':2, 
#                   'refid':1,
#                   'refine_level':1,
#                   'accFactor':1,
#                   'downsample_factor':1, 
#                   'alpha':0.1,
#                   'I_thresh': 15,
#                   'lib_path':'/home/felix/Documents/PhD/Software/3D optical Flow/LKPR3D/RTTracker_v02'}
#    
#    # this will save the output of the optflow computations into that of the optflow folder. 
#    # what is the normal computation time? -> seems a bit slow. 
#    optflow.RTT_optflow3d_batch(dataset_files, out_aligned_folder2, out_flow_folder, flow_config, timer=True) 
#    
#    
##==============================================================================
##   Load the optical flow files and compute the superpixel tracks 
##==============================================================================
#    
#    n_spixels = 20000
#    I_thresh = 10
#    masked = True
#    dense = False # not yet implemented. 
#    
#    dataset_files = fio.load_dataset(out_aligned_folder2, ext='.tif', split_key='TP_',split_position=1) # load in the just aligned files.
#    optflow_files = fio.load_dataset(out_flow_folder, ext='.mat', split_key='field_',split_position=1) # load in the just aligned files.
#    
#    meantracks3D = svoxel_tracks.compute3D_superpixel_tracks(dataset_files, optflow_files, key_flow='motion_field', 
#                                                             n_spixels=n_spixels, spixel_spacing=None, masked=masked, dense=dense, I_thresh=I_thresh)
#    
#    
##    # visualize the tracks ? # implement a visualiser. 
#    spio.savemat('test_meantracks3D.mat', {'meantracks':meantracks3D})       
#    
##==============================================================================
##   Correct for drift in 3D to reveal the brownian motion. 
##==============================================================================    
#    meantracks3D = spio.loadmat('test_meantracks3D.mat')['meantracks']
#    
#    
##    track3D_tpy = svoxel_tracks.tracks2tpy_3D(meantracks3D)
###    mtrack3D = svoxel_tracks.tpy3D_2tracks(track3D_tpy)
## 
##    import trackpy as tp 
##    drift3D = tp.compute_drift(track3D_tpy, pos_columns=['x','y','z'])
##    meantracks3D_dedrift = tp.subtract_drift(track3D_tpy, drift=drift3D, inplace=False)
##    
##    meantracks3D_dedrift = svoxel_tracks.tpy3D_2tracks(meantracks3D_dedrift)
#    
##==============================================================================
##   Compute MOSES mesh if desired. (need to verify this accuracy by unzipping (ofc there will be distortion))
##==============================================================================
#
#    mean_pos, mean_disps = svoxel_tracks.compute_mean_displacement3D_vectors(meantracks3D)
##    vimgviz.viz_vector_field(mean_pos, mean_disps)
#    
#    # read in a test volume
#    test_volume = fio.read_multiimg_PIL(dataset_files[0])
#    # cut through the volume? 
#    vimgviz.viz_vector_field(mean_pos, mean_disps, vscale=3, scale_factor=7, 
#                             volumeimg=test_volume, thresh=10, sampling=1, vsampling=1, cmap='gray', opacity=0.3)
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        