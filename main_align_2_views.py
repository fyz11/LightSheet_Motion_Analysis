#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:03:04 2018

@author: felix
"""

    
if __name__=="__main__":
    
    import numpy as np 
    import Utility_Functions.file_io as fio
    import Utility_Functions.stack as stack
    from Visualisation.grid_img import viz_grid
    import pylab as plt 
    import Registration.registration as registration
    from skimage.exposure import rescale_intensity 
    
    import Unzipping.unzip as uzip
    import scipy.io as spio 
    import os
   

    dataset_folder = '../../Data/Holly/czifile_test_tiff'
    out_view_aligned_folder = os.path.join(dataset_folder, 'view_aligned3'); fio.mkdir(out_view_aligned_folder)
    
    
    """
    Load dataset and pair up data. 
    """
    dataset_files = fio.load_dataset(dataset_folder, ext='.tif', split_key='_',split_position=3) # load in the just aligned files.
    view_pair_files = fio.pair_views(dataset_files, ext='.tif', split_key='_',split_position=3, view_by=2)

    
    """
    Do Sift3D registration to align the sequence of paired views. 
    """
    processfiles = np.hstack(view_pair_files)
   
    reg_config = {'downsample': 8., 
                  'lib_path': '../Pipeline/SIFT3D/build/lib/wrappers/matlab/', 
                  'mode':1,
                  'return_img':0}
                  
    # register views: mode=1
    tforms = registration.register3D_SIFT_wrapper(dataset_files, dataset_folder, out_view_aligned_folder, reg_config)



    
    
    
    
