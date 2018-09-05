#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:22:56 2018

@author: felix
"""


def load_CNN_model(filename):
    
    from keras.models import load_model
    
    model = load_model(filename)
    
    return model

    
def enhance_volume(im, model, n_pad=3, max_scale=16, I_thresh=15):
    
    from tqdm import tqdm 
    import numpy as np 
    import pylab as plt 
    
    n_z, n_rows, n_cols = im.shape
        
    pad_im = np.pad(im, [[n_pad//2, n_pad//2],[0,0],[0,0]], mode='reflect')
    
#    n_rows_ = int(2**(np.ceil(np.log2(n_rows)))) # find the nearest power of 2. 
#    n_cols_ = int(2**(np.ceil(np.log2(n_cols)))) 

    n_rows_ = max_scale*int(np.ceil(n_rows/float(max_scale))) # find the nearest power of 2. 
    n_cols_ = max_scale*int(np.ceil(n_cols/float(max_scale))) 
    
    out_stack = []
    
    for i in range(n_z):
        x = pad_im[i:i+n_pad]/255.; x = x.transpose(1,2,0)
        
        if np.sum(x>=I_thresh/255.) == 0:
            out = np.zeros((n_rows, n_cols))
            
        else:
            in_im = np.zeros((n_rows_, n_cols_, n_pad))
            in_im[:n_rows,:n_cols] = x.copy() # enhancing the contrast introduces too much noise
            
            y = model.predict(in_im[None,:])
            y = np.squeeze(y); y = y[:n_rows,:n_cols]
            y = y[:,:,0]
#            out = np.uint8(255*rescale_intensity(y[:,:,0]))
            if np.sum(y>0.01):            
                out = rescale_intensity(y)
            else:
                out = y
        
        out_stack.append(out)

    return np.array(out_stack)        
        

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
    from skimage.transform import rescale
    
#==============================================================================
#     load in the dataset 
#==============================================================================

    dataset_folder = '/media/felix/Elements/Shankar LightSheet/Example Timelapse/test'
    
    """
    Create output folders.
    """
    out_pre_folder = os.path.join(dataset_folder, 'preprocess_full'); fio.mkdir(out_pre_folder)
    out_enhance_folder = os.path.join(dataset_folder, 'enhance_scaled'); fio.mkdir(out_enhance_folder)
    out_aligned_folder = os.path.join(dataset_folder, 'align-enhance_scaled-mm'); fio.mkdir(out_aligned_folder)
    
##==============================================================================
##   Registration: Register dataset. (similarity transform - no shear). we try using SIFT.
##==============================================================================
#    dataset_files = fio.load_dataset(out_pre_folder, ext='.tif', split_key='TP_',split_position=1)
#    
#    
#    modelfile = 'Models/Felix-full_mse_model.h5'
#    model = load_CNN_model(modelfile)
#
#
#    for dataset_file in tqdm(dataset_files[1:]):
#        im = fio.read_multiimg_PIL(dataset_file)
#        
#        im = rescale(im, scale=1/2., mode='reflect', preserve_range=True, multichannel=False, anti_aliasing=True)
#        im = np.clip(im, 0, 255) # clip into the range. 
#        
#        # conduct the enhancement
#        im_ = enhance_volume(im, model, n_pad=3)
#    
#        fio.save_multipage_tiff( np.uint8(255*im_), dataset_file.replace(out_pre_folder, out_enhance_folder))
        
        
##==============================================================================
##  Registration
##==============================================================================
#    dataset_files = fio.load_dataset(out_enhance_folder, ext='.tif', split_key='TP_',split_position=1)
#    
#    """
#    Declare similarity registration settings
#    """
#    reg_config_similarity = {'downsample' :8.0, 
#                             'modality' :'multimodal', 
#                             'max_iter':500,
#                             'type': 'similarity',
#                             'return_img':0}
##    # mode 2 = sequential registration mode. 
##    reg_config = {'downsample': 4., #8  ant
##                  'lib_path': '/home/felix/Documents/Software/SIFT3D/build/lib/wrappers/matlab', 
##                  'mode':2,
##                  'return_img':0}
#    
#    sim_tforms = registration.matlab_register_similarity_batch(dataset_files, out_enhance_folder, out_aligned_folder, reg_config_similarity, timer=True, debug=True)
##    tforms = registration.register3D_SIFT_wrapper(dataset_files[:5], out_pre_folder, out_aligned_folder, reg_config)
#    
#    
    
    
    
    
    
    
    
        
        
        
        
        