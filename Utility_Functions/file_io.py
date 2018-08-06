#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:49:32 2017

@author: felix
"""
import numpy as np


# read a single frame from a multi-page .tif file.
def read_PIL_frame(tiffile, frame):

    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)
    
    Output:
    -------
    an image as a numpy array either (n_rows x n_cols) for grayscale or (n_rows x n_cols x 3) for RGB
    
    """
    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    img.seek(frame)

    return np.array(img)
    

def read_multiimg_PIL(tiffile):
    
    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)

    Output:
    -------
    a numpy array that is either:
        (n_frames x n_rows x n_cols) for grayscale or 
        (n_frames x n_rows x n_cols x 3) for RGB

    """

    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    imgs = []
    read = True

    frame = 0

    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break

    return np.concatenate(imgs, axis=0)


def save_multipage_tiff(np_array, savename):
    
    """
    save numpy array of images as a multipage tiff file.... 
    
    Input:
    =====
    np_array: (n_frames, n_rows, n_cols)
    savename: filepath to save the output .tif stack. 
    
    Output:
    =====
    void function
    
    """    
    from tifffile import imsave
    
    if np_array.max() < 1.1:
    	imsave(savename, np.uint8(255*np_array))
    else:
	imsave(savename, np.uint8(np_array))
    
    return [] 


def load_dataset(infolder, ext='.tif', split_position=0, split_key='_'):
    
    import os 
    
    f = os.listdir(infolder)
    
    files = []
    
    for ff in f:
        if ext in ff:
            frame_No = int((ff.split(ext)[0]).split(split_key)[split_position])
            files.append([frame_No, os.path.join(infolder, ff)])
            
    files = sorted(files, key=lambda x: x[0])
    
    return np.hstack([ff[1] for ff in files])
    
def mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []

def get_basename(pathname, ext):
    
    return (pathname.split('/')[-1]).split(ext)[0]
    


def load_opt_flow_files(infolder, key='.mat', keyword='flow'):
    
    import os 
    
    files = os.listdir(infolder)
    
    f = []

    for ff in files:
        if key in ff and keyword in ff:
            ind = ff.split(key)[0]
            ind = int(ind.split('_')[-1])
            f.append([os.path.join(infolder, ff), ind])
        
    f = sorted(f, key=lambda x: x[1])       
    
    return np.array(f)[:,0]


def read_optflow_file(optflowfile, key='motion_field'):
    
    # result is 4D vector, which is x,y,z, 3D flow. 
    import scipy.io as spio 
    
    obj = spio.loadmat(optflowfile)
    
    return obj['motion_field']

    
    
    
    
    
    
