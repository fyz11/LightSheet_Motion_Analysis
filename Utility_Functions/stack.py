#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:39:07 2018

@author: felix

Various image analysis algorithms to work on stacks of 3D volumes 

"""
import numpy as np 

def pad_z_stack(im, pad_slices=4):
    
    import scipy.ndimage as ndimage
    
    # how much to pad. 
    n_z, n_y, n_x = im.shape
    
    im_new = np.zeros((n_z*pad_slices, n_y, n_x))
    
    # linearly interpolate between layers. 
    for i in range(n_z-1):
        cnt = 0
        arr1 = im[i].copy()
        arr2 = im[i+1].copy()
        
        # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
        arr = np.r_['0,3', arr1, arr2]
        
        X, Y = np.meshgrid(np.arange(n_x), np.arange(n_y))
        
        for j in range(pad_slices*i, pad_slices*(i+1)):
            coordinates = np.ones((n_y,n_x)) * 1./pad_slices * cnt, Y, X 
            newarr = ndimage.map_coordinates(arr, coordinates, order=1)

#            im_new[j] = ((cnt)*(1./pad_slices) * (im[i+1] - im[i]) + im[i]).copy() # linear interpolation
            im_new[j] = newarr.copy()
            cnt+=1
            
    for i in range(n_z-1, n_z):
        for j in range(pad_slices*i, pad_slices*(i+1)):
            im_new[j] = im[i].copy()
        
    return im_new


def downsample_stack(vidstack, scale=1./2):

    from skimage.transform import rescale
    imgs = []

    for im in vidstack:
        imgs.append(rescale(im, scale=scale, mode='reflect'))
        
    return np.array(imgs)


def split_stack(vidstack, n_splits=2):

    """
    This is only if there is multiple channels that have been flattened. e.g. by ImageJ
    """

    stacks = []

    for i in range(n_splits):
    	stacks.append(vidstack[0+i::n_splits])

    return stacks
    
