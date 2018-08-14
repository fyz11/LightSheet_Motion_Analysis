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
    from tqdm import tqdm
    # how much to pad. 
    n_z, n_y, n_x = im.shape
    
    im_new = np.zeros((n_z*pad_slices, n_y, n_x), dtype=np.uint8)
    X, Y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    
    # linearly interpolate between layers. 
    for i in tqdm(range(n_z-1)):
        cnt = 0

        # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
        arr = np.r_['0,3', im[i], im[i+1]]
        
        # linear interpolation. 
        for j in range(pad_slices*i, pad_slices*(i+1)):
            coordinates = np.ones((n_y,n_x)) * 1./pad_slices * cnt, Y, X 
            newarr = ndimage.map_coordinates(arr, coordinates, order=1)

            im_new[j] = np.uint8(newarr).copy()
            cnt+=1
            
    for i in range(n_z-1, n_z):
        for j in range(pad_slices*i, pad_slices*(i+1)):
            im_new[j] = im[i].copy()
        
    return im_new
    

def pad_z_stack_adj(im, pad_slices=4, min_I=15, min_count=200):
    
    """
    adjustably adds and interpolates 
    """
    import scipy.ndimage as ndimage
    from tqdm import tqdm
    # how much to pad. 
    n_z, n_y, n_x = im.shape
    
#    im_new = np.zeros((n_z*pad_slices, n_y, n_x), dtype=np.uint8)
    im_new = []
    X, Y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    
    # linearly interpolate between layers. 
    for i in tqdm(range(n_z-1)):
        cnt = 0

        # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
        arr = np.r_['0,3', im[i], im[i+1]]
        
        if np.sum(arr>min_I) < 2*min_count:
            im_new.append(im[i+1]) # add the one. 
        else:
            # do some padding. 
            # linear interpolation. 
            for j in range(pad_slices*i, pad_slices*(i+1)):
                coordinates = np.ones((n_y,n_x)) * 1./pad_slices * cnt, Y, X 
                newarr = ndimage.map_coordinates(arr, coordinates, order=1)
    
                im_new.append(np.uint8(newarr))
                cnt+=1
            
    for i in range(n_z-1, n_z):
        # for the last slice. 
        im = im[i]
        if np.sum(im>min_I) < min_count:
            im_new.append(im)
        else:
            for j in range(pad_slices*i, pad_slices*(i+1)):
                im_new.append(im)
                
    return np.uint8(np.array(im_new)) # return the new array.


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
    
    
def bounding_box(mask3D):
    
    import numpy as np 
    
    coords = np.where(mask3D>0)
    coords = np.array(coords).T
    
    min_x, max_x = np.min(coords[:,0]), np.max(coords[:,0])
    min_y, max_y = np.min(coords[:,1]), np.max(coords[:,1])
    min_z, max_z = np.min(coords[:,2]), np.max(coords[:,2])
    
    return np.hstack([min_x, min_y, min_z, max_x, max_y, max_z])
    
    
def expand_bounding_box(bbox3D, clip_limits, border=50, border_x=None, border_y=None, border_z=None):
    
    import numpy as np 
    
    clip_x, clip_y, clip_z = clip_limits
    
    new_bounds = np.zeros_like(bbox3D)
    
    for i in range(len(new_bounds)):
        if i==0:
            if border_x is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_x, clip_x[0], clip_x[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_x[0], clip_x[1])
        if i==3:
            if border_x is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_x, clip_x[0], clip_x[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_x[0], clip_x[1])
        if i==1:
            if border_y is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_y, clip_y[0], clip_y[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_y[0], clip_y[1])
        if i==4:
            if border_y is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_y, clip_y[0], clip_y[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_y[0], clip_y[1])
        if i==2:
            if border_z is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_z, clip_z[0], clip_z[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_z[0], clip_z[1])
        if i==5:
            if border_z is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_z, clip_z[0], clip_z[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_z[0], clip_z[1])
    
    return new_bounds
    

def crop_img_2_box(volume, bbox3D):
    
    x1,y1,z1,x2,y2,z2 = bbox3D
    
    return volume[x1:x2,y1:y2,z1:z2]
    
    
