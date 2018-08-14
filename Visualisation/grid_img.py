#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:20:44 2018

@author: felix


This script contains scripts to visualise a panel of images easily using matplotlib in a grid like manner

"""

def viz_grid(im_array, shape=None, cmap='gray', figsize=(10,10)):
    
    """
    im_array: n_img x n_rows x n_cols x 1/3
    """
    # shape is the desired size. (should be specified else takes the sqrt. )
    import pylab as plt 
    import numpy as np 
    
    n_imgs = len(im_array)
    
    if shape is not None:
        nrows, ncols = shape
    else:
        nrows = int(np.ceil(np.sqrt(n_imgs)))
        ncols = nrows
    
    color=True
    if len(im_array.shape) == 3:
        color=False
    if len(im_array.shape) == 4 and im_array.shape[-1]==1:
        color=False
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    for i in range(nrows):
        for j in range(ncols):
            im_index = i*ncols + j 
            if im_index<n_imgs:
                if color:
                    ax[i,j].imshow(im_array[im_index])
                else:
                    ax[i,j].imshow(im_array[im_index], cmap=cmap)
            # switch off all gridlines. 
            ax[i,j].axis('off')
            ax[i,j].grid('off')
           
    # return the plot object. 
    return (fig, ax) 
    
    

