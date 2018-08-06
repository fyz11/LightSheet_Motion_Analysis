#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:49:32 2017

@author: felix
"""
import pylab as plt
import numpy as np 

def imshowpair(ax,im1,im2):

    from skimage.exposure import rescale_intensity
    img = np.dstack([im1, im2, np.zeros_like(im2)])
    ax.imshow(rescale_intensity(img))
    
    return []
