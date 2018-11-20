#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:14:17 2018

@author: felix
"""

import numpy as np 
import Utility_Functions.file_io as fio


def pad_vol_2_size(im, shape):
    
    shift_1 = (shape[0] - im.shape[0]) // 2
    shift_2 = (shape[1] - im.shape[1]) // 2
    shift_3 = (shape[2] - im.shape[2]) // 2
    
    new_im = np.zeros(shape, dtype=np.uint8)
    new_im[shift_1:shift_1+im.shape[0],
           shift_2:shift_2+im.shape[1],
           shift_3:shift_3+im.shape[2]] = im.copy()

    return new_im

def mean_vol_img(vol_img_list, target_shape):

    # work in float but convert to ubytes for allocation? 
    mean_vol = np.zeros(target_shape) 
    n_imgs = len(vol_img_list)

    for v in vol_img_list:
        im = fio.read_multiimg_PIL(v)
        im = pad_vol_2_size(im, target_shape) 
        mean_vol += (im/255.)/n_imgs
        
    return np.uint8(255*mean_vol)


def max_vol_img(vol_img_list, target_shape):

    # work in float but convert to ubytes for allocation? 
    mean_vol = np.zeros(target_shape) 
#    n_imgs = len(vol_img_list)

    for v in vol_img_list:
        im = fio.read_multiimg_PIL(v)
        im = pad_vol_2_size(im, target_shape) 
        mean_vol = np.maximum(mean_vol, im)       
        
    return mean_vol

        
        