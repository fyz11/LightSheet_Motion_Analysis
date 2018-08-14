#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:35:20 2018

@author: felix
"""

def apply_affine_tform(volume, matrix, sampling_grid_shape=None):
    
    """
    given a homogeneous transformation matrix, create an affine matrix and use dipy to apply the transformation.
    """
    
    import numpy as np 
    from dipy.align.imaffine import AffineMap
    
    identity = np.eye(4)
    affine_map = AffineMap(matrix,
                           volume.shape, identity,
                           volume.shape, identity)
    
    
    out = affine_map.transform(volume, sampling_grid_shape=sampling_grid_shape)
    
    return out
    

        