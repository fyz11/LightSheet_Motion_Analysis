#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:35:20 2018

@author: felix
"""
import Geometry.geometry as geom

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
    
def correct_tilt_matrix(angle_x,angle_y, center=None, imshape=None):
    
    import numpy as np 
    affine_x = geom.get_rotation_x(angle_x)
    affine_y = geom.get_rotation_y(angle_y) # the top has coordinates (r, np.pi/2. 0)
    affine = affine_x.dot(affine_y)
    
    if center is not None:
        affine[:-1,-1] = center 
    if imshape is not None:
        decenter = np.eye(4); decenter[:-1,-1] = [-imshape[0]//2, -imshape[1]//2, -center[2]]
    else:
        decenter = np.eye(4); decenter[:-1,-1] = -center
    T = affine.dot(decenter)
    
    return T
    
    
def correct_axial_tilt(vol, I_thresh=10, ksize=3, mode='pca', pole='S', mask=None, use_im_center=True, out_shape=None):
    """
    image: x,y,z format
    
    estimates the axial tilt of a volume image either using the specified pole or using eigenvalue analysis. 
    """
    import Unzipping.unzip as uzip
    import numpy as np 
    import sys
    
    vol_ = vol.copy()

    # optionally use voxel mapping. 
    if mask is not None:
        vol_[mask==1] = 0
    # fast segment the volume image.  
    _, border = uzip.segment_contour_embryo(vol_, I_thresh=I_thresh, ksize=ksize)
    

    coords = np.array(np.where(border>0)).T
    center = np.mean(coords, axis=0) # get the center. 
    
    if mode=='pole':
        
        """
        method 1: find the largest distant point to correct for the tilt. 
        """
        r, polar, azi = geom.xyz_2_spherical(coords[:,0],
                                             coords[:,1],
                                             coords[:,2], center=center)
        thresh = np.pi/2.
        
        if pole == 'S':
            select = polar < thresh 
        if pole == 'N':
            select = polar >= thresh
            
        emb_pole = (coords[select])[np.argmax(r[select])]

    if mode == 'pca': # this is more stable. 
        
        """
        method 2: Find the principal rotation vectors using PCA analysis. 
        """
        
        points = coords - center[None,:] # de-mean
        
        cov = np.cov(points.T) # 3 x N.
        evals, evecs = np.linalg.eig(cov)
        
        sort_indices = np.argsort(evals)[::-1]
        
        emb_pole = evecs[:,sort_indices] # principal vectors. 
        emb_pole = emb_pole[:,0]
        
    """
    compute the correction angle. 
    """
    if mode=='pole':
        angle_y = -np.arctan2((emb_pole[0]-center[0]),(emb_pole[2]-center[2])) # negative is for image coordinate conventions! which is LH axis.
        angle_x = np.arctan2((emb_pole[1]-center[1]),(emb_pole[2]-center[2])) # image uses a LH coordinates not RH coordinates!
    if mode == 'pca':
        angle_y = -np.arctan2(emb_pole[0], emb_pole[2]) # negative is for image coordinate conventions! which is LH axis.
        angle_x = np.arctan2(emb_pole[1], emb_pole[2]) # image uses a LH coordinates not RH coordinates!
    

    # constrain this to lie only in +/- np.pi/2 (help correct without flipping.)
    if np.abs(angle_y) >np.pi/2.:
        angle_y = -np.sign(angle_y) * (np.pi-np.abs(angle_y)) # push in positive direction.
    if np.abs(angle_x) >np.pi/2.:
        angle_x = -np.sign(angle_x) * (np.pi-np.abs(angle_x))

    
    print('volumed is tilted by: ', 'angle_x-', angle_x/np.pi*180, 'angle_y-', angle_y/np.pi*180)

        
    # construct the correction matrix and transform the image. 
    imshape = vol.shape
    if use_im_center:
        rot_mat = correct_tilt_matrix(-angle_x, -angle_y, center, imshape)
    else:
        rot_mat = correct_tilt_matrix(-angle_x, -angle_y, center)
        
    if out_shape is None:
        vol_out = apply_affine_tform(vol, rot_mat, imshape)
    else:
        vol_out = apply_affine_tform(vol, rot_mat, out_shape)
        
    return rot_mat, vol_out
        
        