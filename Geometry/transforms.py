#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:35:20 2018

@author: felix
"""
#import Geometry.geometry as geom
import geometry as geom

def apply_affine_tform(volume, matrix, sampling_grid_shape=None, check_bounds=False, contain_all=False, domain_grid_shape=None, codomain_grid_shape=None, domain_grid2world=None, codomain_grid2world=None, sampling_grid2world=None):
    
    """
    given a homogeneous transformation matrix, create an affine matrix and use dipy to apply the transformation.
    """
    
    import numpy as np 
    from dipy.align.imaffine import AffineMap
    
    if domain_grid_shape is None:
        domain_grid_shape = volume.shape
    if codomain_grid_shape is None:
        codomain_grid_shape = volume.shape
        
    if check_bounds:
        if contain_all:
            in_out_corners, out_shape, tilt_tf_ = compute_transform_bounds(domain_grid_shape, matrix, contain_all=True)
        else:
            in_out_corners, out_shape = compute_transform_bounds(domain_grid_shape, matrix, contain_all=False)
            tilt_tf_ = None
#        print out_shape
    affine_map = AffineMap(matrix,
                               domain_grid_shape=domain_grid_shape, domain_grid2world=domain_grid2world,
                               codomain_grid_shape=codomain_grid_shape, codomain_grid2world=codomain_grid2world)
        
    if check_bounds:
        out = affine_map.transform(volume, sampling_grid_shape=out_shape, sampling_grid2world=tilt_tf_)
    else:
        out = affine_map.transform(volume, sampling_grid_shape=sampling_grid_shape, sampling_grid2world=sampling_grid2world)
        
    return out

# helper function to determine. 
def general_cartesian_prod(objs):
    
    import itertools
    import numpy as np 
    
    out = []
    for element in itertools.product(*objs):
        out.append(np.array(element))
        
    out = np.vstack(out)
    
    return out


def compute_transform_bounds(im_shape, tf, contain_all=True):
    
    import numpy as np 
    obj = [[0, ii-1] for ii in list(im_shape)] # remember to be -1 
    in_box_corners = general_cartesian_prod(obj)
        
    # unsure whether its left or right multiplication ! - i believe this should be the left multiplcation i,e, pts x tf. 
    out_box_corners = tf.dot(np.vstack([in_box_corners.T, np.ones(in_box_corners.shape[0])[None,:]]))[:-1].T
  
    in_out_corners = (in_box_corners, out_box_corners)
    
    if contain_all:
        out_shape = np.max(out_box_corners, axis=0) - np.min(out_box_corners, axis=0)
        out_shape = (np.rint(out_shape)).astype(np.int)
        
        # to shift the thing, we need to change the whole world grid!. 
        mod_tf = np.min(out_box_corners, axis=0)  # what is the required transformation parameters to get the shape in line? # double check? in terms of point clouds?
        tf_mod = np.eye(4)
        tf_mod[:-1,-1] = mod_tf # this is the one to change ffs. # reverse this. 
        
        # this adjustment needs to be made in a left handed manner!. 
        # here we probably need to create an offset matrix then multiply this onto the tf_ which is a more complex case.... of transformation? # to avoid sampling issues. 
        return in_out_corners, out_shape, tf_mod
        
    else:
        out_shape = np.max(out_box_corners,axis=0) # this should touch the edges now right? #- np.min(out_box_corners,axis=0) # not quite sure what dipy tries to do ?
        out_shape = (np.rint(out_shape)).astype(np.int)
    
        return in_out_corners, out_shape

    
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

def rotation_matrix_xyz(angle_x,angle_y, angle_z, center=None, imshape=None):
    
    import numpy as np 
    affine_x = geom.get_rotation_x(angle_x)
    affine_y = geom.get_rotation_y(angle_y) # the top has coordinates (r, np.pi/2. 0)
    affine_z = geom.get_rotation_z(angle_z)
    affine = affine_z.dot(affine_x.dot(affine_y))
    
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
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=imshape)
    else:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=out_shape)
        
    return rot_mat, vol_out


def correct_axial_tilt_manual(vol, vol_center, angle_x, angle_y, use_im_center=True, out_shape=None):
    """
    image: x,y,z format
    
    manual correction of the tilt of an object by manually specifying the angles at which the volume deviates from the center 'z' line that passes through the embryo centroid.
    
    """
    import numpy as np 
    
    angle_x = angle_x/180. * np.pi
    angle_y = angle_y/180. * np.pi
       
    # construct the correction matrix and transform the image. 
    imshape = vol.shape
    if use_im_center:
        rot_mat = correct_tilt_matrix(-angle_x, -angle_y, vol_center, imshape)
    else:
        rot_mat = correct_tilt_matrix(-angle_x, -angle_y, vol_center)
        
    if out_shape is None:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=imshape)
    else:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=out_shape)
        
    return rot_mat, vol_out
        

def rotate_volume_xyz(vol, vol_center, angle_x, angle_y, angle_z, use_im_center=True, out_shape=None):
    """
    image: x,y,z format
    
    rotation of a volumetric object by manual rotation angle specification
    
    """
    import numpy as np 
    
    angle_x = angle_x/180. * np.pi
    angle_y = angle_y/180. * np.pi
    angle_z = angle_z/180. * np.pi
       
    # construct the correction matrix and transform the image. 
    imshape = vol.shape
    if use_im_center:
        rot_mat = rotation_matrix_xyz(-angle_x, -angle_y, -angle_z, vol_center, imshape)
    else:
        rot_mat = rotation_matrix_xyz(-angle_x, -angle_y, -angle_z, vol_center)
        
    if out_shape is None:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=imshape)
    else:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=out_shape)
        
    return rot_mat, vol_out
