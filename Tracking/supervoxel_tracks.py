#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:53:48 2018

@author: felix
"""

import Utility_Functions.file_io as fio
import numpy as np
import scipy.io as spio 
    
# uses vectorization to efficiently track regular superpixels over time, propagating their centroid positions according to the computed optical flow from frame to frame.
def prop_spixels_frame_by_frame_forward3D( spixel_pos, optflowfiles, spixel_size, key_flow='motion_field'):

    """
    Propagates each superpixel centroid according to the average flow within its area from frame to frame to generate superpixel tracks.
    
    Inputs:
    -------
    spixel_pos: (n_superpixels x 3) numpy array of superpixel centroids given as (y,x)-image coordinates
    optflowfiles: (n_rows, n_cols, n_z, 3) numpy array of frame-to-frame optical flow
    spixel_size: estimated mean superpixel width assuming they are square.
    
    Output:
    -------
    meantracks: (n_superpixels, n_frames, 3) numpy array
    
    """
    
    import itertools
    import numpy as np
    from tqdm import tqdm 
    
    radius = int(np.ceil((spixel_size)/2.)) # compute the radii.

    # load the first optflowfile to retrieve info
    optflow0 = fio.read_optflow_file(optflowfiles[0], key=key_flow)
    ny, nx, nz, _ = optflow0.shape
    nframes = len(optflowfiles)
    nspixels = spixel_pos.shape[0]

    # set the boundary limits for propagation. (and clipping)
    x_limits = [0, nx]
    y_limits = [0, ny]
    z_limits = [0, nz]

    meantracks = np.zeros((nspixels, nframes+1, 3), dtype=np.int) # retain only integer pixel positions.
    meantracks[:,0,:] = np.floor(spixel_pos).astype(np.int) # floor to stay in the limits. (x,y,z)
    
    # build the permutation matrix for iteration;
    offset_matrix = [np.array([i,j,k]).ravel() for i,j,k in itertools.product(np.arange(-radius,radius+1), repeat=3)]
    offset_matrix = np.vstack(offset_matrix) # creates a n_pixels x n_d... 

    
    # now we update everything in parallel. 
    for i in tqdm(range(nframes)[:]): 
        
        # 1. compute the mean displacement vectors.
        pos0 = meantracks[:,i,:] # (should be x,y,z) 
        
        spixel_range = offset_matrix
        expand_pos0 = pos0[:,None,:] + spixel_range[None,:] # n_superpixels x n_points x n_coords
        
        # thresh and mask points that do not fit within the image. 
        expand_pos_0 = (np.floor(expand_pos0)).astype(np.int)
        
#        print expand_pos_0[:,:,0].min(), expand_pos_0[:,:,0].max()
        mask_pos_0_x = np.logical_and(expand_pos_0[:,:,0] >=x_limits[0], expand_pos_0[:,:,0] <=x_limits[1]-1)
        mask_pos_0_y = np.logical_and(expand_pos_0[:,:,1] >=y_limits[0], expand_pos_0[:,:,1] <=y_limits[1]-1)
        mask_pos_0_z = np.logical_and(expand_pos_0[:,:,2] >=z_limits[0], expand_pos_0[:,:,2] <=z_limits[1]-1)

        final_mask = (np.logical_and(np.logical_and(mask_pos_0_x, mask_pos_0_y), mask_pos_0_z)).astype(np.int) # this should give 1 where things are valid

        # first apply mask to expanded pos        
        expand_pos_0 *= final_mask[:,:,None]
#        print expand_pos_0.shape    

        final_mask = final_mask.astype(np.float)
        final_mask[final_mask==0] = np.nan # this is so x by number should give nan and we can exploit nansum

        """ read in the optflow field """
        flow_frame = fio.read_optflow_file(optflowfiles[i], key=key_flow) # (y,x,z,3)

        # still index with y,x,z 
        # flow_frame is (y,x,z,3)
        flow_pos_0 = flow_frame[expand_pos_0[:,:,1].ravel(), expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,2].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmean(flow_pos_0, axis=1) # use average. 
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
  
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos # add the x,y,z 
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], x_limits[0], x_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], y_limits[0], y_limits[1]-1)
        pos1[:,2] = np.clip(pos1[:,2], z_limits[0], z_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    

def prop_spixels_frame_by_frame_forward3D_masked( spixel_pos, optflowfiles, imgfiles, spixel_size, thresh=10, key_flow='motion_field'):

    """
    Propagates each superpixel centroid according to the average flow within its area from frame to frame to generate superpixel tracks.
    
    Inputs:
    -------
    spixel_pos: (n_superpixels x 3) numpy array of superpixel centroids given as (y,x)-image coordinates
    optflowfiles: (n_rows, n_cols, n_z, 3) numpy array of frame-to-frame optical flow
    spixel_size: estimated mean superpixel width assuming they are square.
    
    Output:
    -------
    meantracks: (n_superpixels, n_frames, 3) numpy array
    
    """
    
    import itertools
    import numpy as np
    from tqdm import tqdm 
    from skimage.morphology import ball, binary_closing
    
#    full_mask = np.concatenate([mask[:,:,:,None], mask[:,:,:,None], mask[:,:,:,None]], axis=-1)
#    full_mask = full_mask.transpose(1,2,0,3) # to have y,x,z
    radius = int(np.ceil((spixel_size)/2.)) # compute the radii.

    # load the first optflowfile to retrieve info
    optflow0 = fio.read_optflow_file(optflowfiles[0], key=key_flow)
    
    print 'hello'
    print optflow0.max(), optflow0.min(), optflow0.mean()
    print '===='
    
    ny, nx, nz, _ = optflow0.shape
    nframes = len(optflowfiles)
    nspixels = spixel_pos.shape[0]

    # set the boundary limits for propagation. (and clipping)
    x_limits = [0, nx]
    y_limits = [0, ny]
    z_limits = [0, nz]

    meantracks = np.zeros((nspixels, nframes+1, 3), dtype=np.int) # retain only integer pixel positions.
    meantracks[:,0,:] = np.floor(spixel_pos).astype(np.int) # floor to stay in the limits. (x,y,z)
    
    # build the permutation matrix for iteration;
    offset_matrix = [np.array([i,j,k]).ravel() for i,j,k in itertools.product(np.arange(-radius,radius+1), repeat=3)]
    offset_matrix = np.vstack(offset_matrix) # creates a n_pixels x n_d... 

    
    # now we update everything in parallel. 
    for i in tqdm(range(nframes)[:]): 
        
        # 1. compute the mean displacement vectors.
        pos0 = meantracks[:,i,:] # (should be x,y,z) 
        
        spixel_range = offset_matrix
        expand_pos0 = pos0[:,None,:] + spixel_range[None,:] # n_superpixels x n_points x n_coords
        
        # thresh and mask points that do not fit within the image. 
        expand_pos_0 = (np.floor(expand_pos0)).astype(np.int)
        
#        print expand_pos_0[:,:,0].min(), expand_pos_0[:,:,0].max()
        mask_pos_0_x = np.logical_and(expand_pos_0[:,:,0] >=x_limits[0], expand_pos_0[:,:,0] <=x_limits[1]-1)
        mask_pos_0_y = np.logical_and(expand_pos_0[:,:,1] >=y_limits[0], expand_pos_0[:,:,1] <=y_limits[1]-1)
        mask_pos_0_z = np.logical_and(expand_pos_0[:,:,2] >=z_limits[0], expand_pos_0[:,:,2] <=z_limits[1]-1)

        final_mask = (np.logical_and(np.logical_and(mask_pos_0_x, mask_pos_0_y), mask_pos_0_z)).astype(np.int) # this should give 1 where things are valid

        # first apply mask to expanded pos        
        expand_pos_0 *= final_mask[:,:,None]
#        print expand_pos_0.shape    

        final_mask = final_mask.astype(np.float)
        final_mask[final_mask==0] = np.nan # this is so x by number should give nan and we can exploit nansum

        
        """ read in the image for masking (the image should be in .tif) """
        ref_vid = fio.read_multiimg_PIL(imgfiles[i])
        mask = ref_vid > thresh; mask = binary_closing(mask, ball(5)); mask = np.logical_not(mask)
        full_mask = np.concatenate([mask[:,:,:,None], mask[:,:,:,None], mask[:,:,:,None]], axis=-1)
        full_mask = full_mask.transpose(1,2,0,3) # to have y,x,z
        full_mask = full_mask.astype(np.float)
        
        """ read in the optflow field """
        flow_frame = fio.read_optflow_file(optflowfiles[i], key=key_flow) # (y,x,z,3)
        flow_frame = flow_frame * full_mask # mask out. # could actually adopt a smoothened mask? 

        # still index with y,x,z 
        # flow_frame is (y,x,z,3)
        flow_pos_0 = flow_frame[expand_pos_0[:,:,1].ravel(), expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,2].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None] # this had an error before? - what is wrong with this bit ? 

        # this can still be nan....        
        av_flow_pos = np.nanmean(flow_pos_0, axis=1) # use average. 
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement

        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos # add the x,y,z 
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], x_limits[0], x_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], y_limits[0], y_limits[1]-1)
        pos1[:,2] = np.clip(pos1[:,2], z_limits[0], z_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    
    
def find_non_const_tracks(meantracks):
    
    """
    find nonmoving tracks. 
    """
    disps = meantracks[:,1:] - meantracks[:,:-1]
    disps = np.sum(disps**2, axis=-1)
    disps = np.sum(disps, axis=1)
    return disps > 0
    

#==============================================================================
#   extract tracks generally in each slice of the image.     
#==============================================================================
def fetch_tracks_zslice(pos_xyz_initial, axis=2, slice_no=100):
    
    uniq_z = np.unique(pos_xyz_initial[:,axis])
    dist_z = np.abs(uniq_z - slice_no)
    select_z = uniq_z[np.argmin(dist_z)]
                      
    select = pos_xyz_initial[:,axis].astype(np.int) == int(select_z)
    
    return select

#==============================================================================
#   this is a transfer function maybe best to include in track utils for working with tracks? 
#==============================================================================
def tracks2tpy_3D(meantracks):
    
    import numpy as np 
    import pandas as pd 
    
    n_spixels, n_frames, _ = meantracks.shape
    
    tables = []
    
    for i in range(n_spixels):
        pos_xyz = meantracks[i,:,:] 
        frame = np.arange(n_frames)
        particle_id = i*np.ones(len(pos_xyz))
        
        data = np.hstack([pos_xyz, frame[:,None], particle_id[:,None]])
        tab = pd.DataFrame(data, index=np.arange(len(frame)), columns=['x','y','z', 'frame', 'particle']) # in the form of (x,y,z) tuples. 
        tables.append(tab)
        
    tables = pd.concat(tables,ignore_index=True)
    tables.index = np.arange(tables.shape[0])
    
    return tables
    
    
def tpy3D_2tracks(tpy_table):

    # reverts the table in tracks2tpy_3D:
    particle_ids = np.unique(tpy_table['particle'].values)
    
    tracks = []
    
    for i in range(len(particle_ids)):
        particle_id = particle_ids[i]
        data = tpy_table.loc[tpy_table['particle'].values==particle_id]
        data = data.sort_values(by='frame', axis='index')
        
        tracks.append(data.loc[:,'x':'z'].values)
        
    return np.array(tracks)
    
    
def compute_mean_displacement3D_vectors(meantracks3D):
    
    n_spixels, n_frames, n_D = meantracks3D.shape
    
    mean_pos = np.mean(meantracks3D, axis=1)
    disps = (meantracks3D[:,1:] - meantracks3D[:,:-1]).astype(np.float)
    mean_disps = np.mean(disps, axis=1)
    
    return mean_pos, mean_disps
    
    
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    
    return out
    
    
def seed_3D_points(volume, spacing=[5,5,5]):
    
    n_z, n_y, n_x = volume.shape
    
    z_points = np.linspace(spacing[2]/2., n_z-spacing[2]/2.-1, int(n_z/float(spacing[2])))
    y_points = np.linspace(spacing[1]/2., n_y-spacing[1]/2.-1, int(n_y/float(spacing[1])))
    x_points = np.linspace(spacing[0]/2., n_x-spacing[0]/2.-1, int(n_x/float(spacing[0])))
    
    z_points = z_points.astype(np.int)
    y_points = y_points.astype(np.int)
    x_points = x_points.astype(np.int)
    
    # create fast cartesian product
    return cartesian([x_points, y_points, z_points], out=None)
    
    
    
def cassini_proj(u,v):
    
    x_c = np.arcsin(np.cos(u)*np.sin(v))
    y_c = np.arctan2(np.tan(u), np.cos(v))
    
    return x_c, y_c
    

def load_matlab_img(tiffile, frame):
    
    import scipy.io as spio
    
    if frame == 0: 
        im = (spio.loadmat(tiffile)['fixed_image']).transpose((2,0,1))
    else:
        im = (spio.loadmat(tiffile)['movingRegistered']).transpose((2,0,1))
        
        
    return im 
    
    
def cylindrical(r,theta,phi, theta0=0, phi0=np.pi/4.):
    """
    unzipping from any reference
    """
    y = (theta - theta0) *np.cos(phi0)
    x = (phi - phi0)
    
    return x, y
    
    
def cart2spherical(x,y,z, demean=True):
    
    if demean:
        xx = np.mean(x); yy = np.mean(y); zz = np.mean(z)
    else:
        xx = 0; yy= 0; zz = 0;
    
    dx = x - xx; dy = y-yy; dz = z - zz
    r = np.sqrt(dx**2+dy**2+dz**2)
    theta = np.arccos(dz/r)
    phi = np.arctan2(dy, dx)
    
    return r, theta, phi
    
    
def construct_proj_img(I, x_proj, y_proj, scale=10):
    
    scale_x = x_proj*scale ; scale_x =  scale_x - np.min(scale_x) ; scale_x = scale_x.astype(np.int)
    scale_y = y_proj*scale ; scale_y = scale_y - np.min(scale_y) ; scale_y = scale_y.astype(np.int)
    img_x = np.max(scale_x) + 1
    img_y = np.max(scale_y) + 1
    
    im = np.zeros((img_y, img_x))
    counts = np.zeros_like(im)
    
    for ii, intensity in enumerate(I):
        x = scale_x[ii]
        y = scale_y[ii]
        
        im[y,x] = np.max([im[y,x], intensity])
#        im[y,x] += intensity
#        counts[y,x] += 1
        
        # or use max_projection. 
        
#    im = im / (counts + 1e-8)
    
    return im 
    
"""
Conversion of velocities between cartesian and spherical basis
""" 
def cart2spherical_v(vxyz, angles):

    v_out = [] 
    
    for i in range(len(vxyz)):
        theta, phi = angles[i]
        T = np.array([[-np.sin(theta), np.cos(theta), 0],
                      [-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(phi), np.cos(phi) ],
                      [np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)]])
        
        v_ = T.dot(vxyz[i][:,None]) # to make it 3xN
        v_out.append(v_)
        
    v_out = np.array(v_out)
    
    return v_out
    
    
    
#==============================================================================
# Add a boiler code to compute tracks from the flow and return the tracks. 
#==============================================================================
def compute3D_superpixel_tracks(tif_files, optflowfiles, key_flow='motion_field', n_spixels=None, spixel_spacing=None, masked=False, dense=False, I_thresh=None):
    """
    Given a set of images and the corresponding flow between images, construct the 3d tracks, 
    tracks can either be masked or dense or both;
    """
    
    im0 = fio.read_multiimg_PIL(tif_files[0])
    n_z, n_y, n_x = im0.shape
    
    if n_spixels is not None:
        approx_spacing = int((n_x*n_y*n_z / float(n_spixels))**(1./3))
    else:
        approx_spacing = spixel_spacing
        
#==============================================================================
#   Seed the 3D points
#==============================================================================
    zeros = np.zeros((n_z, n_y, n_x))
    pos_xyz_initial = seed_3D_points(zeros, spacing=[approx_spacing,approx_spacing,approx_spacing])
    spixel_size = approx_spacing
    
#==============================================================================
#    Propagate the initial points (add in dense propagation.)
#==============================================================================
    if masked:
        meantracks3D = prop_spixels_frame_by_frame_forward3D_masked( pos_xyz_initial, optflowfiles, tif_files, spixel_size, thresh=I_thresh)
    else:
        meantracks3D = prop_spixels_frame_by_frame_forward3D(pos_xyz_initial, optflowfiles, spixel_size)
    
    return meantracks3D

    
    
    