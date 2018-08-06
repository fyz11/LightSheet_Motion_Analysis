#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:53:48 2018

@author: felix
"""



def load_opt_flow_files(infolder, key='.mat'):
    
    import os 
    
    files = os.listdir(infolder)
    
    f = []

    for ff in files:
        if key in ff and 'flow' in ff:
            ind = ff.split(key)[0]
            ind = int(ind.split('_')[-1])
            f.append([os.path.join(infolder, ff), ind])
        
    f = sorted(f, key=lambda x: x[1])       
    
    return np.array(f)[:,0]


def read_optflow_file(optflowfile):
    
    import scipy.io as spio 
    
    obj = spio.loadmat(optflowfile)
    
    # result is 4D vector, which is x,y,z, 3D flow. 
    
    return obj['motion_field']


def compute_supervoxels(spixels):
    
    ny,nx,nz = spixels.shape
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
    
    print ny, nx, nz 
    print 'Y: ', np.min(Y), np.max(Y)
    print 'X: ', np.min(X), np.max(X)
    print 'Z: ', np.min(Z), np.max(Z)
    print Y.shape
    print spixels.shape
    
    positions_x = []
    positions_y = []
    positions_z = []
    vols = []
    regions = np.unique(spixels)
    print len(regions)
    
    for reg in regions:
        positions_x.append(np.mean(X[spixels==reg]))
        positions_y.append(np.mean(Y[spixels==reg]))
        positions_z.append(np.mean(Z[spixels==reg]))
        vols.append(np.sum(spixels==reg))
    
    # concatenate the positions inta large vector in the form (y,x)
    positions_x = np.array(positions_x).ravel()
    positions_y = np.array(positions_y).ravel()
    positions_z = np.array(positions_z).ravel()
        
    pos = np.vstack([positions_x, positions_y, positions_z]).T
    mean_width_superpixel = np.mean(vols)**(1./3)
            
            
    return pos, mean_width_superpixel
    
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
    
    imsave(savename, np_array.astype(np.uint8))
    
    return [] 


# uses vectorization to efficiently track regular superpixels over time, propagating their centroid positions according to the computed optical flow from frame to frame.
def prop_spixels_frame_by_frame_forward3D( spixel_pos, optflowfiles, spixel_size):

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
    optflow0 = read_optflow_file(optflowfiles[0])
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
        flow_frame = read_optflow_file(optflowfiles[i]) # (y,x,z,3)
#        print flow_frame.shape
#        flow_frame = flow_frame.transpose(1,0,2,3)
#        print flow_frame.shape
#        print np.max(expand_pos_0[:,:,0].ravel())
#        print np.max(expand_pos_0[:,:,1].ravel())
#        print np.max(expand_pos_0[:,:,2].ravel())

        # still index with y,x,z 
        # flow_frame is (y,x,z,3)
        flow_pos_0 = flow_frame[expand_pos_0[:,:,1].ravel(), expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,2].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmean(flow_pos_0, axis=1) # use average. 
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
#        print i, av_flow_pos.min(), av_flow_pos.max(), np.mean(av_flow_pos)
        
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos # add the x,y,z 
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], x_limits[0], x_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], y_limits[0], y_limits[1]-1)
        pos1[:,2] = np.clip(pos1[:,2], z_limits[0], z_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    

def prop_spixels_frame_by_frame_forward3D_masked( spixel_pos, optflowfiles, imgfiles, spixel_size, thresh=10):

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
    optflow0 = read_optflow_file(optflowfiles[0])
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

        
        """ read in the image """
        ref_vid = load_matlab_img(imgfiles[i], i); mask = ref_vid > thresh ; mask = binary_closing(mask, ball(5)); mask = np.logical_not(mask)
        full_mask = np.concatenate([mask[:,:,:,None], mask[:,:,:,None], mask[:,:,:,None]], axis=-1)
        full_mask = full_mask.transpose(1,2,0,3) # to have y,x,z
        
        """ read in the optflow field """
        flow_frame = read_optflow_file(optflowfiles[i]) # (y,x,z,3)
        flow_frame = flow_frame * full_mask # mask out. # could actually adopt a smoothened mask? 
#        print flow_frame.shape
#        flow_frame = flow_frame.transpose(1,0,2,3)
#        print flow_frame.shape
#        print np.max(expand_pos_0[:,:,0].ravel())
#        print np.max(expand_pos_0[:,:,1].ravel())
#        print np.max(expand_pos_0[:,:,2].ravel())

        # still index with y,x,z 
        # flow_frame is (y,x,z,3)
        flow_pos_0 = flow_frame[expand_pos_0[:,:,1].ravel(), expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,2].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmean(flow_pos_0, axis=1) # use average. 
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
#        print i, av_flow_pos.min(), av_flow_pos.max(), np.mean(av_flow_pos)
        
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos # add the x,y,z 
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], x_limits[0], x_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], y_limits[0], y_limits[1]-1)
        pos1[:,2] = np.clip(pos1[:,2], z_limits[0], z_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    
    
def plot_tracks3D(tracks, ax, color=None, lw=1., alpha=1):
    
    """
    
    Input:
    ------
    tracks: (n_superpixels x n_frames x 2) numpy array, giving the meantracks
    ax: matplotlib ax object
    color: color of the plotted lines given either named or as (r,g,b) value.   
    lw: linewidth, c.f. matplotlib
    alpha: transparency of the plotted lines, c.f. matplotlib
        
    Output:
    -------
    None, void function
    
    """
    
    n_spixels = tracks.shape[0]

    for i in range(n_spixels):
        ax.plot(tracks[i,:,1], tracks[i,:,0], tracks[i,:,2], c = color, lw=lw, alpha=alpha)
        
    return []


def find_non_const_tracks(meantracks):
    
#    n_spixels, n_frames, _ = meantracks.shape
    
    disps = meantracks[:,1:] - meantracks[:,:-1]
    disps = np.sum(disps**2, axis=-1)
    disps = np.sum(disps, axis=1)
    return disps > 0
    
    
def fetch_tracks_zslice(pos_xyz_initial, axis=2, slice_no=100):
    
    uniq_z = np.unique(pos_xyz_initial[:,axis])
    
    dist_z = np.abs(uniq_z - slice_no)
    select_z = uniq_z[np.argmin(dist_z)]
                      
    select = pos_xyz_initial[:,axis].astype(np.int) == int(select_z)
    
    return select

    
def load_tifs(infolder, key='.tif'):
    
    import os 
    
    files = os.listdir(infolder)
    
    ff = []
    
    for f in files:
        if key in f:
            
            frame_num = int(f.split('_TP')[1].split('.tif')[0])
            ff.append([os.path.join(infolder, f), frame_num])
            
    ff = sorted(ff, key=lambda x: x[1])
    ff = np.array(ff)
            
    return ff[:,0]


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
        tab = pd.DataFrame(data, index=np.arange(len(frame)), columns=['y','x','z', 'frame', 'particle'])
        tables.append(tab)
        
    tables = pd.concat(tables,ignore_index=True)
    tables.index = np.arange(tables.shape[0])
    
    return tables

    
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
    
    
def cart2spherical(mean_xyz, demean=False, means=None, ):
    
    if demean:
        m_x = means[1]
        m_y = means[0]
        m_z = means[2]
    else:
        m_x = 0
        m_y = 0
        m_z = 0
        
    r_x = mean_xyz[:,1] - m_x
    r_y = mean_xyz[:,0] - m_y
    r_z = mean_xyz[:,2] - m_z
    
    r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    theta  = np.arccos(r_z/r)
    phi = np.arctan2(r_y, r_x)
    
    return r, theta, phi

    
def cassini_proj(u,v):
    
    x_c = np.arcsin(np.cos(u)*np.sin(v))
    y_c = np.arctan2(np.tan(u), np.cos(v))
    
    return x_c, y_c
    
    
def mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []
    
    
def load_dataset_tif(infolder, ext='.tif',keysplit='.'):
    
    import os 
    
    f = os.listdir(infolder)
    
    files = []
    
    for ff in f:
        if ext in ff:
            print ff
            frame_No = int((ff.split(ext)[0]).split('_')[1])
            files.append([frame_No, os.path.join(infolder, ff)])
            
    files = sorted(files, key=lambda x: x[0])
    
    return np.hstack([ff[1] for ff in files])
    

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
    
    print img_x, img_y

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
    
    
    
if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    from mpl_toolkits.mplot3d import Axes3D
    from Visualisation_Tools.track_plotting import plot_tracks
    from Utility_Functions.file_io import read_multiimg_PIL
    import scipy.io as spio
    import os 
#    import trackpy as tp 
    
    # first load all the optical flow files
#    optflowfolder = '../RTTracker Scripts'
#    optflowfolder = '../RTTracker Scripts/custom_register_padded'
    optflowfolder = '../Data/2018-07-12_Matt_2color/repad/transformed_stack/matlab_refined_similarity_cleaned/optflow'
    optflowfiles = load_opt_flow_files(optflowfolder)
    
    savetrackfolder = '../Data/2018-07-12_Matt_2color/repad/transformed_stack/matlab_refined_similarity_cleaned/tracks-cleaned' ; mkdir(savetrackfolder)
    
    """
    Load images. 
    """
    tif_folders = '../Data/2018-07-12_Matt_2color/repad/transformed_stack/matlab_refined_similarity_cleaned'
    tif_files = load_dataset_tif(tif_folders, ext='.mat')
    
    I_thresh = 10
    
    im1 = load_matlab_img(tif_files[0], 0); mask = np.logical_not(im1 > I_thresh)
    
    nz, ny, nx = im1.shape
    
    """
    Call the masked track production. 
    """
    
    n_spixels = 20000
    approx_spacing = int((nx*ny*nz / float(n_spixels))**(1./3))
    """
    Step 1: Seed supervoxels through points.  
    """
    zeros = np.zeros((nz, ny, nx))
    
    pos_xyz_initial = seed_3D_points(zeros, spacing=[approx_spacing,approx_spacing,approx_spacing])
    spixel_size = approx_spacing
    
    print spixel_size
#     plot to check the sampling is correct. 

    """
    Do the propagation to link into tracks. 
    """
    meantracks3D = prop_spixels_frame_by_frame_forward3D_masked( pos_xyz_initial, optflowfiles, tif_files, spixel_size, thresh=10)
    
###     save the 3d tracks. 
    # still dominated by some alignment artifacts. hm.... (have to think about this a little) (include the postprocessing into the whole alignment?)
    savetrackfile = os.path.join(savetrackfolder, '3D_tracks_Registered_fast_mean_%d_flow_masked.mat' %(n_spixels))
###    3D_tracks_Felix_Registered_fast_seeded.mat
###    savetrackfile = '3D_tracks_Felix_Registered_fast_seeded.mat'
    spio.savemat(savetrackfile, {'meantracks':meantracks3D})

    meantracks3D = spio.loadmat(savetrackfile)['meantracks']
    mean_pos, mean_disps = compute_mean_displacement3D_vectors(meantracks3D[:,:])
    
    """
    Filter with the mask. 
    """
    select = mask[meantracks3D[:,0,2], meantracks3D[:,0,1], meantracks3D[:,0,0]]
    select = select == False
    
    mean_pos = mean_pos[select]
    mean_disps = mean_disps[select]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    
    sampling = 1        
    
    ax.quiver(mean_pos[::sampling,0], mean_pos[::sampling,1], mean_pos[::sampling,2], 
              mean_disps[::sampling,0], mean_disps[::sampling,1], mean_disps[::sampling,2], length=10, pivot='middle', normalize=True, arrow_length_ratio=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.auto_scale_xyz([0, 190], [0, 280], [0, 202])
    plt.show() 
   
    

    """
    Projection
    """
    x, y, z = mean_pos[:,0], mean_pos[:,1], mean_pos[:,2]
    r,theta,phi = cart2spherical(x,y,z, demean=False)
    proj_v = np.squeeze(cart2spherical_v(mean_disps, np.vstack([theta, phi]).T))
    
    angle = np.arctan2(proj_v[:,1], proj_v[:,0])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    
    sampling = 1        
    
    q = ax.quiver(mean_pos[::sampling,0], mean_pos[::sampling,1], mean_pos[::sampling,2], 
              mean_disps[::sampling,0], mean_disps[::sampling,1], mean_disps[::sampling,2], angle, length=10, pivot='middle', normalize=True, arrow_length_ratio=1, cmap='hsv')
    q.set_array(angle)
    
#    q = surf = ax.plot_surface(mean_pos[::sampling,0], mean_pos[::sampling,1], mean_pos[::sampling,2], cmap='hsv',
#                       linewidth=0, antialiased=False)
#    q.set_array(angle)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.auto_scale_xyz([0, 190], [0, 280], [0, 202])
    plt.show() 
    
    
    
    
    proj_v = np.squeeze(cart2spherical_v(mean_disps, np.vstack([theta, phi]).T))

#    x_proj, y_proj = miller_cylindrical(r,theta,phi)
    x_proj, y_proj = cylindrical(r,theta,phi, theta0=0, phi0=0)
#    I = im1[y,x,z]
    
    select = np.logical_not(np.logical_or(np.isnan(x_proj), np.isnan(y_proj)))
    x_proj = x_proj[select]
    y_proj = y_proj[select]
#    I = I[select]


    fig, ax = plt.subplots()
    ax.quiver(x_proj, y_proj, proj_v[:,0], proj_v[:,1], units='xy')
    plt.show()
    
    
    """
    Unzip the skin 
    """
    
    from skimage.morphology import binary_closing, binary_dilation, ball, binary_erosion
    binary_emb = im1 >= 10;
    binary_emb = binary_closing(binary_emb, ball(3))
    
    # now create a shell
    binary_outer = binary_dilation(binary_emb, ball(3))
    binary_inner = binary_erosion(binary_emb, ball(7))
    
    shell = np.logical_xor(binary_outer, binary_inner)
    
    # connected component filter to extract shell. 
    from skimage.measure import label
    shell_ = label(shell); wanted_id = np.argmax([np.sum(shell_==lab) for lab in np.unique(shell_)[1:]]) + 1
    shell = shell_ == wanted_id

    """
    From the shell create a series of x, y, z points. 
    """
    # this bit is wrong !
    X, Y, Z = np.meshgrid(range(shell.shape[2]), range(shell.shape[1]), range(shell.shape[0]))
    X = X.transpose(2,0,1); Y = Y.transpose(2,0,1); Z = Z.transpose(2,0,1)
    x = X[shell==True]
    y = Y[shell==True]
    z = Z[shell==True]
    

    r,theta,phi = cart2spherical(x,y,z, demean=False)
    x_proj, y_proj = cylindrical(r,theta,phi, theta0=0, phi0=0)
    I = im1[z,y,x]
    
    select = np.logical_not(np.logical_or(np.isnan(x_proj), np.isnan(y_proj)))
    x_proj = x_proj[select]
    y_proj = y_proj[select]
    I = I[select]
    
    flatten_img = construct_proj_img(I, x_proj, y_proj, scale=200)
#    from skimage.exposure import rescale_intensity
#    flatten_img_ = rescale_intensity(flatten_img, in_range=(0,100))

    plt.figure()
    plt.imshow(flatten_img)
    plt.show()
    
    
    
#    """
#    Attempt to unwrap. 
#    """
#    
#    # ok now we attempt: 
#    fig, ax = plt.subplots()
#    ax.quiver(mean_pos[::sampling,0], mean_pos[::sampling,1], mean_disps[::sampling,0], mean_disps[::sampling,1], units='xy')
#    plt.show()
#    
#    fig, ax = plt.subplots()
#    ax.quiver(mean_pos[::sampling,0], mean_pos[::sampling,2], mean_disps[::sampling,0], mean_disps[::sampling,2], units='xy')
#    plt.show()
#    
#    fig, ax = plt.subplots()
#    ax.quiver(mean_pos[::sampling,1], mean_pos[::sampling,2], mean_disps[::sampling,1], mean_disps[::sampling,2], units='xy')
#    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    