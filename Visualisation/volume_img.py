#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:25:24 2018

@author: felix
"""

def viz_volume(volumeimg, sampling=1, cmap='gray', opacity=0.5, thresh=10):
    
    from mayavi import mlab 
    from skimage import measure
    import numpy as np 
    
    verts, faces, _, vals = measure.marching_cubes_lewiner(volumeimg, thresh, spacing=(1, 1, 1), step_size=sampling)
    world_centre = np.mean(verts, axis=0) # compute the world centre coords. 
    
    mlab.triangular_mesh(verts[:,0]-world_centre[0], verts[:,1]-world_centre[1], verts[:,2]-world_centre[2], faces, scalars=vals, colormap=cmap, opacity=opacity)
    
    mlab.show()
    
    return []
    
    
def viz_vector_field(position, velocity, vscale=3, scale_factor=7, volumeimg=None, thresh=10, sampling=1, vsampling=1, cmap=None, opacity=None):
    from mayavi import mlab 
    from skimage import measure 
    from skimage.morphology import binary_dilation, ball
    import numpy as np 
    
    if volumeimg is not None:
        
        """
        Cut the isovolume and restrict display to the thresholded volume of the velocity field. 
        """

        mask = volumeimg > thresh; mask = binary_dilation(mask, ball(3))
        n_z, n_y, n_x = mask.shape
    

        pos_int = position.astype(np.int)
        select_coords = mask[pos_int[:,2], pos_int[:,1], pos_int[:,0]]
    
        # restrict the visualization.
        position = position[select_coords==1]
        velocity = velocity[select_coords==1]
        
        
        verts, faces, _, vals = measure.marching_cubes_lewiner(volumeimg, thresh, spacing=(1, 1, 1), step_size=sampling)
        world_centre = np.mean(verts, axis=0) # compute the world centre coords. 
        mlab.triangular_mesh(verts[:,0]-world_centre[0], verts[:,1]-world_centre[1], verts[:,2]-world_centre[2], faces, scalars=vals, colormap=cmap, opacity=opacity)
    
    else:
        world_centre = np.mean(position, axis=0)
    
    mlab.quiver3d(position[::vsampling,2]-world_centre[0], position[::vsampling,1]-world_centre[1], position[::vsampling,0]-world_centre[2], 
                  vscale*velocity[::vsampling,2], vscale*velocity[::vsampling,1], vscale*velocity[::sampling,0], scale_factor=scale_factor, transparent=False)
    
    mlab.show()
    
    return []


def depth_proj_surface_2D(im_array, rot_x=0, rot_y=0, rot_z=0, axis=0, proj_type='max', proj='exp', min_I=0, exp_tau=0.5, cut_off_fraction=0.5, enhance_gamma=1., background='white', unsharp_filt=False, unsharp_strength=0.1, unsharp_radius=3):
    """ weighted (linear or exponential) projection volume along the given dimension following (optional rotation) 
        
        **note** only uint8 bit inputs.
    """
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import square, disk, binary_dilation, binary_closing, remove_small_objects, binary_erosion
    from scipy.ndimage.morphology import binary_fill_holes
    import cv2
    # check these imports 
    from Geometry.geometry import rotate_vol
    from Geometry.transforms import rotate_volume_xyz
    from skimage.exposure import rescale_intensity, adjust_gamma
    import numpy as np 

    def largest_region(binary):
        from skimage.measure import label, regionprops
        if binary.max() > 0:
            
            labelled = label(binary)
            regprop = regionprops(labelled)
            
            regareas = np.hstack([re.area for re in regprop])
            largest_id = (np.unique(labelled)[1:])[np.argmax(regareas)]
            
            return labelled == largest_id
            
        else:
            return binary
    
    im_array_ = im_array.transpose(0,2,1) # put the long axis to the first axis. 
    vol_center = np.hstack(im_array_.shape)//2
    
    if np.isclose(rot_x,0) and np.isclose(rot_y,0) and np.isclose(rot_z,0):
        pass
    else:
        rot_matrix, im_array_ = rotate_volume_xyz(im_array_, 
                                                  vol_center, 
                                                  angle_x=rot_x, 
                                                  angle_y=rot_y, 
                                                  angle_z=rot_z,
                                                  use_im_center=True, out_shape=im_array_.shape)
    im_array_ = im_array_.transpose(0,2,1)
    
    n_z, n_y, n_x = im_array_.shape
    weights_array = np.zeros( im_array_.shape, dtype=np.float32)
    
    cutoff_z = int(n_z*cut_off_fraction)
    if proj == 'exp':
        weights = np.exp(-1./exp_tau*np.linspace(0.,1., cutoff_z))
        weights = np.hstack([weights, np.zeros(n_z - len(weights))])
        weights_array[:] = weights[:,None,None] # broadcast weights. 
    if proj == 'lin_weight':
        weights = np.linspace(1.,0., cutoff_z)
        weights = np.hstack([weights, np.zeros(n_z - len(weights))])
        weights_array[:] = weights[:,None,None]
    if proj == 'argmax':
        weights = np.ones(cutoff_z)
        weights = np.hstack([weights, np.zeros(n_z - len(weights))])
        weights_array[:] = weights[:,None,None]
        
    #    This gets the depth map and is useful for debugging mainly. 
    ZZ = np.argmax(im_array_> 0, axis=axis); #ZZ = gaussian(ZZ, sigma=11, preserve_range=True).astype(np.int)
    weighted_im = weights_array*im_array_
   
    if proj_type == 'max':
        # do a maximum projection 
        im_proj_weights = np.max(weighted_im, axis=0)
        print(im_proj_weights.max())
    
    if proj_type == 'min_dist':
        ZZ = np.argmax(im_array_ > 100, axis=0);
        
        XX, YY = np.meshgrid(range(n_x), range(n_y))
        im_proj_weights = im_array_[ZZ,YY,XX]
        
        import pylab as plt 
        
        plt.figure()
        plt.imshow(ZZ)
        plt.show()
        
        plt.figure()
        plt.imshow(im_proj_weights)
        plt.show()
        
    im_proj_weights = adjust_gamma(im_proj_weights, gamma=enhance_gamma) # brightness enhancement to aid background finding. 
    
    if background == 'white':
        # use user specified minimum intensity threshold. 
        mask = im_proj_weights > min_I
        mask = largest_region(mask) # take only the largest segmented region. 
        
        mask = remove_small_objects(mask, 100)
        mask = binary_closing(mask, disk(11)) 
        mask = np.logical_not(mask); mask = binary_dilation(mask, disk(5))
        mask = np.uint8(255*(mask)); 
        # add feathering to be more visually pleasing
        mask = np.uint8(np.clip(gaussian(mask, sigma=5, preserve_range=True), 0, 255))

        guide_filter = cv2.ximgproc.createGuidedFilter(np.uint8(im_proj_weights), radius=1, eps=.5) # was 5 
        mask_filt = guide_filter.filter(np.uint8(mask))
        original = im_proj_weights * 255/np.max(im_proj_weights) + mask_filt
        original = np.uint8(255*rescale_intensity(np.clip(original*1., 0, 255)))
        
        if unsharp_filt == True:
            blurred = np.clip(gaussian(im_proj_weights, unsharp_radius, preserve_range=True), 0, 255)
            im_proj_weights = original + unsharp_strength * (original - blurred)
        else:
            im_proj_weights = original
    
    return np.uint8(255.*rescale_intensity(np.clip(im_proj_weights, 0, 255)*1.)), ZZ


def apply_white_bg(im, min_I=1):
    
    from skimage.morphology import remove_small_objects, binary_closing, binary_dilation, disk
    from skimage.filters import gaussian 
    from skimage.exposure import rescale_intensity
    import cv2
    # create a mask and apply feathering using guided filtering.
    mask = im > min_I
    mask = remove_small_objects(mask, 100)
    mask = binary_closing(mask, disk(5))
    mask = np.logical_not(mask); mask = binary_dilation(mask, disk(1))
    mask = np.uint8(255*(mask)); 
    # add feathering to be more visually pleasing
    mask = np.uint8(np.clip(gaussian(mask, sigma=15, preserve_range=True), 0, 255))
    
#        mask = np.uint8(gaussian(mask, sigma=1, preserve_range=True) * 255)
    guide_filter = cv2.ximgproc.createGuidedFilter(np.uint8(im), radius=5, eps=.5)
    mask_filt = guide_filter.filter(np.uint8(mask))
   
    original = im * 255./np.max(im) + mask_filt
    original = np.uint8(255.*rescale_intensity(np.clip(original*1., 0, 255)))
#        original = np.uint8(np.clip(im_proj_weights * 255/np.max(im_proj_weights) + mask_filt, 0, 255))    
#        original = np.uint8(np.clip(im_proj_weights + mask_filt, 0, 255))
    blurred = np.clip(gaussian(im, 3, preserve_range=True), 0, 255)
    im = original + 0.1 * (original - blurred)
    
    return np.uint8(255.*rescale_intensity(np.clip(im, 0, 255)*1.))

def mark_checkerboard_lines_vol(img_2d_flat, color_1=(0,255,0), color_2=(255,255,0), spacing=64, thickness=3):
    
    img_2d_copy = np.dstack([img_2d_flat,
                             img_2d_flat,
                             img_2d_flat])
    
    for thick_i in range(int(np.rint(thickness))):
        img_2d_copy[thick_i::spacing,:,:] = np.hstack(color_1)[None,:] 
        img_2d_copy[:,thick_i::spacing,:] = np.hstack(color_2)[None,:] 
    
    return img_2d_copy


