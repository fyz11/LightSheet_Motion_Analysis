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
    
    
def viz_vector_field(position, velocity, vscale=3, scale_factor=7, volumeimg=None, thresh=10, sampling=None, vsampling=1, cmap=None, opacity=None):
    from mayavi import mlab 
    from skimage import measure 
    import numpy as np 
    
    if volumeimg is not None:
        verts, faces, _, vals = measure.marching_cubes_lewiner(volumeimg, thresh, spacing=(1, 1, 1), step_size=sampling)
        world_centre = np.mean(verts, axis=0) # compute the world centre coords. 
        mlab.triangular_mesh(verts[:,0]-world_centre[0], verts[:,1]-world_centre[1], verts[:,2]-world_centre[2], faces, scalars=vals, colormap=cmap, opacity=opacity)
    
    else:
        world_centre = np.mean(position, axis=0)
    
    mlab.quiver3d(position[::vsampling,2]-world_centre[0], position[::vsampling,1]-world_centre[1], position[::vsampling,0]-world_centre[2], 
                  vscale*velocity[::vsampling,2], vscale*velocity[::vsampling,1], vscale*velocity[::sampling,0], scale_factor=scale_factor, transparent=False)
    
    mlab.show()
    
    return []