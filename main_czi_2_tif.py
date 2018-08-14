#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:03:04 2018

@author: felix
"""

def remove_small_objects( volume, minsize=100):

    import skimage.morphology as morph
    st = []
    
    for v in volume:
        st.append(morph.remove_small_objects(v, min_size=minsize))
        
    return np.array(st)

    
def bounding_box(mask3D):
    
    import numpy as np 
    
    coords = np.where(mask3D>0)
    coords = np.array(coords).T
    
    min_x, max_x = np.min(coords[:,0]), np.max(coords[:,0])
    min_y, max_y = np.min(coords[:,1]), np.max(coords[:,1])
    min_z, max_z = np.min(coords[:,2]), np.max(coords[:,2])
    
    return np.hstack([min_x, min_y, min_z, max_x, max_y, max_z])
    

if __name__=="__main__":
    
    import numpy as np 
    import Utility_Functions.file_io as fio
    import Utility_Functions.stack as stack
    from skimage.exposure import rescale_intensity 
    
    import Unzipping.unzip as uzip
    import time
    
    infolder = '../../Data/Holly/czifile_test'
    outfolder = '../../Data/Holly/czifile_test_tiff'; fio.mkdir(outfolder)
    datasets = fio.load_dataset(infolder, ext='.czi', split_position=3, split_key='_')
    
    
    pad_num = 11
    
    for datafile in datasets[:]:
        
        t1 = time.time()
        im = fio.read_czifile(datafile); 
        im = im[:,::-1]
        
        """
        Rescale to uint8
        """
        im = np.uint8(255*rescale_intensity(im*1.)) # rescale the image intensity
        n_x, n_y, n_z = im.shape
        
        """
        Crop and pad. 
        """
        im_mask = uzip.segment_embryo(im, I_thresh=15, minsize=100, apply_morph=False)
        bbox_bounds = stack.bounding_box(im_mask)
        clip_limits = [[0, n_x], [0, n_y], [0,n_z]] 
        bbox_bounds_ = stack.expand_bounding_box(bbox_bounds, clip_limits, border=50, border_x=None, border_y=None, border_z=None)
    
        im = stack.crop_img_2_box(im, bbox_bounds)
        im = stack.pad_z_stack_adj(im, pad_slices=pad_num, min_I=15)
        
        fio.save_multipage_tiff(im, (datafile.replace(infolder, outfolder)).replace('.czi', '.tif'))
        
        print 'time elapsed: ', time.time() - t1
        

    
 







