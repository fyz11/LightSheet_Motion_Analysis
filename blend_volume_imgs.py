#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:22:56 2018

@author: felix
"""

def blend_z_simple(stack, winsize=5):
    
    new_stack = np.pad(stack, [[winsize//2, winsize//2],[0,0],[0,0]], mode='reflect')
    
    return np.concatenate([np.mean(new_stack[i:i+winsize], axis=0)[None,:] for i in range(stack.shape[0])])


#def blend_2vols_simple(stack1, stack2, axis):
#    
#    new_stack = np.zeros_like(stack1)
#    
#    
#    return np.concatenate([np.mean(new_stack[i:i+winsize], axis=0)[None,:] for i in range(stack.shape[0])])
#


# function to join two volume stacks 
def simple_join(stack1, stack2, cut_off=None, blend=True, offset=10, weights=[0.7,0.3]):
    
    """
    stack1 and stack2 are assumed to be the same size. 
    """
    
    if blend:
        combined_stack = np.zeros_like(stack1)
        combined_stack[:cut_off-offset] = stack1[:cut_off-offset]
        combined_stack[cut_off-offset:cut_off] = weights[0]*stack1[cut_off-offset:cut_off]+weights[1]*stack2[cut_off-offset:cut_off]
        combined_stack[cut_off:cut_off+offset] = weights[1]*stack1[cut_off:cut_off+offset]+weights[0]*stack2[cut_off:cut_off+offset]
        combined_stack[cut_off+offset:] = stack2[cut_off+offset:]         
    else:
        combined_stack = np.zeros_like(stack1)
        combined_stack[:cut_off] = stack1[:cut_off] 
        combined_stack[cut_off:] = stack2[cut_off:] 
    
    return combined_stack


def sigmoid_join(stack1, stack2, cut_off=None, blend=True, gradient=200, shape=1, debug=False):
    
    """
    stack1 and stack2 are assumed to be the same size. 
    """
    def generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient):
        
        x = np.arange(0,stack1.shape[0])
        weights2 = 1./((1+np.exp(-grad*(x - cut_off)))**(1./shape))
        weights1 = 1. - weights2
        
        return weights1, weights2
    
    weights1, weights2 = generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient)
    if debug:
        plt.figure()
        plt.plot(weights1, label='1')
        plt.plot(weights2, label='2')
        plt.legend()
        plt.show()
        
    return stack2*weights1[:,None,None] + stack1*weights2[:,None,None]


if __name__=="__main__":
    
    """
    Script aims to implement a full pipeline based on the library. 
    """
    
    import Utility_Functions.file_io as fio
    import Utility_Functions.stack as stack_utils
    from Visualisation.imshowpair import imshowpair
    from skimage.exposure import rescale_intensity
#    import Registration.registration as registration
    import Registration.registration_new as registration
    import Optical_Flow.optflow as optflow
    import Tracking.supervoxel_tracks as svoxel_tracks
    import Visualisation.volume_img as vimgviz
    import os 
    import numpy as np 
    
    import time
    import pylab as plt 
    from tqdm import tqdm
    import scipy.io as spio
    import Utility_Functions.stack as stack
    from scipy.ndimage import zoom
#==============================================================================
#     load in the dataset 
#==============================================================================

    dataset_folder1 = '/media/felix/Elements1/Shankar LightSheet/Data/Holly_Test/Volume_Blending/L871_Emb2_a1_registered'
    dataset_folder2 = '/media/felix/Elements1/Shankar LightSheet/Data/Holly_Test/Volume_Blending/L871_Emb2_a2_registered'
    
    
####==============================================================================
####   Load data
####==============================================================================
#     load the files. 
    dataset_files1 = fio.load_dataset(dataset_folder1, ext='.tif', split_position=1)
    dataset_files2 = fio.load_dataset(dataset_folder2, ext='.tif', split_position=1)
    
    join_axis = 0
    for i in range(len(dataset_files1))[:1]:
        
        vol1 = fio.read_multiimg_PIL(dataset_files1[i])
        vol2 = fio.read_multiimg_PIL(dataset_files2[i])
        
        # normal blending
        com1, com2 = registration.COM_2d(vol1.mean(axis=1), vol2.mean(axis=1))
        
        join_point = int(.5*(com1[0] + com2[0]))
        
#        # simple linear blending. 
#        vol3 = registration.simple_join(vol2, vol1, cut_off=join_point, blend=True, offset=50, weights=[0.7,0.3])
#        print join_point
#        
#        exp_join(vol2, vol1, cut_off=join_point, blend=True, gradient=200)
        vol3 = registration.sigmoid_join(vol1, vol2, cut_off=join_point, blend=True, gradient=1./5, shape=1, debug=True) # only join across like 3 of them?


        plt.figure(figsize=(15,15))
        plt.subplot(131)
        plt.imshow(vol3.max(axis=0))
        plt.subplot(132)
        plt.imshow(vol3.max(axis=1))
        plt.subplot(133)
        plt.imshow(vol3.max(axis=2))
        plt.show()



#        plt.figure()
#        t = np.linspace(0,240)
#        plt.plot(t, np.exp(-t/200.))
#        plt.plot(t, 1-np.exp(-t/200.))
#        plt.show()
        
    
    
#    mean_imgs = []    
#    mean_imgs_yz = []
#    mean_imgs_xz = []
#    
#    for f in tqdm(dataset_files[:]):
#        vidstack = fio.read_multiimg_PIL(f)
#        max_img = vidstack.max(axis=0)[None,:]
#        mean_imgs.append(max_img)
#        mean_imgs_yz.append(vidstack.max(axis=1)[None,:])
#        mean_imgs_xz.append(vidstack.max(axis=2)[None,:])
##        vidstack = 255*rescale_intensity(zoom(vidstack, [.5,.5,.5])/255.)
#    
#    # progressive saving out.     
#    all_shapes = np.vstack([im.shape for im in mean_imgs])
#    max_shape = np.max(all_shapes, axis=0)
#        
##    mean_imgs = np.concatenate(mean_imgs, axis=0)
#    new_stack = np.zeros((len(mean_imgs), max_shape[1], max_shape[2]))
#    for ii, im in enumerate(mean_imgs):
#        new_stack[ii,:im.shape[1],:im.shape[2]] = im[0,:im.shape[1],:im.shape[2]].copy()
#
#    fio.save_multipage_tiff(new_stack, os.path.join(out_debug_folder, 'mean-angle-view1-xy.tif'))
#    
#    all_shapes = np.vstack([im.shape for im in mean_imgs_yz])
#    max_shape = np.max(all_shapes, axis=0)
#    new_stack = np.zeros((len(mean_imgs_yz), max_shape[1], max_shape[2]))
#    for ii, im in enumerate(mean_imgs_yz):
#        new_stack[ii,:im.shape[1],:im.shape[2]] = im[0,:im.shape[1],:im.shape[2]].copy()
#
#    fio.save_multipage_tiff(new_stack, os.path.join(out_debug_folder, 'mean-angle-view1-yz.tif'))
#    
#    all_shapes = np.vstack([im.shape for im in mean_imgs_xz])
#    max_shape = np.max(all_shapes, axis=0)
#    new_stack = np.zeros((len(mean_imgs_xz), max_shape[1], max_shape[2]))
#    for ii, im in enumerate(mean_imgs_xz):
#        new_stack[ii,:im.shape[1],:im.shape[2]] = im[0,:im.shape[1],:im.shape[2]].copy()
#
#    fio.save_multipage_tiff(new_stack, os.path.join(out_debug_folder, 'mean-angle-view1-xz.tif'))
 

    
    
    
    
    
    
        
        
        
        
        