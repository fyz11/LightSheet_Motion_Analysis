# -*- coding: utf-8 -*-
"""
Holly Hathrell
Main.py
Last edited 05/02/2018
"""
import os
import sys


    
def unique_rows(a):
    
    return np.vstack({tuple(row) for row in a})

def compT(t1, t2):
    return tuple(t1) == tuple(t2)
    
def fetch_coord_index(query, ref):
    
    index = []
    for ii, r in enumerate(ref):
        if compT(query, r):
            index.append(ii)
            
    return np.hstack(index)
    
    
def fetch_uniq_intensities(coords, I):
    
    from tqdm import tqdm
    uniq_coords = unique_rows(coords)
    
    uniq_Is = [[] for i in range(len(uniq_coords))]
    
    """
    iterate over the ref coords just once.
    """
    for ii, coord in tqdm(enumerate(coords)):
        select = fetch_coord_index(coord, uniq_coords)
        uniq_Is[select[0]].append(I[ii])
        
    return uniq_coords, [np.max(i) for i in uniq_Is]


"""
recode this to alleviate the problems of max order .
"""

def rescale_volume_im(im):
    
    from skimage.exposure import equalize_hist
    
    new_im = equalize_hist(im.ravel()).reshape(im.shape)
    
    return np.uint8(255*new_im)

    
from skimage.morphology import skeletonize


def gen_ref_map(ref_coord_set, nrows=None, ncols=None):
    """
    Builds a unique polar projection reference map based on a reference set.  
    """
    import numpy as np 
    uniq_x = np.unique(ref_coord_set[:,0])
    uniq_y = np.unique(ref_coord_set[:,1])
    
    if ncols is None:
        x_space = np.linspace(uniq_x[0], uniq_x[-1], len(uniq_x))
    else:
        x_space = np.linspace(uniq_x[0], uniq_x[-1], ncols)
    
    if nrows is None:
        y_space = np.linspace(uniq_y[0], uniq_y[-1], len(uniq_y))
    else:
        y_space = np.linspace(uniq_y[0], uniq_y[-1], nrows)
    
    ref_map_x, ref_map_y = np.meshgrid(x_space, y_space)
 
    ref_map = np.dstack([ref_map_x, ref_map_y])
    
    return ref_map


def shrink_emb_mask(mask, pad=10):
    
    from skimage.morphology import binary_erosion, square
    
    new_mask = []
    
    for m in mask:
        new_mask.append(binary_erosion(m, square(pad))[None,:])
    
    new_mask = np.concatenate(new_mask, axis=0)
    
    return new_mask




''' Set variables '''
#os.chdir('/home/holly/Codes/')
surface = True      # Do you want to find the embryo surface?
membranes = False    # Do you want to enhance the membranes?
gpu = False
voxel = 5.23         # if you are only running find_membranes, can set to 1.0.
depth = 20.0
border = False



#from Unzipping.preprocess import preprocess_image
#from Unzipping.File2_ed_felix import order_image
#from Unzipping.File3_ed_felix import map_max_projection
##from Unzipping.inputdata import load_and_rescale_image
import Utility_Functions.file_io as fio
import Unzipping.unzip_backup as uzip
import numpy as np
import pylab as plt 
from skimage.exposure import rescale_intensity, equalize_adapthist
import Geometry.transforms as tf
import Registration.registration_new as registration

# load in a set of files. 
dataset_folder = '/media/felix/Elements1/Shankar LightSheet/Example Timelapse/test'
out_aligned_folder = os.path.join(dataset_folder, 'aligned2')
dataset_files = fio.load_dataset(out_aligned_folder, ext='.tif', split_key='TP_',split_position=1) # load in the just aligned files.

out_folder = 'test_non-rigid-registered';
fio.mkdir(out_folder)

from scipy.misc import imsave
from tqdm import tqdm 
import Visualisation.volume_img as vol_img
import Geometry.meshtools as meshtools

# test the parametrization approach again? o
for i in tqdm(range(len(dataset_files))[1:-1]):
    
#    reg_config = {'alpha':0.1,
#                  'levels':[8,4,2,1],
#                  'warps':[4,2,0,0]}
    
    reg_config = {'level':4,
                  'warps':[100,100,10]}
    
    infile1 = dataset_files[1] # pick one reference don't do sequential !. 
    infile2 = dataset_files[i+1]
    
#    savefile='test_nonrigid.tif'
#    savetransformfile = 'test_transformfile.mat'
    
    basename = infile2.split('/')[-1]
    savefile= os.path.join(out_folder, 'nonrigid-demons-' + basename)
    savetransformfile = os.path.join(out_folder, 'transform-demons' + basename.replace('.tif','.mat'))
    
    print(i, savefile)
#    reg_config['alpha'], reg_config['levels'], reg_config['warps']
#    ret_val = registration.nonregister_3D(infile1, infile2, savefile, savetransformfile, reg_config)
    ret_val = registration.nonregister_3D_demons(infile1, infile2, savefile, savetransformfile, reg_config)
    
    
    
    
    
    
    
    
    
    