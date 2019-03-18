#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:37:34 2019

@author: felix
"""

import Utility_Functions.file_io as fio

if __name__=="__main__":
    
    import sys 
    import nibabel as nib
    import numpy as np 

    # provide filepath.    
    filepath = sys.argv[1]
    # savepath = sys.argv[2]
    
    # read in the z-stack
    im = fio.read_multiimg_PIL(filepath)
    
    img_nibel = nib.Nifti1Image(im, np.eye(4))
    img_nibel.get_data_dtype() == np.dtype(np.int8) # it is an 8-bit image.

    savepath = filepath.replace('.tif', '.nii.gz')
    nib.save(img_nibel, savepath)

    print('converted .tif to nifti format for itk-snap')
        
