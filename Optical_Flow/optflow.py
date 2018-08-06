#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 23:46:52 2018

@author: felix
"""
import Utility_Functions.file_io as fio
import numpy as np 
import scipy.io as spio


def RTT_optflow3d_batch(dataset_files, in_folder, out_folder, reg_config, timer=True):
    
    import matlab.engine
    
    eng = matlab.engine.start_matlab() 
    
    # can only pass in python lists for internal conversion, does not support numpy!
    success_bit = eng.RTT_optflow(list(dataset_files), out_folder, 
                                   reg_config['method'], 
                                   reg_config['refid'],
                                   reg_config['refine_level'],
                                   reg_config['accFactor'],
                                   reg_config['downsample_factor'], 
                                   reg_config['alpha'],
                                   reg_config['I_thresh'],
                                   reg_config['lib_path'])
    
    return success_bit
    
