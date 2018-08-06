# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:10:30 2016

@author: felix

File utils for Organoid or other grayscale images. 

"""
import numpy as np 

# helper function which can locate all the meantracks folders. 
def load_experiments(infolder, key_split):

    import os
    import numpy as np 

    all_directories = []
    all_full_paths = []

    for dirName, subdirList, fileList in os.walk(infolder):
        for subdir in subdirList:
#            print subdir
            if key_split in subdir:
                all_directories.append(dirName.split('/')[-1])
#                all_directories.append(subdir)
                all_full_paths.append(os.path.join(dirName, subdir))
            
    all_directories = np.array(all_directories).ravel()
    all_full_paths = np.array(all_full_paths).ravel()
    
    return all_directories, all_full_paths
    
def find_meantracks_folder(infolder, ext='.mat', include=None):
    
    import os 
    
    track_files = []    
    files = os.listdir(infolder)
    
    for f in files:    
        if ext in f:    
            if include is not None:
                if include in f:
                    track_files.append(os.path.join(infolder, f)) 
            else:
                track_files.append(os.path.join(infolder, f))
    
    return np.array(sorted(track_files)).ravel()
    
def find_meantracks_folder_exclude(infolder, ext='.mat', exclude=None):
    
    import os 
    
    track_files = []    
    files = os.listdir(infolder)
    
    for f in files:    
        if ext in f:    
            if exclude is not None:
                if exclude not in f:
                    track_files.append(os.path.join(infolder, f)) 
            else:
                track_files.append(os.path.join(infolder, f))
    
    return np.array(sorted(track_files)).ravel()


def find_meantracks_folder_include(infolder, ext='.mat', include=None):
    
    import os 
    
    track_files = []    
    files = os.listdir(infolder)
    
    for f in files:    
        if ext in f:    
            if include is not None:
                if include in f:
                    track_files.append(os.path.join(infolder, f)) 
            else:
                track_files.append(os.path.join(infolder, f))
    
    return np.array(sorted(track_files)).ravel()

