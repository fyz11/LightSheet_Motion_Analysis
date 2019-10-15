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



##### helper scripts to parse file naming conventions specific to Holly's analysis.
def parse_conditions_fnames(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
    """ get the tissue information """
    tissue = []
    if 'mtmg' in fname or 'epi' in fname:
        tissue = 'epi'
    if 'ttr' in fname or 've' in fname:
        tissue = 've'
        
    """ check if distal projection """
    proj = 'cylindrical'
    
    if 'distal' in fname:
        proj = 'distal'
        
    """ get the embryo number """
    emb_no = re.findall('L\d+', fname)

    if emb_no:
        emb_no = emb_no[0].split('L')[1]
    else:
        emb_no = np.nan
        
    """ get the angle of the unwrapping """
    ang_cand = re.findall(r"\d+", fname)

    ang = '000'
        
    for a in ang_cand:
        if len(a) == 3 and a!=emb_no:
            ang = a
            
    """ get the timepoint of the reference """      
    if '_tp' in fname:
        tp_val = fname.split('_tp')[1].split('.')[0]          
    else:
        tp_val = 'NA'
            
    """ check to see if angle is transposed """
    tp = 'No'
    tp_cand = re.findall(r"\d+tp", fname)      
    if len(tp_cand) > 0:
        # do we also need to check for lenght of the digits in front?
        tp = 'Yes'
    
    return tissue, proj, emb_no, ang, tp_val, tp
    
def fetch_assoc_unwrap_params(vidfile, folder, ext='.mat'):
    
    import glob
    import os 
    
    files = np.hstack(glob.glob(os.path.join(folder, '*'+ext)))

    v_condition = '-'.join(parse_conditions_fnames(vidfile))
    unwrap_conditions = np.hstack(['-'.join(parse_conditions_fnames(f)) for f in files])
    
    return files[unwrap_conditions == v_condition]


def pair_videofiles(videofiles, vid_conditions):
    
    comb_conditions = np.hstack(['-'.join(c) for c in vid_conditions])
    epi_select = vid_conditions[:,0]=='epi'
    ve_select = vid_conditions[:,0]=='ve'
    all_epi_files = videofiles[epi_select]; all_epi_cond = comb_conditions[epi_select]
    all_ve_files = videofiles[ve_select]; all_ve_cond = comb_conditions[ve_select]
    
    n_epi = len(all_epi_files)
    n_ve = len(all_ve_files)
    
    pair_files = []
    
    if n_epi >= n_ve:
        for ii in range(n_epi):
            f_epi = all_epi_files[ii]; f_epi_cond = all_epi_cond[ii]            
            f_ve = all_ve_files[all_ve_cond==f_epi_cond.replace('epi','ve')]; 
            f_ve_cond = all_ve_cond[all_ve_cond==f_epi_cond.replace('epi','ve')]

            if len(f_ve) > 0:
                f_ve = f_ve[0]; f_ve_cond = f_ve_cond[0]
            pair_files.append([f_ve, f_epi])
    else :
        for ii in range(n_ve):
            f_ve = all_ve_files[ii]; f_ve_cond = all_ve_cond[ii] 
            f_epi = all_epi_files[all_ve_cond==f_ve_cond.replace('ve','epi')]; 
            f_epi_cond = all_epi_cond[all_ve_cond==f_ve_cond.replace('ve','epi')]            

            if len(f_ve) > 0:
                f_ve = f_ve[0]; f_ve_cond = f_ve_cond[0]
            pair_files.append([f_ve, f_epi])
            
    return pair_files

