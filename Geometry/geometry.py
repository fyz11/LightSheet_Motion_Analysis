#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:53:00 2018

@author: felix
"""

import numpy as np 
import Geometry.transforms as tf


def get_rotation_x(theta):
    
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[1,0,0],
                   [0, np.cos(theta), -np.sin(theta)], 
                   [0, np.sin(theta), np.cos(theta)]])
    
    return R_z
    
def get_rotation_y(theta):
    
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0], 
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    return R_z
    
def get_rotation_z(theta):
    
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], 
                   [0, 0, 1]])
    
    return R_z


def shuffle_Tmatrix_axis_3D(Tmatrix, new_axis):
    
    """
    used to shuffle the Tmatrix axis to allow different image conventions. (useful e.g. if moving between Python and Matlab)
    """
    Tmatrix_new = Tmatrix[:3].copy()
    Tmatrix_new = Tmatrix_new[new_axis,:] # flip rows first. (to flip the translation.)
    Tmatrix_new[:,:3] = Tmatrix_new[:,:3][:,new_axis] #flip columns (ignoring translations)
    Tmatrix_new = np.vstack([Tmatrix_new, [0,0,0,1]]) # make it homogeneous 4x4 transformation. 
    
    return Tmatrix_new


def rotate_vol(vol, angle, centroid, axis, check_bounds=True):
    
    """
    note to self to add option to prepad before transform. 
    """
    rads = angle/180.*np.pi
    if axis == 'x':
        rot_matrix = get_rotation_x(rads)
    if axis == 'y':
        rot_matrix = get_rotation_y(rads)
    if axis == 'z':
        rot_matrix = get_rotation_z(rads)
    
    im_center = np.array(vol.shape)//2
    rot_matrix[:-1,-1] = np.array(centroid)
    decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
    
    T = rot_matrix.dot(decenter)
    print(T)

    if check_bounds:
        vol_out = tf.apply_affine_tform(vol, T,
                                        sampling_grid_shape=None,
                                        check_bounds=True,
                                        contain_all=True,
                                        codomain_grid_shape=None,
                                        domain_grid2world=None,
                                        codomain_grid2world=None,
                                        sampling_grid2world=decenter)
    else:
        vol_out = tf.apply_affine_tform(vol, T,
                                        sampling_grid_shape=vol.shape)
        
    return vol_out
    

def xyz_2_spherical(x,y,z, center=None):
    
    if center is None:
        center = np.hstack([np.mean(x), np.mean(y), np.mean(z)])
        
    x_ = x - center[0]
    y_ = y - center[1]
    z_ = z - center[2]
    
    r = np.sqrt(x_**2+y_**2+z_**2)
    polar = np.arccos(z_/r) # normal circular rotation (longitude)
    azimuthal = np.arctan2(y_,x_) # elevation (latitude)
    
    return r, polar, azimuthal
    
def spherical_2_xyz(r, polar, azimuthal, center=None):
    
    if center is None:
        center = np.hstack([0, 0, 0])
        
    x = r*np.sin(polar)*np.cos(azimuthal)
    y = r*np.sin(polar)*np.sin(azimuthal)
    z = r*np.sin(polar)
        
    x_ = x + center[0]
    y_ = y + center[1]
    z_ = z + center[2]
    
    return x_, y_, z_
    

def xyz_2_longlat(x,y,z, center=None):
    
    if center is None:
        center = np.hstack([np.mean(x), np.mean(y), np.mean(z)])
        
    x_ = x - center[0]
    y_ = y - center[1]
    z_ = z - center[2]
    
    r = np.sqrt(x_**2+y_**2+z_**2)
    latitude = np.arcsin(z_/r) # normal circular rotation (longitude)
    longitude = np.arctan2(y_,x_) # elevation (latitude)
    
    return r, latitude, longitude

def latlong_2_spherical(r,lat,long):
    
    polar = lat + np.pi/4.
    azimuthal = long
    
    return r, polar, azimuthal
    
def spherical_2_latlong(r, polar, azimuthal):
    
    lat = polar - np.pi/4.
    long = azimuthal
    
    return r, lat, long
    
    
def map_gnomonic(r, lon, lat, lon0=0, lat0=0):
    
    cos_c = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)
    x = r*np.cos(lat)*np.sin(lon-lon0) / cos_c
    y = r*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0))/ cos_c
    
    return x, y
    
def map_gnomonic_xy(x,y,z,r):
    
    # from similar triangles.  
    X = r/z * x
    Y = r/z * y
    
    return X, Y 
    
def azimuthal_ortho_proj3D(latlong, pole='pos'):
    
    r, lat, long = latlong
    
    x = r*np.cos(lat)*np.cos(long)
    y = r*np.cos(lat)*np.sin(long) 
    
    # determine clipping points
    if pole == 'pos':
        select = lat <= 0
    else:
        select = lat >= 0 
    return x[select], y[select], select 


def azimuthal_equidistant(latlong, lat1=0, lon0=0, center=None):
    
    r, lat, lon = latlong

    cos_c = np.sin(lat1)*np.sin(lat) + np.cos(lat1)*np.cos(lat)*np.cos(lon-lon0)
    c = np.arccos(cos_c)
    k_ = c/np.sin(c)
    
    x_p = k_ * np.cos(lat) * np.sin(lon-lon0)
    y_p = k_ * (np.cos(lat1)*np.sin(lat) - np.sin(lat1)*np.cos(lat)*np.cos(lon-lon0))

    return r*x_p, r*y_p 
    
    
    

