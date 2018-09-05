#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:53:00 2018

@author: felix
"""

import numpy as np 

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
    
    
def azimuthal_ortho_proj3D(latlong, pole='pos'):
    
    r, lat, long = latlong
    
    x = r*np.cos(lat)*np.cos(long)
    y = r*np.cos(lat)*np.sin(long) 
    
    # determine clipping points
    if pole == 'pos':
        select = lat < 0
    else:
        select = lat > 0 
#    return x[select], y[select], select 
    return x[select], y[select], select 