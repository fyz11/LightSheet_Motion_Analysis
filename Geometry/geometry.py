#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:53:00 2018

@author: felix
"""

import numpy as np 


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
    
    
    

