#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:45:41 2018

@author: felix

Provides mesh cleaning and construction tools. 
"""

import numpy as np 
# helper functions

def unique_rows(a):
    
    return np.vstack({tuple(row) for row in a})


def sort_rotation(pts):
    
    import numpy as np 
    
    center = pts.mean(axis=0)
    pts_ = pts - center[None,:]
    angle = np.arctan2(pts_[:,1], pts_[:,0]) 

    inds = np.argsort(angle)
    return pts[inds]
    

def fit_closed_curve(pts, n_pts=100, kind='cubic'):
    
    from scipy.interpolate import interp1d 
    
    """
    join the end points.
    """
    
    pts = np.r_[pts, pts[0][None,:]]
    i = np.arange(len(pts))
    interp_i = np.linspace(0, i[-1], n_pts)
    
    xi = interp1d(i, pts[:,0], kind=kind)(interp_i)
    yi = interp1d(i, pts[:,1], kind=kind)(interp_i)
    
    return np.vstack([xi,yi]).T


"""
Core functions
"""
# concave hull function 
def concave(points,alpha_x=150,alpha_y=250):
    
    from scipy.spatial import Delaunay, ConvexHull
    import networkx as nx
    
    points = [(i[0],i[1]) if type(i) != tuple else i for i in points]
    de = Delaunay(points)
    dec = []
    a = alpha_x
    b = alpha_y
    for i in de.simplices:
        tmp = []
        j = [points[c] for c in i]
        if abs(j[0][1] - j[1][1])>a or abs(j[1][1]-j[2][1])>a or abs(j[0][1]-j[2][1])>a or abs(j[0][0]-j[1][0])>b or abs(j[1][0]-j[2][0])>b or abs(j[0][0]-j[2][0])>b:
            continue
        for c in i:
            tmp.append(points[c])
        dec.append(tmp)
    G = nx.Graph()
    for i in dec:
            G.add_edge(i[0], i[1])
            G.add_edge(i[0], i[2])
            G.add_edge(i[1], i[2])
    ret = []
    for graph in nx.connected_component_subgraphs(G):
        ch = ConvexHull(graph.nodes())
        tmp = []
        for i in ch.simplices:
            tmp.append(list(graph.nodes())[i[0]])
            tmp.append(list(graph.nodes())[i[1]])
        ret.append(tmp)
    return ret  



def findPointNormals(points, nNeighbours, viewPoint=[0,0,0], dirLargest=True):
    
    """
    construct kNN and estimate normals from the local PCA
    
    reference: https://uk.mathworks.com/matlabcentral/fileexchange/48111-find-3d-normals-and-curvature
    """
    
    # construct kNN object to look for nearest neighbours. 
    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=nNeighbours+1)
    neigh.fit(points)
    
    nn_inds = neigh.kneighbors(points, return_distance=False) # also get the distance, the distance is used for cov computation.
    nn_inds = nn_inds[:,1:] # remove self
    
    # find difference in position from neighbouring points (#technically this should be relative to the centroid of the patch!)
    # refine this computation. to take into account the central points. 
#    p = points[:,None,:] - points[nn_inds]
    p = points[nn_inds] - (points[nn_inds].mean(axis=1))[:,None,:]
    
    # compute covariance
    C = np.zeros((len(points), 6))
    C[:,0] = np.sum(p[:,:,0]*p[:,:,0], axis=1)
    C[:,1] = np.sum(p[:,:,0]*p[:,:,1], axis=1)
    C[:,2] = np.sum(p[:,:,0]*p[:,:,2], axis=1)
    C[:,3] = np.sum(p[:,:,1]*p[:,:,1], axis=1)
    C[:,4] = np.sum(p[:,:,1]*p[:,:,2], axis=1)
    C[:,5] = np.sum(p[:,:,2]*p[:,:,2], axis=1)
    C = C / float(nNeighbours)
    
    # normals and curvature calculation 
    normals = np.zeros(points.shape)
    curvature = np.zeros((len(points)))
    
    for i in range(len(points))[:]:
        
        # form covariance matrix
        Cmat = np.array([[C[i,0],C[i,1],C[i,2]],
                [C[i,1],C[i,3],C[i,4]],
                [C[i,2],C[i,4],C[i,5]]])
        
        # get eigen values and vectors
        [d,v] = np.linalg.eigh(Cmat);
        
#        d = np.diag(d);
        k = np.argmin(d)
        lam = d[k]
        
        # store normals
        normals[i,:] = v[:,k]
        
        #store curvature
        curvature[i] = lam / np.sum(d);

    # flipping normals to point towards viewpoints
    #ensure normals point towards viewPoint
    points = points - np.array(viewPoint).ravel()[None,:]; # this is outward facing

    if dirLargest:
        idx = np.argmax(np.abs(normals), axis=1)
        dir = normals[np.arange(len(idx)),idx]*points[np.arange(len(idx)),idx] < 0;
    else:
        dir = np.sum(normals*points,axis=1) < 0;
    
    normals[dir,:] = -normals[dir,:];
    
    return normals, curvature



def clean_point_clouds3D(points, MLS_search_r=3, MLS_pol_order=2, MLS_pol_fit=True, fil_mean_k=5, fil_std=3):

    """
    how to reduce the external dependence? 
    """
    import pcl 
    
    # Moving least squares local fitting.
    p = pcl.PointCloud(points.astype(np.float32))
    MLS = p.make_moving_least_squares()
    MLS.set_search_radius(MLS_search_r) # this should be nearest neighbour? 
    MLS.set_polynomial_order(MLS_pol_order)
    MLS.set_polynomial_fit(MLS_pol_fit)
    p = MLS.process()
    
    # remove outliers. 
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(fil_mean_k)  # what to do here? 
    fil.set_std_dev_mul_thresh (fil_std)
    p = fil.filter()
    
    return np.asarray(p)




def create_clean_mesh_points(im_array, clean_pts, n_pts=100, kind='linear', min_pts=10, eps=1, alpha=[1000,1000]):
    
    import numpy as np 

    mesh_points = []

    for x in range(im_array.shape[0]):

        select = np.logical_and(clean_pts[:,0]>x-eps,
                                clean_pts[:,0]<x+eps)
        
        if np.sum(select) > min_pts:
            # if there is sufficient
            points = concave(clean_pts[select,1:].astype(np.int), alpha_x=alpha[0], alpha_y=alpha[1])
            n_hulls = len(points) # how many did it find? 
            
            if n_hulls == 1:
                
                hull = points[0]
                
                # sort in rotation order. 
                pts = sort_rotation(unique_rows(hull))
                xyi = fit_closed_curve(pts, n_pts=n_pts, kind=kind)
                
                xi, yi = xyi[:,0], xyi[:,1]
                zi = np.ones(n_pts) * x
                xyz = np.vstack([zi, xi, yi]).T
                
                mesh_points.append(xyz)
            """
            resample the points in the concave hull to make it dense! 
            """
    mesh_points = np.vstack(mesh_points)
    
    return mesh_points

