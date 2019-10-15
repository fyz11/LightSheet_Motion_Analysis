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

def resample_surf_points_lines( point_data, bins, ref_point=None, t_axis=None, max_t_val=None, range=None, axis=0, sigma1d=5, smooth1dmode='wrap', remove_dup=False, resample=True, periodic=False, n_samples=1000, smoothing=1000, poly_order=3, return_intermediates=True, reparametrise_fn=None, *args, **kwargs):

    """
    Uses spline interpolation to interpolate along ordered lines, enables smooth reparametrisation of surface along each of the axis
    
    point_data : should be (N,5) array, (x,y,z, i,j) where i,j is the effective (u,v) mapping
    axis : which dimension we are going to discretise into lines and smooth 
    
    """
    import numpy as np 
    from scipy import ndimage, interpolate
    import pylab as plt 
    from tqdm import tqdm

    # create discretised binning 
    if range is not None:
        all_uniq_values = np.linspace(range[0],
                                      range[1], 
                                      bins + 1)
    else:
        all_uniq_values = np.linspace(point_data[:,axis].min(),
                                      point_data[:,axis].max(), 
                                      bins + 1)

    # create a new tabular data
    # compile a table of uniq angles and sorted distances to reference point
    point_data_binned = []
    binned_data_debug = []

    for jj in tqdm(np.arange(len(all_uniq_values))[:-1]):
        
        # select out data from this bin. 
        select = np.logical_and(point_data[:,axis] >= all_uniq_values[jj], 
                                point_data[:,axis] < all_uniq_values[jj+1])

        # slice out the points  
        line = point_data[select]
        binned_data_debug.append(line)

        if axis == -1: 
            sort_order = np.argsort(line[:,-2])
        if axis == -2: 
            sort_order = np.argsort(line[:,-1])
        line = line[sort_order] # sort by the other axis.

        # check if there is enough points to do anything with 
        if len(line) > poly_order:

            # print(all_uniq_values[jj])
            # smoothen the coordinates first
            xp = line[:,0]; yp = line[:,1]; zp = line[:,2]; 

            if ref_point is not None:
                xp = np.hstack([ref_point[0], xp])
                yp = np.hstack([ref_point[1], yp])
                zp = np.hstack([ref_point[2], zp])
            
            if t_axis is not None:
                dp = line[:,t_axis]
            else:
                if axis == -1:
                    dp = line[:,-1]
                if axis == -2:
                    dp = line[:,-2]

                # give a monotonicity check on dp!. must be increasing....  
                
            # print(line[:,-2])
            if remove_dup:
                # print('removing duplications')
                xp = xp[:-1]
                yp = yp[:-1]
                zp = zp[:-1]

                jump = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2+np.diff(zp)**2) 
                smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                xp, yp, zp = xp[:-1], yp[:-1], zp[:-1]
                dp = dp[:-1]

                xp = xp[(jump > 0) & (smooth_jump < limit)]
                yp = yp[(jump > 0) & (smooth_jump < limit)]
                zp = zp[(jump > 0) & (smooth_jump < limit)]
                dp = dp[(jump > 0) & (smooth_jump < limit)] 

            else:
                xp = ndimage.gaussian_filter1d(xp, sigma1d, mode=smooth1dmode)
                yp = ndimage.gaussian_filter1d(yp, sigma1d, mode=smooth1dmode)
                zp = ndimage.gaussian_filter1d(zp, sigma1d, mode=smooth1dmode)
                # dp = ndimage.gaussian_filter1d(dp, sigma1d, mode=smooth1dmode)

            # if we have ref point then add this on.
            if ref_point is not None:
                xp = np.hstack([ref_point[0], xp])
                yp = np.hstack([ref_point[1], yp])
                zp = np.hstack([ref_point[2], zp])
                dp = np.hstack([0, dp])

            # else:
                # print('no ref point')

            # print(xp.shape, yp.shape, zp.shape, dp.shape)
            if t_axis is None:
                if periodic == True:
                    tck, u = interpolate.splprep([xp,
                                                  yp,
                                                  zp], per=1, s=smoothing)
                else:
                    tck, u = interpolate.splprep([xp,
                                                yp,
                                                zp], s=smoothing)
            else:
                # dp = np.linspace(dp[0], dp[-1], len(dp))
                if periodic == True:
                    tck, u = interpolate.splprep([xp,
                                                  yp,
                                                  zp], s=smoothing, u=dp / max_t_val, per=1)
                else:
                    tck, u = interpolate.splprep([xp,
                                                  yp,
                                                  zp], s=smoothing, u=dp / max_t_val)

            if t_axis is None:
                if resample == False:
                    u_fine = np.linspace(0, 1, len(xp)) # no interpolation 
                else:
                    u_fine = np.linspace(0, 1, n_samples) # no interpolation 
            else:

                if resample == False:
                    u_fine = np.linspace(0, (dp / max_t_val).max(), len(xp)) 
                else:
                    u_fine = np.linspace(0, (dp / max_t_val).max(), n_samples)

            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

            # refine the information along the axis based on the interpolation. 
            val_fine = .5*(all_uniq_values[jj] + all_uniq_values[jj+1]) * np.ones(len(x_fine))
            
            new_point_data = np.zeros((len(x_fine), 5))
            new_point_data[:,0] = x_fine
            new_point_data[:,1] = y_fine
            new_point_data[:,2] = z_fine
            new_point_data[:,axis] = val_fine

            if reparametrise_fn is not None:
                if axis == -1: 
                    new_point_data[:,-2] = reparametrise_fn(new_point_data[:,:3], *args, **kwargs)
                if axis == -2:  
                    new_point_data[:,-1] = reparametrise_fn(new_point_data[:,:3], *args, **kwargs)
            else:
                if axis == -1: 
                    new_point_data[:,-2] = np.linspace(np.min(line[:,-2]), np.max(line[:,-2]), len(x_fine))
                if axis == -2:  
                    new_point_data[:,-1] = np.linspace(np.min(line[:,-1]), np.max(line[:,-1]), len(x_fine))

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(point_data[::200,0], 
            #            point_data[::200,1], 
            #            point_data[::200,2], 'o')
            # ax.scatter(xp, yp, zp,'go')
            # ax.plot(x_fine, y_fine, z_fine, '-ro')
            # plt.show()

            point_data_binned.append( new_point_data )
        
    point_data_resampled = np.vstack(point_data_binned)

    if return_intermediates:
        return point_data_resampled, binned_data_debug
    else:
        return point_data_resampled


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
    
    # remove outliers.PCL 
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(fil_mean_k)  # what to do here? 
    fil.set_std_dev_mul_thresh (fil_std)
    p = fil.filter()
    
    return np.asarray(p)


def smooth_points_neighbours(points, n_neighbours=15, radius=30, neighbor_type='knn', avg_func=np.mean):
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='auto').fit(points)
    
    if neighbor_type == 'knn': # this one seems to work better? 
        _, indices = nbrs.kneighbors(points)
    if neighbor_type == 'radius': 
        _, indices = nbrs.radius_neighbors(points)
    
    points_ = []
    
    for i, _ in enumerate(indices):
        points_.append(avg_func(points[indices[i]], axis=0))
        
    return np.vstack(points_)


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

