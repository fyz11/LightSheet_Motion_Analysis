#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:01:36 2018

@author: felix

Reusable Functions to do unzipping of embryo.
"""

def keep_largest_component(mask3D):
    
    """
    remove small objects and keep the largest connected component
    """
    from skimage.measure import label
    import numpy as np 
    labelled = label(mask3D)
    uniq_regs = np.unique(labelled)[1:]
    areas = [np.sum(labelled == reg) for reg in uniq_regs]
             
    keep = labelled == uniq_regs[np.argmax(areas)]
             
    return keep
    
    
def remove_small_objects( mask3D, minsize=100):

    import skimage.morphology as morph
    import numpy as np 
    
    st = []
    
    for v in mask3D:
        st.append(morph.remove_small_objects(v, min_size=minsize))
        
    return np.array(st)
        

def segment_embryo(im_array, I_thresh=10, ksize=3, minsize=100, apply_morph=True):
    
    """
    im_array : grayscale 3d volume
    """
    from skimage.morphology import ball, binary_closing, binary_dilation
    
    n_z, n_y, n_x = im_array.shape
    
    mask = im_array >= I_thresh
    mask = remove_small_objects(mask, minsize)
    
    if apply_morph:
        mask = binary_closing(mask, ball(ksize))
        mask = binary_dilation(mask, ball(ksize))

    return mask
    

def contour_seg_mask(im, mask):
    
    import cv2
    import numpy as np
    im2,contours,heirachy = cv2.findContours(np.uint8(255*mask),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)    # Note: plot the contours not the image output!
    
    # index into image. 
    contours_im = np.zeros_like(im)
    
    if len(contours) > 0:
        
        # find the longest.
        len_contours = [len(cnt) for cnt in contours]
        contours = np.squeeze(np.array(contours)[np.argmax(len_contours)])

        # index into image. 
        contours_im = np.zeros_like(im)
#        print contours_im.shape
        if len(contours.shape) == 2:
            contours_im[contours[:,1], contours[:,0]] = 1
    
    return im*contours_im

    
def contour_seg_embryo(im_array, mask_array):
    
    """
    exploits cv2 contour function to create a mask of the contour. 
    """
    import numpy as np 
    
    return np.array([contour_seg_mask(im_array[i], mask_array[i]) for i in range(len(im_array))])

    
def segment_contour_embryo(im_array, I_thresh=10, ksize=3):
    
    mask_array = segment_embryo(im_array, I_thresh=I_thresh, ksize=ksize)
    contour_array = contour_seg_embryo(im_array, mask_array)  
    
    return mask_array, contour_array
    
    
def unwrap_embryo_surface(im_array, embryo_mask, contour_mask, depth=20, voxel=5.23, return_no_interp=False):   
   
    """
    Given a segmentation and shell segmentation we can unwrap the embryo surface into 2D map.
    
    Inputs:
        im_array: volume_img
        embryo_mask: binary mask of the volume
        contour_mask: binary_mask of the outer contour with the intensity vals. 

    Outputs:
        unwrap_params: geometrical properties of the unwrapping to apply to more embryos. 
        emb_map: unwrapped 'raw' map
        emb_map_normalised: map is stretched to fill the full sides. (this unwrapping is not normal? --> double check)
    """   
    # 1. find the cylindrical coordinates
    xzyp_indices, center_of_mass = find_cylindrical_coords(contour_mask, embryo_mask)
    max_order, xzypI_order = sort_coord_order(xzyp_indices) # sort the coordinates? is this necessary? 
    emb_map, max_pixel = map_max_projection(im_array, center_of_mass, xzypI_order, max_order, depth=depth, voxel=voxel) # max order is the maximum width.
    
    map_coords, radii = normalise_projection(emb_map, xzypI_order, max_order)
    
    if return_no_interp:
        (emb_map_normalised, emb_map_wrap), (emb_no_interp, emb_wrap_no_interp) = interpolate_projection(map_coords, radii, return_no_interp=return_no_interp)
    else:
        emb_map_normalised, emb_map_wrap = interpolate_projection(map_coords, radii, return_no_interp=return_no_interp)
    
    # save as a dictionary. 
    # to convert to cylindrical needs the required COM, also needs the max_radii in order to set the map size. 
    unwrap_params = {'max_radii':max_order,
                     'radii':radii, 
                     'center':center_of_mass, 
                     'map_coords':map_coords}
    
    if return_no_interp:
        return unwrap_params, (emb_map, emb_map_wrap, emb_map_normalised, max_pixel, emb_no_interp, emb_wrap_no_interp)  
    else:
        return unwrap_params, (emb_map, emb_map_wrap, emb_map_normalised, max_pixel)  
    
    
def find_cylindrical_coords(contour_mask, embryo_mask):
    
    ''' 
    Find (x,z,y,phi) for the outer shell coordinates (basically the cylindrical coords.)
    Inputs: 
            self.mask shell
            self.threshold
    Outputs: 
            self.center
            self.xzyp_indices
            self.non_zero_entrys
            self.phi
    Returns: 
            none
    '''
    import numpy as np 
    from scipy.ndimage.measurements import center_of_mass
    
    non_zero_indices_ = np.where(contour_mask > 0)   # (x,z,y) All the indices of entries in the binary outer shell that are non-zero
    non_zero_indices = np.array(non_zero_indices_).T
#    z_x_y_indices = non_zero_indices[:,[1,0,2]]
    z_x_y_indices = np.transpose(non_zero_indices_)
#    z_x_y_indices = non_zero_indices.transpose(1,0,2)  # This sorts the columns into tuples of (z,x,y)

    # center of mass
    center = np.array(center_of_mass(embryo_mask))      # Find the COM of the whole embryo using the filled in threshold array
    real_zxy = z_x_y_indices.astype(float)                          # Change the integer indices to float values

    # Find the angles of the outer shell pixels - this uses the distance from the COM 
    # This assumes the embryo is vertical along the X axis - done using transforms in Fiji MVRA
    # Shift the y and z axes into the COM coords, from (x,z,y) to (x,zc,yc)
    COM_xzy = np.copy(real_zxy)                         # Copy stops the arrays becomming linked
    
    # change the centre of references. 
    COM_xzy[:,0] = np.copy(real_zxy[:,0])               # Want to keep x starting at zero
    COM_xzy[:,1] = real_zxy[:,1]-center[1]              # Shift the z coordinates to have the COM in the middle
    COM_xzy[:,2] = real_zxy[:,2]-center[2]              # Shift the y coordinates to have the COM in the middle
    
    phi = np.arctan2((COM_xzy[:,2]), (COM_xzy[:,1]))    # Find the asymuthal angle in cylindrical coords (note: arctan does not find the correct quadrants)
    
    I =  contour_mask[non_zero_indices[:,0], non_zero_indices[:,1], non_zero_indices[:,2]]
    xzypI = np.hstack([non_zero_indices, phi[:,None], I[:,None]])
    
    return xzypI, center # return the centre of mass of embryo
    
    
def sort_coord_order(xzypI): 
        ''' 
            Sort the coordinates into order, first by x (radii), then by angle, phi second
            Inputs: 
                    xzypI
            Outputs: 
                    self.full_array_orderxpo
                    self.max_order
            Returns: 
                    self.full_array_orderxpo, self.max_order
        '''
        import numpy as np 
        
        ind = np.lexsort((xzypI[:,3],xzypI[:,0]))   # Sorting with indices.
        xzypI_order = xzypI[ind]                          

        uniq_x = np.unique(xzypI_order[:,0])
        order_matrix = np.zeros(len(xzypI_order), dtype=np.int)
        
        for x in uniq_x:
            select = xzypI_order[:,0]== x 
            order_matrix[select] = np.arange(1, np.sum(select)+1, dtype=np.int)

        xzypI_order = np.append(xzypI_order, order_matrix[:,None], axis=1)    # Append the order column to the data
        
        # this gives the maximum radii over the whole image to unroll. 
        max_order = np.max(xzypI_order[:,5])                                # This gives the maximum width to be unrolled
        
        return max_order, xzypI_order
    
     
def map_max_projection(im_array, center_of_mass, xzypI_order, max_order, depth=20.0, voxel=5.23):
    
    ''' 
    
    Given some sorted coordinates this function max projects them and unwraps them onto a 2D map. 
    
        Parameters are ...
            Input path: string, path to files
            Output path: string, pathe to location in which to save files
            gpu: whether the code is run on GPU or not
        Returns:    images of a the unwrapped embryo surface: 'emb_map' 
                    normalised surface: 'emb_map_normalised'
                    normalised surface with front data centered: 'emb_map_normalised_wrap'       
            
    '''

    import numpy as np 
    n,p,q = im_array.shape    
    max_order = int(max_order) # proxies
            
    # Loop over each x (in order of phi) 
    # Stack the pixels and assign the max of them to the original pixel coordinates in a flat array (x,order)
    
    # initialise the arrays. 
    emb_map = np.zeros(shape=(n, max_order),dtype=(np.uint8))    # Initialise the embryo map of length x and max width
    pt2 = np.take(center_of_mass, [1,2]).astype(np.int)   # This gives the y,z, central points of that slice (assumes a completely vertical embryo)
        
    for xpoi in xzypI_order:

        fx = int(xpoi[0]) #x
        fz = int(xpoi[1]) #z 
        fy = int(xpoi[2]) #y
        fo = int(xpoi[5]) # order/radii? ( wait x here is the z! )
    
        pt1 = np.asarray([fx,fz,fy]).astype(np.int)                    # The coordinates of one pixel on the embryo surface

        """
        Create a line and project ( forsake the production of intermediate image and expensive boolean operations. )
        """
        
        dist_line = np.linalg.norm(pt1[1:3]-pt2)
        mask_coords = np.array([np.linspace(pt1[1:3][0], pt2[0], 2*int(dist_line)+1),
                                np.linspace(pt1[1:3][1], pt2[1], 2*int(dist_line)+1)]).T
        mask_line = im_array[fx,:,:][mask_coords[:,0].astype(np.int), mask_coords[:,1].astype(np.int)]

        # double check this. should only need the depth. 
        distance_line = np.sqrt(np.sum (np.square (((mask_coords.astype(np.float64))-(pt1[1:3].astype(np.float64)))*[voxel,1]), axis=1))
        mask_distance_less = distance_line<=(depth*voxel)   # Keeps only the values which are less than a given distance
        
        if len(mask_line[mask_distance_less]) > 0: 
            max_pixel = np.mean(mask_line[mask_distance_less]) # mean projection
        else:
            max_pixel = 0
#        max_pixel = np.max(mask_line[mask_distance_less]) # mean projection
        emb_map[fx,fo-1] = max_pixel    #-1 as array starts from 0
                
    return emb_map, max_pixel

    
def normalise_projection(emb_map, xzypI_order, max_order):
    
    ''' Normalise the embryo map to the largest width 
            Inputs: 
                    self.full_array_order_xpo
                    self.emb_map
            Ouputs: 
                    none
            Returns: 
                    none
    '''
    import numpy as np 

    # find the unique z values.     
    fx_values = np.unique(xzypI_order[:,0]).astype(np.uint)   # Finds all the unique values of fx i.e. each slice basically. hm.... 
    
    radii_x = [] # record the maximum radii.
    mapped_coords = []
    
    # iterate line by line. 
    for zz, fx in enumerate(fx_values):  
        
        # now this iterates over the radius. 
        line = xzypI_order[xzypI_order[:,0]==fx,:]
        x = line[:,5].astype(np.int); r = np.max(x)
        p = line[:,3]
        radii_x.append(r)
        
        # get the intensity along the line. 
        max_pixel = np.clip(emb_map[fx, x-1], 0, 255)   # Gives the max_pixels in a correctly ordered array for each fx slice
        xstretch = np.linspace(np.min(p), np.max(p), max_order)
        xnew = np.hstack([np.argmin(np.abs(pp-xstretch))+1 for pp in p])
        
        ynew = zz * np.ones(len(xnew))
        mapped_coords.append(np.hstack([line, max_pixel[:,None], ynew[:,None], xnew[:,None]]))

        
    """
    return the mapped coordinates and maximum radii 
    """        
    
    return np.vstack(mapped_coords), np.hstack(radii_x)
    
    
def interpolate_projection(mapped_coords, radii_x, return_no_interp=False):
    
    ''' 
        Interpolates the 2D image of the projection.  
            Inputs: 
                    self.full_array_order_xpo
                    self.emb_map
            Ouputs: 
                    none
            Returns: 
                    none
    '''
    import numpy as np 
    from scipy.interpolate import griddata
    
    n_rows = len(radii_x)
    n_cols = np.max(radii_x)
    
    if return_no_interp:
        proj_image = np.zeros((n_rows, n_cols))
        proj_image[mapped_coords[:,-2].astype(np.int), mapped_coords[:,-1].astype(np.int)-1] = mapped_coords[:,-3]
        proj_image_r = np.roll(proj_image, shift=n_cols//2, axis=1)
    
#    mapped_coords[0]
#    print mapped_coords[:,[-1,-2]]
    grid_x, grid_y = np.meshgrid(range(n_cols), range(n_rows))
    x = mapped_coords[:,-1] -1 
    y = mapped_coords[:,-2]
    proj_image_interp = griddata(np.vstack([x,y]).T, mapped_coords[:,-3], (grid_x, grid_y), method='linear')
    proj_image_interp_r = np.roll(proj_image_interp, shift=n_cols//2, axis=1)

    if return_no_interp:
        return (proj_image_interp, proj_image_interp_r), (proj_image, proj_image_r)
    else:
        return (proj_image_interp, proj_image_interp_r)
   
        
 
 def gen_ref_phi_map(ref_coord_set):

    """
    Builds a unique cylindrical projection reference map based on a reference mapping.  
    """
    uniq_x = np.unique(ref_coords[:,0])
    ref_ordered = [ref_coords[ref_coords[:,0]==z] for z in uniq_x ]
    
    stats = np.array([[len(r), np.min(r[:,3]), np.max(r[:,3])] for r in ref_ordered])
    max_radii = int(np.max(stats[:,0]))
    
    ref_map = np.zeros( ( len(stats), max_radii))
    
    for i in range(len(stats)):
        ref_map[i] = np.linspace(stats[i,1], stats[i,2], max_radii) # evenly place the angles # (assumes 0-2pi)
        
    return ref_map 


    
def map_intensities(mapped_coords, query_I, shape, interp=True):
    """
    Given a coordinate mapping
    """
    from scipy.interpolate import griddata

    if interp:
        # m_coords = np.vstack([m for m in mapped_coords if len(m)>0])
        m_coords = m_coords.astype(np.int) # convert to int. 

        grid_x, grid_y = np.meshgrid(range(shape[1]), range(shape[0]))
        mapped_image_interp = griddata(m_coords[:,::-1], query_I, (grid_x, grid_y), method='linear')

        mapped_img = np.zeros(shape)
        mapped_img[m_coords[:,0], m_coords[:,1]] = query_I

        return mapped_image_interp, mapped_img

    else:
        mapped_img = np.zeros(shape)
        # m_coords = np.vstack([m for m in mapped_coords if len(m)>0])
        m_coords = m_coords.astype(np.int) # convert to int. 
        
        # how to handle non unique mapping? 
        mapped_img[m_coords[:,0], m_coords[:,1]] = query_I
        
        return mapped_img
    


# function used by the map_coords_to_ref_coords_map, can be used directly if one already has the info. 
def map_coords_to_ref_map(query_coords_order, ref_map):
    """
    given a ref_map assigns the query coordinates onto the positions of the map using fast NN trees.
        note query_coords are ordered.
    """    
    from sklearn.neighbors import NearestNeighbors
    
    # define the tree
    neigh = NearestNeighbors(n_neighbors=1)
    
    neighbor_index = []
    
    for ii, q in enumerate(query_coords_order):
        if len(q) > 0 : 
            query = q[:,3][:,None] # get the query phi
            ref = ref_map[ii]
            neigh.fit(ref[:,None])
            res = neigh.kneighbors(query, return_distance=False)
            neighbor_index.append(np.hstack([ii*np.ones(len(q))[:,None], res]))

    return neighbor_index


def map_coords_to_ref_coords_map_cylindrical(query_coords, ref_params):
    """
    given a ref_map assigns the query coordinates onto the positions of the map using fast NN trees.
        note query_coords are ordered.
    """    
    import numpy as np 

    COM = ref_params['center']
    query_coords_derefer = query_coords.copy()
    query_coords_derefer[:,1] = query_coords[:,1] - COM[1]
    query_coords_derefer[:,2] = query_coords[:,2] - COM[2] 
    query_phi = np.arctan2(test_coords_derefer[:,2], test_coords_derefer[:,1])
    query_coords_ = np.hstack([query_coords, query_phi[:,None]])

    ref_coords = ref_params['map_coords']
    ref_radii = ref_params['radii']
    ref_max_radii = ref_params['max_radii']
    ref_map = gen_ref_phi_map(ref_coords) # this #ref map is purely generated by a linear interpolation between min and max.. ( of the angles )
   

    uniq_z = np.unique(ref_coords[:,0])
    query_coords_rearrange = [query_coords_[query_coords_[:,0]==z] for z in uniq_z ] 
   
    # map coordinates. 
    mapped_coords = map_coords_to_ref_map(query_coords_rearrange, ref_map) 

    # unflatten the output. 
    mapped_coords = np.vstack([m for m in mapped_coords if len(m)>0]).T
    query_coords_out = np.vstack([q for q in query_coords_rearrange if len(q)>0]).T

    return np.hstack([query_coords_out, mapped_coords]), ref_map

    