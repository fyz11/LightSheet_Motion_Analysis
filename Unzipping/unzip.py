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
        

def segment_embryo(im_array, I_thresh=10, ksize=3):
    
    """
    im_array : grayscale 3d volume
    """
    from skimage.morphology import ball, binary_closing, binary_dilation
    
    n_z, n_y, n_x = im_array.shape
    
    mask = im_array >= I_thresh
    mask = keep_largest_component(mask)
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
    
    
def unwrap_embryo_surface(im_array, embryo_mask, contour_mask, depth=20, voxel=5.23):   
   
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
    emb_map = map_max_projection(im_array, center_of_mass, xzypI_order, max_order, depth=depth, voxel=voxel) # max order is the maximum width.
    emb_map_wrap, emb_map_normalised, radii = normalise_projection(emb_map, xzypI_order, max_order)

    # save as a dictionary. 
    # to convert to cylindrical needs the required COM, also needs the max_radii in order to set the map size. 
    unwrap_params = {'max_radii':max_order,
                     'radii':radii, 
                     'center':center_of_mass}

    return unwrap_params, (emb_map, emb_map_wrap, emb_map_normalised)  
    
    
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

        order = 1               
        order_matrix = [1]

        for l in range(0,len(xzypI_order)-1):                                # If the x value is the same for the next point i.e. same slice, increase the order for that coordinate
            if xzypI_order[l+1,0] == xzypI_order[l,0]:
                order+=1
            else:                                                                   # Else if the x value increases i.e. it is a new slice, reset the order count to 1
                order=1
            order_matrix.append(order)
#            order_matrix = np.append(order_matrix,order)                            # Append the order to the order matrix
        order_matrix = np.hstack(order_matrix)[:,None]
   
#        order_matrix = np.reshape(order_matrix,(len(order_matrix),1))               # Reshape for appending
        xzypI_order = np.append(xzypI_order, order_matrix, axis=1)    # Append the order column to the data
        
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
        max_pixel = np.max(mask_line[mask_distance_less]) # max projection along the line !.
        
        emb_map[fx,fo-1] = max_pixel    #-1 as array starts from 0

    emb_map=emb_map[np.any(emb_map!=0, axis=1)]      
        
    return emb_map
    
    
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
    from scipy.interpolate import interp1d
    
    fx_values = np.unique(xzypI_order[:,0]).astype(np.uint)   # Finds all the unique values of fx i.e. each slice basically. hm.... 
    emb_map_normalised = np.zeros_like(emb_map)
    emb_map_wrap = np.zeros_like(emb_map)
    
    radii_x = []
    
    for fx in fx_values:  
        
        # now this iterates over the radius. 
        x = xzypI_order[np.where(xzypI_order[:,0]==fx), 5].astype(np.uint)   # Finds all the order values for that x = full_array_orderxpo[:,5] 
        radii_x.append(np.max(x))
                        
        for fo in x:
            max_pixel = emb_map[fx,fo-1]                                                    # Gives the max_pixels in a correctly ordered array for each fx slice
        x = np.reshape(x,len(x.T))
        
        # interpolate if not enough elements hm ? 
        if len(x)<5:
            f2 = interp1d(x.T, max_pixel, kind='linear')                    # If at the tip of the embryo, interpolate between fo(order) and fx(x-value) linearly as x vs y on map
        else:
            f2 = interp1d(x.T, max_pixel, kind='cubic')                     # for the rest of the pixels, interpolate between fo(order) and fx(x-value) cubically x vs y on map
        xnew = np.linspace(1, len(x), num=max_order, endpoint=True)         # Interpolate the correct number of values to normalise each width to the maximum
                                                                            # True means does include the last sample, as it starts from 1 
        emb_map_normalised[fx,:] = f2(xnew)                                 # Stretches the interpolated values into a normalised array
    
    # this is 
    for fx in range(0,int(max(fx_values)+1)):      # write length
        for fo in range(0,int(max_order)):      # write length
            if emb_map_normalised[fx,fo]>254:        # Prevents saturation tending to zero
                emb_map_normalised[fx,fo]=255 # is the max range. 
                
            max_pixel2 = emb_map_normalised[fx,fo]
            
            # hm.... double check this! rounding, important.  
            F=fo+np.rint((max_order)/2)     # Wrap around so best data is in the middle # round to the nearest integer. 
            if F > (max_order-1):
                F=F-(max_order)
            emb_map_wrap[fx,int(F)] = max_pixel2
    
    # why is this needed? 
    emb_map_normalised=emb_map_normalised[np.any(emb_map_normalised!=0, axis=1)]
    emb_map_wrap=emb_map_wrap[np.any(emb_map_wrap!=0, axis=1)]
#    emb_map=self.emb_map[np.any(self.emb_map!=0, axis=1)]        
#    print('Finished normalising the embryo map to the largest width. End of code.')
    
    return emb_map_wrap, emb_map_normalised, np.hstack(radii_x)
    
    
    
 