function [T] = register3D_SIFT_wrapper(imgfile1, imgfile2, outsavefile, downsample, lib_path, return_img, nnthresh, sigmaN, numKpLevels)

    %%%%%%%%%%%%%%%%%%%
    %   imgfile1 -
    %   imgfile2 - 
    %   outsavefiles: what we save the registered image as 
    %   mode: 1: list of paired files, 2: list of sequential files. 
    %   tmp_folder: is where we dump the output temporarily to as matlab
    %   array to feed into python ( much much faster ).
    %%%%%%%%%%%%%%%%%%%

    addpath(lib_path) % add the sift library path
    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    
    im1 = loadtiff(imgfile1); im1 = permute(im1,[3,1,2]);
    im2 = loadtiff(imgfile2); im2 = permute(im2,[3,1,2]);
    
    if return_img == 1
        'matlab saving image'
        [T, registered] = register3D_SIFT(im1, im2, downsample, lib_path, return_img, nnthresh, sigmaN, numKpLevels);
        
        % save out registered. 
        saveastiff(registered, outsavefile);
    else
        'returning transform'
        T = register3D_SIFT(im1, im2, downsample, lib_path, return_img, nnthresh, sigmaN, numKpLevels);
    end
    
end