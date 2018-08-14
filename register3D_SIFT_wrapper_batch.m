function [tforms] = register3D_SIFT_wrapper_batch(imgfiles, outsavefiles, downsample, lib_path, mode)

    %%%%%%%%%%%%%%%%%%%
    %   imgfiles: will be a cell either with (img1, img2) pairs. 
    %   outsavefiles: what we save the registered image as 
    %   mode: 1: list of paired files, 2: list of sequential files. 
    %   tmp_folder: is where we dump the output temporarily to as matlab
    %   array to feed into python ( much much faster ).
    %%%%%%%%%%%%%%%%%%%
    addpath(lib_path) % add the sift library path
    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    
    if mode == 1
        n_sets = uint16(length(imgfiles)/2);
        tforms = zeros(n_sets,3,4);
        % treat as pairs
        for i=1:n_sets
            im1 = loadtiff(imgfiles{2*i-1});
            im2 = loadtiff(imgfiles{2*i});
            
            [T, registered] = register3D_SIFT(im1, im2, downsample, lib_path);
            tforms(i,:,:) = T;
            
            % otherwise save as .mat
            saveastiff(registered, outsavefiles{i});
        end
    end
    
    if mode == 2
        tforms = zeros(length(imgfiles)-1,3,4);
        
        fixed = loadtiff(imgfiles{1});
        saveastiff(permute(fixed, [3,1,2]), outsavefiles{1}); % save the reference.
        
        % treat as sequential pairs.
        for i=1:length(imgfiles)-1
            
            moving = loadtiff(imgfiles{i+1}); % take the next image along             
            [T, registered] = register3D_SIFT(fixed, moving, downsample, lib_path);
            tforms(i,:,:) = T;
            
            saveastiff(registered, outsavefiles{i+1});
            fixed = registered; % update the fixed. 
        end
        
    end 
end