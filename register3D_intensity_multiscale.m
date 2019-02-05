function [transform] = register3D_intensity_multiscale(im1file,im2file, outsavefile, transformfile, initialise, downsample_factor, mode, iterations, type, return_img)

    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    
    % read in the tiffile images and cast into uint8
    im1 = uint8(loadtiff(im1file)); % (y,z,x python convention)
    im2 = uint8(loadtiff(im2file));

    % iterate through the downsampling. 
    n_scales = length(downsample_factor)

    % set up the basic registration settings
    [optimizer, metric] = imregconfig(mode);
    % 1+1 evolutionary
    optimizer.InitialRadius = optimizer.InitialRadius/3.5; % this appears more stable, c.f. the documentation in matlab for aligning medical images.

    if initialise==1
        initial_transform=load(transformfile,'tform'); 
        initial_transform=initial_transform.tform;
        initial_transform = initial_transform'; % transpose, to be in matlab format. 
    end

    for level=1:n_scales
        % downsample image to correct scale.  
        img1_ = imresize3d(im1, 1./downsample_factor(level));
        img2_ = imresize3d(im2, 1./downsample_factor(level));

        % contrast adjust? 
        img1_ = reshape(imadjust(img1_(:)), size(img1_));
        img2_ = reshape(imadjust(img2_(:)), size(img2_)); % this does the same job as imadjustn

        optimizer.MaximumIterations = iterations(level); % set the specified number of iterations at this level.

        if level == 1
            if initialise==1
                % scale the initial transform (defined for the full volume to a subvolume for the translation). 
                initial_transform(4,1:3) = initial_transform(4,1:3) / double(downsample_factor); % we assume the initial transform is for the full scale. 
                initial_transform = affine3d(initial_transform) % for some reason this initialisation is not good? 
                initial_transform.T % print for debugging.
                tform = imregtform(img2_, img1_, type, optimizer, metric, 'InitialTransformation', initial_transform, 'PyramidLevels', 1);
            else
                tform = imregtform(img2_, img1_, type, optimizer, metric, 'PyramidLevels', 1);
            end
        else
            % we use the previous to initialise. 
            tf_prev = tform.T; 
            tf_prev(4,1:3) = tf_prev(4,1:3) * double(downsample_factor(level-1)/downsample_factor(level)); % correct for translational discrepancy. 
            tform = affine3d(tf_prev)
            tform = imregtform(img2_, img1_, type, optimizer, metric, 'InitialTransformation', tform, 'PyramidLevels', 1);
        end
    end

    % (z,y,x) in python convention.
    transform = tform.T; % grab the matrix 
    transform = transform'; % transpose the matrix
    transform(1:3,4) = transform(1:3,4)*downsample_factor(length(downsample_factor)); %multiply the last column by the very last downsample_factor. 

    if return_img == 1
        tform.T = transform'; % update the matrix inside. 
        movingRegistered = imwarp(im2, tform,'OutputView', imref3d(size(im1)));
        
        % save the image as tiff.
        saveastiff(movingRegistered, outsavefile)
    end