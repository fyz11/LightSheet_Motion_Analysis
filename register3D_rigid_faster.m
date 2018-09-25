function [transform] = register3D_rigid_faster(im1file,im2file, outsavefile, transformfile, initialise, downsample_factor, mode, iterations, type, return_img)

    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    
    %m1 = load(im1file,'im1'); im1=im1.im1; 
    %im2 = load(im2file,'im2'); im2=im2.im2;
    im1 = uint8(loadtiff(im1file)); % (y,z,x python convention)
    im2 = uint8(loadtiff(im2file));
    
    % 3d downsampling.
    img1_ = imresize3d(im1, 1./downsample_factor);
    img2_ = imresize3d(im2, 1./downsample_factor);

    % contrast adjustment for best results.
    img1_ = reshape(imadjust(img1_(:)), size(img1_));
    img2_ = reshape(imadjust(img2_(:)), size(img2_)); % this does the same job as imadjustn
    
    [optimizer, metric] = imregconfig(mode);
    optimizer.MaximumIterations = iterations;
    optimizer.InitialRadius = optimizer.InitialRadius/3.5; % this appears more stable, c.f. the documentation in matlab for aligning medical images.
    
    % do we initialise or not?
    if initialise==1
        initial_transform=load(transformfile,'tform'); initial_transform=initial_transform.tform;
        initial_transform = affine3d(initial_transform);
        tform = imregtform(img2_, img1_, type, optimizer, metric, 'InitialTransformation', initial_transform, 'PyramidLevels', 3);
    else
        tform = imregtform(img2_, img1_, type, optimizer, metric, 'PyramidLevels', 3);
    end 
    
    % (z,y,x) in python convention.
    transform = tform.T; % grab the matrix 
    transform = transform'; % transpose the matrix
    transform(1:3,4) = transform(1:3,4)*downsample_factor; %multiply the last column
    
    if return_img == 1
        tform.T = transform'; % update the matrix inside. 
        movingRegistered = imwarp(im2, tform,'OutputView', imref3d(size(im1)));
        
        % save the image as tiff.
        saveastiff(movingRegistered, outsavefile)
    end
    
    
 