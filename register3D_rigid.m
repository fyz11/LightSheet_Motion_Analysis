function [movingRegistered, transform] = register3D_rigid(im1file,im2file, transformfile, initialise, downsample_factor, mode, iterations, type)

    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    
    im1 = load(im1file,'im1'); im1=im1.im1; 
    im2 = load(im2file,'im2'); im2=im2.im2;
    
    %%%%
    %  TO DO: implement ver. change in matlab. 
    %%%%
    %img1_ = imresize(im1, 1./downsample_factor);
    %img2_ = imresize(im2, 1./downsample_factor);
    img1_ = imresize3d(im1, 1./downsample_factor); % 3d resizing. 
    img2_ = imresize3d(im2, 1./downsample_factor);
    
    [optimizer, metric] = imregconfig(mode);
    optimizer.MaximumIterations = iterations;
    
    
    % do we initialise or not?
    if initialise==1
        initial_transform=load(transformfile,'tform'); initial_transform=initial_transform.tform;
        initial_transform = affine3d(initial_transform);
        tform = imregtform(img2_, img1_, type, optimizer, metric, 'InitialTransformation', initial_transform, 'PyramidLevels', 3);
    else
        tform = imregtform(img2_, img1_, type, optimizer, metric, 'PyramidLevels', 3);
    end 
    
    transform = tform.T; % grab the matrix 
    transform(4,1:3) = transform(4,1:3) * downsample_factor    
    tform.T = transform %update the translation scaling 
    
    movingRegistered = imwarp(im2, tform,'OutputView', imref3d(size(im1)));
    
    %% needs faster saving functionality hm... 
    save('tmp/registered.mat', 'movingRegistered') % save the variable.
    
    
    % register the first one. 
    % translation step is not helpful at all, still need to fix the displacement later. 
    %%%%%%%%%%%%%%%%%%%%
    %'fixing translation'
    %movingRegistered = imwarp(img2_, tform,'OutputView',imref3d(size(img1_)));
    %movingRegistered_ = imwarp(im2, tform,'OutputView',imref3d(size(im1)));
    
    %% register the translation now. 
    %optimizer.MaximumIterations = iterations/2;
    %tform = imregtform(movingRegistered, img1_, 'translation', optimizer, metric);
    %
    %movingRegistered = imwarp(movingRegistered_, tform,'OutputView', imref3d(size(im1)));