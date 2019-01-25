function [T, vargout] = register3D_SIFT(im1, im2, downsample, lib_path, return_img, nnthresh, sigmaN, numKpLevels)

    addpath(lib_path) % add the sift library path
    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
        
    %%% for v. large images recommended to learn the transform on the
    %%% downsampled image. 
    downsample_factor=downsample;
    
    im1_ = uint8(imresize3d(im1, 1./downsample_factor)); 
    im2_ = uint8(imresize3d(im2, 1./downsample_factor));
    
    %% V. important ! rescale the image intensity.
    im1_ = reshape(imadjust(im1_(:)), size(im1_));
    im2_ = reshape(imadjust(im2_(:)), size(im2_)); % this does the same job as imadjustn
    
    %% registering with 3D sift library.
    %'registering with sift'
    [A, matchSrc, matchRef] = registerSift3D(im1_,im2_, 'nnThresh',nnthresh, 'sigmaN', sigmaN, 'numKpLevels', numKpLevels); % this looser threshold is needed to get matches.
    A(1:3,4) = A(1:3,4) * downsample_factor; % correct the scaling due to downsampling.

    if return_img == 1
        'applying transform in matlab'
        %%% now apply the transform A ( 4x3 matrix giving affine transformatin from ref to src)
        %%% s.t.[xt yt zt]' = A * [x y z 1]'
        % convert to matlab affine transformation object (scaling the translation component accordingly for the larger image.) 
        Aprime = zeros(4,4);
        Aprime(1:3,1:4) = A; 
        Aprime(4,4) = 1;
        Aprime = Aprime'; % transpose
        
        A_tfm = affine3d(Aprime);

        %% imwarp, noting that matlab subscript uses (y,x,z) but applies geometric (x,y,z)!
        registered = imwarp(permute(im2, [2,1,3]), A_tfm, 'OutputView',imref3d([size(im1,2), size(im1,1), size(im1,3)]));
        registered = uint8(permute(registered, [2,1,3])); % permute back
        vargout{1} = registered;
    end

    T=A %save out the transform
end