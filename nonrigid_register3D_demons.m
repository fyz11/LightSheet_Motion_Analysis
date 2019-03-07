function [done] = nonrigid_register3D_demons(ref_file,moving_file, outsavefile, outtransformfile, downsample, warps, alpha)
    %%%
    %Void function
    %-------------

    %ref_file: reference volume image
    %moving_file: moving volume image
    %outsavefile: filepath to save the registered moving_file
    %transformfile: filepath to save the registered parameters for the moving pair
    %levels: what is the multiresolution setting in the registration
    %warps: how many iterations to run the algorithm at each level   
    %%%
    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    addpath('ThirdParty/MIND');
    
    %im1 = load(im1file,'im1'); im1=im1.im1; 
    %im2 = load(im2file,'im2'); im2=im2.im2;
    im1 = uint8(loadtiff(ref_file)); % (y,z,x python convention)
    im2 = uint8(loadtiff(moving_file));
    
    % downsample the volumes.
    vol1=volresize(im1,size(im1)./downsample);
    vol2=volresize(im2,size(im2)./downsample);
    
%     % match histogram?
%     fixedHist = imhist(vol1);
%     vol2 = histeq(vol2,fixedHist);

    % run demons
    % normalise histogram images. 
    [D,~] = imregdemons(vol2,vol1,warps, 'PyramidLevels', length(warps), 'AccumulatedFieldSmoothing', alpha);
      
    % if downsample is not 1 then we resize the field then apply
    [u1_,v1_,w1_]=resizeFlow(squeeze(D(:,:,:,1)),squeeze(D(:,:,:,2)),squeeze(D(:,:,:,3)),size(im1));
    D_up = cat(4, u1_,v1_,w1_);
    deformed2 = imwarp(im2, D_up); % warp the original according to new flow. 
    
    % save only the downsampled fields. 
    u1 = squeeze(D(:,:,:,1));
    v1 = squeeze(D(:,:,:,2));
    w1 = squeeze(D(:,:,:,3));
    
    %if isfile(outtransformfile)
    if exist(outtransformfile, 'file') == 2
        % File exists.
        delete(outtransformfile)
        save(outtransformfile,'u1','v1','w1')
    else
        % File does not exist.
        save(outtransformfile,'u1','v1','w1')
    end
        
    %if isfile(outsavefile)
    if exist(outsavefile, 'file') == 2
        % File exists.
        delete(outsavefile)
        saveastiff(uint8(deformed2), outsavefile)
    else
        % File does not exist.
        saveastiff(uint8(deformed2), outsavefile)
    end
    
    done = 1;