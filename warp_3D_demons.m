function [success] = warp_3D_demons(imgfile, saveimgfile, transformfile, downsample, direction)
    %%%
    %Void function
    %-------------

    %imgfile: input volume image
    %transformfile: filepath to the registered parameters for the moving pair
    %downsample: original highest level downsampling 
    %%%
    addpath('Utility_Functions'); % required to install the tiff loading script and 3d resizing. 
    addpath('Registration');
    addpath('ThirdParty/MIND');
    
    im = uint8(loadtiff(imgfile)); % (y,z,x python convention)
    
    size(im)
    % load the transform parameters. 
    u1 = load(transformfile, 'u1'); u1 = u1.u1;
    v1 = load(transformfile, 'v1'); v1 = v1.v1;
    w1 = load(transformfile, 'w1'); w1 = w1.w1;

    size(u1)
    % resize the flow to the correct size as the input image. 
    % if downsample is not 1 then we resize the field then apply
    [u1_,v1_,w1_]=resizeFlow(squeeze(u1), squeeze(v1), squeeze(w1), size(im));
    D_up = cat(4, u1_,v1_,w1_);

    if direction == 1
        deformed = imwarp(im, D_up); % warp the original according to new flow. 
    end

    if direction == -1
        deformed = imwarp(im, -D_up); % warp the original according to the reverse of the flow.
    end

    %if isfile(outsavefile)
    if exist(saveimgfile, 'file') == 2
        % File exists.
        delete(saveimgfile)
        saveastiff(uint8(deformed), saveimgfile);
    else
        % File does not exist.
        saveastiff(uint8(deformed), saveimgfile);
    end
    
    success = 1;