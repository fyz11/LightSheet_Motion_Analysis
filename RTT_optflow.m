function val = RTT_optflow(tif_files, save_dir, method, refid, refine_level, accFactor, downsample_factor, alpha, I_thresh, library_path)

    %%%%%%%%%%%%%%%%%%%%%%%%
    %   Settings found for embryo
    % id_registration_method = 2; % Horn-Schunk L2L1 algorithm 
    % reference_dynamic = 1; % use first image. 
    % nb_raffinement_level = 1; % number of refinements. 
    % accelerationFactor = 1; % set this off, since we want the flow at the highest ( finest detail level ), original resolution. but doesn't seem to make too much difference so its fine.  
    % downsample_factor = 1; % this was only appropriate for previous. 
    % alpha = 0.1; % set this low, else everything is smooth! but not too low!
    %
    %    Inputs:
    %       tif_files: list of tif files to process. (converts to a cell.)
    %       method: opt_flow method to use. 
    %       refid: which frame to use as the reference.
    %       refine_level: this is the number of refinements to use. 
    %       accFactor: skip levels to process the pyramid faster.      
    %       downsample_factor: process on a coarser image for speed.
    %       alpha: the amount of regularisation to use. 
    %
    %%%%%%%%%%%%%%%%%%%%%%%%
    addpath(library_path); % add the library path. (make this a variable. )

    %addpath(library_path)
    addpath('Optical_Flow') %add the relative folder paths to include stuff. 
    
    %%%
    % set the parameters:
    id_registration_method = double(method) % Horn-Schunk L2L1 algorithm 
    reference_dynamic = double(refid); % use first image. 
    nb_raffinement_level = double(refine_level); % number of refinements. 
    accelerationFactor = double(accFactor); % set this off, since we want the flow at the highest ( finest detail level ), original resolution. but doesn't seem to make too much difference so its fine.  
    I_thresh = double(I_thresh)
    %%%
    
    mkdir(save_dir) % create the save directory. 
    val = 0;
    
    for i=1:length(tif_files)-1
    
        tiff1 = tif_files{i};
        tiff2 = tif_files{i+1};
    
        im1 = loadtiff(tiff1);
        im2 = loadtiff(tiff2);
        
        % masking is necessary to avoid artifacts from registration. (minimise)
        im1(im1<=I_thresh) = 0;
        im2(im2<=I_thresh) = 0;
        
        % downsample for speed:
        im1 = imresize(im1, 1./downsample_factor); 
        im2 = imresize(im2, 1./downsample_factor); 
        
        % concatenate the two images. 
        data = cat(4, im1, im2);
        data = im2double(data); % converted to doubles. 
        
        % compute the motion field. 
        motion_field = opt_flow3D(data, id_registration_method, reference_dynamic, nb_raffinement_level, accelerationFactor,alpha);
        
        max(motion_field(:))
        '===='        
        % save the motion fields. 
        savematfilename = strcat(save_dir, '/optflow3D_field_',int2str(i),'.mat');
        save(savematfilename, 'motion_field'); % save this to make it functional. else wrongly interpret savematfilename as the name. 
    
    end
    
    % if completes successfully. 
    val = 1;
    

    
    
    
    
    
    
    
    
    
