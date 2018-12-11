function [done] = nonrigid_register3D_MIND(ref_file,moving_file, outsavefile, outtransformfile, alpha, levels, warps)

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
    
    % run MIND3D 
    [u1,v1,w1,u2,v2,w2,deformed1,deformed2]=deformableRegistration(im1,im2,alpha, levels, warps);
        
        
    max(max(max(deformed2)))
        
    %if isfile(outtransformfile)
    if exist(outtransformfile, 'file') == 2
        % File exists.
        delete(outtransformfile)
        save(outtransformfile,'u1','v1','w1','u2','v2','w2')
    else
        % File does not exist.
        save(outtransformfile,'u1','v1','w1','u2','v2','w2')
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