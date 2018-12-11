function [u1,v1,w1,u2,v2,w2,deformed1,deformed2]=deformableRegistration(vol1,vol2,alpha,levels,warps)
% Symmetric Gauss-Newton registration
%
% written by Mattias Heinrich.
% Copyright (c) 2012, University of Oxford. All rights reserved.
% Institute of Biomedical Engineering (IBME)
% See the LICENSE.txt file in the root folder
%
%
% If you use this implementation please cite:
% M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
% Descriptor for Multi-Modal Deformable Registration"
% Medical Image Analysis (2012), 16(7): 1423-1435 and
%
% M.P. Heinrich et al.: "Non-local Shape Descriptor: A New Similarity Metric
% for Deformable Multi-modal Registration" MICCAI (2) 2011: 541-548
%
% Contact: mattias.heinrich(at)eng.ox.ac.uk
%
% for details see Sec. 4.2. in paper
% vol1: input volume (3D)
% vol2: input volume (3D) needs to has the same resolution and dimensions
%
% alpha: regularisation (see Eq. 12) (higher value give smoother field)
%
% levels: no. of resolution levels
%
% u1,v1,w1: output flow-field which transforms vol2 towards vol1
% u2,v2,w2: output flow-field which transforms vol1 towards vol2
%
% deformed1: transformed vol1 (into anatomical space of vol2)
% deformed2: transformed vol2 (into anatomical space of vol1)
%

% to run this code you need to compile (mex) the following files
% trilinearSingle.cpp
% pointsor3d2.c

if nargin<3
    alpha=0.1;
end

if nargin<4
    levels=[4,2,1] % resolution levels for multilevel registration
end

if nargin<5
    warps=[8,4,2] % number of warps per level (increase if needed for better precision)
end

alpha
levels
levels = levels';
warps = warps';

% convert to single precision. 
vol1=single(vol1); vol2=single(vol2);

u1=zeros(2,2,2,'single'); v1=u1; w1=u1; u2=u1; v2=u1; w2=u1;
% initialise flow fields with 0

% h=waitbar(0,'Performing deformable (multi-modal) registration');
% deformable (multi-modal) registration 
complexity=[];
for j=1:length(levels)
    complexity=[complexity,ones(1,warps(j)).*(levels(j)).^(-3)];
end
complexity=cumsum(complexity)/sum(complexity);

% time0=cputime; count=0; % some calculations to determine run-time

for j=1:length(levels)

    maxwarp=warps(j);

    % blurring amount (should i allow this to be changed?)
    hs=fspecial('gaussian',[15,1],(levels(j))/2);
    vol1f=volresize(volfilter(vol1,hs),size(vol1)./levels(j));
    vol2f=volresize(volfilter(vol2,hs),size(vol2)./levels(j));
    % resize volumes for current level
        
    [u1,v1,w1]=resizeFlow(u1,v1,w1,size(vol1f));
    [u2,v2,w2]=resizeFlow(u2,v2,w2,size(vol1f));
    
    % upsample flow to current level
    
    if maxwarp > 0
        for i=1:maxwarp
            warped1=volWarp(vol2f,u1./2,v1./2,w1./2);
            warped2=volWarp(vol1f,u2./2,v2./2,w2./2);
            % transform volumes to intermediate space
            
            mind1=MIND_descriptor(warped1);
            mind2=MIND_descriptor(warped2);
            % extract modality independent neighbourhood descriptor
            
            [Sx,Sy,Sz,S,Sxi,Syi,Szi]=MINDgrad3d(mind1,mind2);
            % calculate derivates of MIND
            
            [u1,v1,w1]=oneStepSOR3d(Sxi,Syi,Szi,S,u1,v1,w1,alpha);
            [u2,v2,w2]=oneStepSOR3d(Sx,Sy,Sz,S,u2,v2,w2,alpha);
            % solve Euler-Lagrange equations using successive overrelaxation
            % includes (compositive) diffusion regularisation in cost term

            clear Sx; clear Sy; clear Sz; clear S; clear Sxi; clear Syi; clear Szi;
            [u1,v1,w1,u2,v2,w2]=symmetricMapping3d(u1,v1,w1,u2,v2,w2);
            % ensure symmetric Mapping (see Sec. 4.3)
            % time1=cputime; count=count+1;
            % waitbar(complexity(count));
            % disp(['remaining time is approx. ',num2str((time1-time0)/complexity(count)*(1-complexity(count))),' secs.']);
        end
    end
end
%close(h);

% warp the volumes within matlab (is this really slow with matlab engine)
deformed2=volWarp(vol2,u1,v1,w1);
deformed1=volWarp(vol1,u2,v2,w2);
% generate final output volumes

