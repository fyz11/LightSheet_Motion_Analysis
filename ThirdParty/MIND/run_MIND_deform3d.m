

% load in the images. 
vol1 = loadtiff('TP_49.tif');
vol2 = loadtiff('TP_50.tif');
alpha = 0.1

% compute the deformation field ? 
tic
[u1,v1,w1,u2,v2,w2,deformed1,deformed2]=deformableRegistration(vol1,vol2,alpha);
toc

%%
%%
%%

figure(1)
title('before')
imshowpair(vol1(:,:,120), vol2(:,:,120));

figure(2)
title('after')
imshowpair(vol1(:,:,120), deformed2(:,:,120));




