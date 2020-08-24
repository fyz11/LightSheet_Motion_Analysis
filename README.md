# LightSheet_Motion_Analysis 
3D Motion Analysis of Embryos acquired with Lightsheet Microscopy (Unpublished, work in progress)


Dependencies:
-------------
Python 2.7 (only the print functionality) \
RealTITracker (3D optical flow), http://bsenneville.free.fr/RealTITracker/ \
SIFT3D, https://github.com/bbrister/SIFT3D.git (3D feature-based registration) \
Mayavi\
Matlab (rigid/similarity + nonrigid registration) \
czifile \
tifffile 

Modules:
--------

[x] 3D rigid/similarity + nonrigid alignment using MATLAB registration toolbox

>	- this removes growth/translation + surface deformation artifacts to create fixed geometrical reference shape for unwrapping

[x] 3D optical flow estimation of cellular motion patterns following alignment (Requires RTTtracker Scripts)\
[x] 3D segmentation of embryos using thresholding \
[x] 3D tracks\
[x] 3D->2D unwrapping module: given an embryo, automatically transform to surface cartesian or distal polar coordinates and unwrap the surface. Intensity is projected based on either local maximum surface depth projection or pullback. 

To Do:
-------

