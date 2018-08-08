# LightSheet_Motion_Analysis
3D Motion Analysis of Embryos acquired with Lightsheet Microscopy 


Dependencies:
-------------
Python 2.7 (only the print functionality) \
RealTITracker (3D optical flow), http://bsenneville.free.fr/RealTITracker/ \
Mayavi\
Matlab (rigid/similarity registration)

Modules:
--------

[x] 3D rigid/similarity alignment using MATLAB registration toolbox

>	- this removes growth/translation artifacts

[x] 3D optical flow estimation of cellular motion patterns following alignment (Requires RTTtracker Scripts)\
[x] 3D segmentation of embryos using thresholding \
[x] 3D tracks\
[x] 3D->2D unwrapping module: given an embryo, automatically transform to cylindrical coordinates and unwrap the surface. Intensity is projected based on drawing a line from the surface coordinate to the center of mass. 

To Do:
-------
[ ] given a 2nd set of coordinates, apply the learnt 3D->2D unpeeling. 
