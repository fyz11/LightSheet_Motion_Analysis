function [u1s,v1s,w1s,u2s,v2s,w2s]=symmetricMapping3d(u1,v1,w1,u2,v2,w2)


[u2i,v2i,w2i]=fastInverse3d(u2./2,v2./2,w2./2);
[u1i,v1i,w1i]=fastInverse3d(u1./2,v1./2,w1./2);
[u1s,v1s,w1s]=combineDeformation3d(u1./2,v1./2,w1./2,u2i,v2i,w2i,'compositive');
[u2s,v2s,w2s]=combineDeformation3d(u2./2,v2./2,w2./2,u1i,v1i,w1i,'compositive');



