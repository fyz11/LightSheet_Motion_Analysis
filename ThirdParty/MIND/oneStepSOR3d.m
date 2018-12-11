function [u1,v1,w1]=oneStepSOR3d(Ix,Iy,Iz,It,u0,v0,w0,alpha)

iterations=25; omega=1.98;

[m,n,o]=size(Ix);

hs=fspecial('gaussian',[11,1],0.5);

Ix=single(Ix); Iy=single(Iy); Iz=single(Iz); It=single(It);

%local smoothing (similar to Bruhn et al. 2005 IJCV)
A11=volfilter(Ix.^2,hs,'replicate'); A12=volfilter(Ix.*Iy,hs,'replicate');
A13=volfilter(Ix.*Iz,hs,'replicate'); A14=volfilter(Ix.*It,hs,'replicate');
A22=volfilter(Iy.^2,hs,'replicate'); A23=volfilter(Iy.*Iz,hs,'replicate');
A24=volfilter(Iy.*It,hs,'replicate'); A33=volfilter(Iz.^2,hs,'replicate');
A34=volfilter(Iz.*It,hs,'replicate');
clear Ix; clear Iy; clear Iz;

%Gauss-Newton update-step with Diffusion Regularisation
[u1,v1,w1]=pointsor3d2(A11,A12,A13,A14,A22,A23,A24,A33,A34,u0,v0,w0,iterations,alpha,omega,m,n,o);
clear A11;clear A12; clear A1; clear A14; clear A22; clear A23; clear A24; clear A33; clear A34;
u1=u1+u0;
v1=v1+v0;
w1=w1+w0;






