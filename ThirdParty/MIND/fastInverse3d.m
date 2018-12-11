function [u2,v2,w2]=fastInverse3d(u1,v1,w1)
%following the approach of to Chen et al. (2008)
[m,n,o]=size(u1);
[x,y,z]=meshgrid(1:n,1:m,1:o);
x=single(x);
y=single(y);
z=single(z);

u2=zeros(size(u1),'single');
v2=zeros(size(v1),'single');
w2=zeros(size(w1),'single');
for i=1:10
    xu=min(max(x+u2,1),n);
    yv=min(max(y+v2,1),m);
    zw=min(max(z+w2,1),o);
    
    u2=-trilinearSingle(single(u1),xu,yv,zw);
    v2=-trilinearSingle(single(v1),xu,yv,zw);
    w2=-trilinearSingle(single(w1),xu,yv,zw);

end

