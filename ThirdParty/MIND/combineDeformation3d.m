function [u_combined,v_combined,w_combined]=combineDeformation3d(u1st,v1st,w1st,u2nd,v2nd,w2nd,method)

if nargin<7
    method='compositive';
end

if strcmp(method,'compositive')
    u_combined=volWarp(u1st,u2nd,v2nd,w2nd)+u2nd;
    clear u1st;
    v_combined=volWarp(v1st,u2nd,v2nd,w2nd)+v2nd;
    clear v1st;
    w_combined=volWarp(w1st,u2nd,v2nd,w2nd)+w2nd;
    clear w1st;
else
    u_combined=u1st+u2nd;
    v_combined=v1st+v2nd;
    w_combined=w1st+w2nd;
end
    