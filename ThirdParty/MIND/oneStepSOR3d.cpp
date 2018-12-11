#include "mex.h"
#include <math.h>
#include <iostream>
#include <sys/time.h>

using namespace std;
#define printf mexPrintf
/* GAUSS-NEWTON optimisation with diffusion regularisation */

void SORsolver(float* u1,float* v1,float* w1,float* Sx,float* Sy,float *Sz,float* S,float* u0,float* v0,float* w0,float alpha,int m,int n,int o){
    
	int sz=m*n*o;
	
    float* A11=new float[sz]; float* A12=new float[sz]; float* A13=new float[sz];
    float* A14=new float[sz]; float* A22=new float[sz]; float* A23=new float[sz];
    float* A24=new float[sz]; float* A33=new float[sz]; float* A34=new float[sz];
    
    for(int i=0;i<sz;i++){
        u1[i]=u0[i]; v1[i]=v0[i]; w1[i]=w0[i];
        A11[i]=Sx[i]*Sx[i]; A12[i]=Sx[i]*Sy[i]; A13[i]=Sx[i]*Sz[i];
        A14[i]=Sx[i]*S[i]; A22[i]=Sy[i]*Sy[i]; A23[i]=Sy[i]*Sz[i];
        A24[i]=Sy[i]*S[i]; A33[i]=Sz[i]*Sz[i]; A34[i]=Sz[i]*S[i];
    }
    
    for(int i=0;i<sz;i++){
        u1[i]=0.0; v1[i]=0.0; w1[i]=0.0;
    }
    
//    volfilter(A11,m,n,o,3,0.625); volfilter(A12,m,n,o,3,0.625); volfilter(A13,m,n,o,3,0.625);
//    volfilter(A14,m,n,o,3,0.625); volfilter(A22,m,n,o,3,0.625); volfilter(A23,m,n,o,3,0.625);
//    volfilter(A24,m,n,o,3,0.625); volfilter(A33,m,n,o,3,0.625); volfilter(A34,m,n,o,3,0.625);

    //volfilter A11, ... [3,1], 0.625

    float nu1,nu2,nu3,nu4,nu5,nu6;
    float nv1,nv2,nv3,nv4,nv5,nv6;
    float nw1,nw2,nw3,nw4,nw5,nw6;
    
    int ind;
    float a11,a22,a33,B1,B2,B3;
    float w=1.98;
    int iterations=25;
    
    for(int i=0;i<sz;i++){
        A11[i]=w/(A11[i]+(6.0*alpha));
        A22[i]=w/(A22[i]+(6.0*alpha));
        A33[i]=w/(A33[i]+(6.0*alpha));
    }
    
    float* B1_m=new float[sz];
    float* B2_m=new float[sz];
    float* B3_m=new float[sz];

    for(int k=0;k<o;k++){
        int kp=min(k+1,o-1);
        int kn=max(k-1,0);
        for(int j=0;j<n;j++){
            int jp=min(j+1,n-1);
            int jn=max(j-1,0);
            for(int i=0;i<m;i++){
                int ip=min(i+1,m-1);
                int ine=max(i-1,0);
                ind=i+j*m+k*m*n;
                B1_m[ind]=-A14[ind]+alpha*(u0[ine+j*m+k*m*n]+u0[i+jn*m+k*m*n]+u0[ip+j*m+k*m*n]+u0[i+jp*m+k*m*n]+u0[i+j*m+kn*m*n]+u0[i+j*m+kp*m*n]-6*u0[ind]);
                B2_m[ind]=-A24[ind]+alpha*(v0[ine+j*m+k*m*n]+v0[i+jn*m+k*m*n]+v0[ip+j*m+k*m*n]+v0[i+jp*m+k*m*n]+v0[i+j*m+kn*m*n]+v0[i+j*m+kp*m*n]-6*v0[ind]);
                B3_m[ind]=-A34[ind]+alpha*(w0[ine+j*m+k*m*n]+w0[i+jn*m+k*m*n]+w0[ip+j*m+k*m*n]+w0[i+jp*m+k*m*n]+w0[i+j*m+kn*m*n]+w0[i+j*m+kp*m*n]-6*w0[ind]);
            }
        }
    }


    
    for(int iter=0;iter<iterations;iter++)
    {
        
        for(int k=0;k<o;k++){
            int kp=min(k+1,o-1);
            int kn=max(k-1,0);
            for(int j=0;j<n;j++){
                int jp=min(j+1,n-1);
                int jn=max(j-1,0);
                for(int i=0;i<m;i++){//normal-graph
                    //for(int i=(iter+k+j)%2;i<m;i+=2){//bipartite-graph
                    
                    int ip=min(i+1,m-1);
                    int ine=max(i-1,0);
                    
                    ind=i+j*m+k*m*n;
                    
                    //SOR EQUATIONS
                    
                    B1=alpha*(u1[ine+j*m+k*m*n]+u1[i+jn*m+k*m*n]+u1[ip+j*m+k*m*n]+u1[i+jp*m+k*m*n]+u1[i+j*m+kn*m*n]+u1[i+j*m+kp*m*n])+B1_m[ind];
                    B2=alpha*(v1[ine+j*m+k*m*n]+v1[i+jn*m+k*m*n]+v1[ip+j*m+k*m*n]+v1[i+jp*m+k*m*n]+v1[i+j*m+kn*m*n]+v1[i+j*m+kp*m*n])+B2_m[ind];
                    B3=alpha*(w1[ine+j*m+k*m*n]+w1[i+jn*m+k*m*n]+w1[ip+j*m+k*m*n]+w1[i+jp*m+k*m*n]+w1[i+j*m+kn*m*n]+w1[i+j*m+kp*m*n])+B3_m[ind];
                    
                    u1[ind]=(1-w)*u1[ind]+A11[ind]*(B1-A12[ind]*v1[ind]-A13[ind]*w1[ind]);
                    v1[ind]=(1-w)*v1[ind]+A22[ind]*(B2-A12[ind]*u1[ind]-A23[ind]*w1[ind]);
                    w1[ind]=(1-w)*w1[ind]+A33[ind]*(B3-A13[ind]*u1[ind]-A23[ind]*v1[ind]);
                    
                    if (u1[ind]!=u1[ind])
                        u1[ind]=0;
                    if (v1[ind]!=v1[ind])
                        v1[ind]=0;
                    if (w1[ind]!=w1[ind])
                        w1[ind]=0;
                    //end of pixel loop
                }
            }
        }
        
    }//inner fixed point
    
    for(int i=0;i<sz;i++){
       u1[i]+=u0[i]; v1[i]+=v0[i]; w1[i]+=w0[i];
    }
    
    delete B1_m; delete B2_m; delete B3_m;
    delete A11; delete A12; delete A13;
    delete A14; delete A22; delete A23;
    delete A24; delete A33; delete A34;
    


    //delete u0; delete v0; delete w0;
    
}



void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){
	timeval time1,time2;
	float* Sx=(float*)mxGetData(prhs[0]);
    float* Sy=(float*)mxGetData(prhs[1]);
    float* Sz=(float*)mxGetData(prhs[2]);
    float* S=(float*)mxGetData(prhs[3]);
    
    float* u0=(float*)mxGetData(prhs[4]);
    float* v0=(float*)mxGetData(prhs[5]);
    float* w0=(float*)mxGetData(prhs[6]);
    
    float alpha=(float)mxGetScalar(prhs[7]);

	const mwSize* dims1=mxGetDimensions(prhs[0]);
	int m=dims1[0]; int n=dims1[1]; int o=dims1[2];
    
    printf("Sizes: %d, %d, %d, alpha: %f\n",m,n,o,alpha);
	
    plhs[0]=mxCreateNumericArray(3,dims1,mxSINGLE_CLASS,mxREAL);
    plhs[1]=mxCreateNumericArray(3,dims1,mxSINGLE_CLASS,mxREAL);
    plhs[2]=mxCreateNumericArray(3,dims1,mxSINGLE_CLASS,mxREAL);

    float* u1=(float*)mxGetData(plhs[0]);
    float* v1=(float*)mxGetData(plhs[1]);
    float* w1=(float*)mxGetData(plhs[2]);

	
	gettimeofday(&time1, NULL);

    SORsolver(u1,v1,w1,Sx,Sy,Sz,S,u0,v0,w0,alpha,m,n,o);

	
	gettimeofday(&time2, NULL);
	float timeSOR=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
	printf("Computation time for SOR-solver: %f secs.\n",timeSOR);
    
}

