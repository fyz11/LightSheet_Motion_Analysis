#include "mex.h"
#include "math.h"

void pointsor3d(float *A11,float *A12,float *A13,float *A14,float *A22,float *A23,float *A24,float *A33,float *A34,float *um,float *vm,float *wm,double iterations1,double alpha1,float *u1,float *v1,float *w1,double omega1, double mr, double nc, double os)
{

	int i,j,k;
    int this;
	int iter;
    int m,n,o;
    float alpha,iterations,omega;
    iterations=iterations1;
    alpha=alpha1;
    omega=omega1;

    m=(mr);
    n=(nc);
    o=(os);


	float nu1,nu2,nu3,nu4,nu5,nu6;
	float nv1,nv2,nv3,nv4,nv5,nv6;
    float nw1,nw2,nw3,nw4,nw5,nw6;
    
    float a11,a22,a33,B1,B2,B3;
    float w=omega;
    float u_upd0,v_upd0,w_upd0;
    float update1;
    
    for(iter=0;iter<iterations;iter++)
 {
        
        update1=0;
        for(k=0;k<(o-0);k++)
        {
            for(j=0;j<(n-0);j++)
            {
                for(i=0;i<(m-0);i++) 
                {
                    //for(i=(iter+k+j)%2;i<m;i+=2)//bipartite-graph

                this=i+j*m+k*m*n;
                		
                    if(i>0){
                    nu1=u1[(i-1)+j*m+k*m*n]+um[(i-1)+j*m+k*m*n];
                    nv1=v1[(i-1)+j*m+k*m*n]+vm[(i-1)+j*m+k*m*n];
                    nw1=w1[(i-1)+j*m+k*m*n]+wm[(i-1)+j*m+k*m*n];
                    }
                    else{
                    nu1=u1[this]+um[this];
                    nv1=v1[this]+vm[this];
                    nw1=w1[this]+wm[this];
                    }
                    if(j>0){
                    nu2=u1[i+(j-1)*m+k*m*n]+um[i+(j-1)*m+k*m*n];
                    nv2=v1[i+(j-1)*m+k*m*n]+vm[i+(j-1)*m+k*m*n];
                    nw2=w1[i+(j-1)*m+k*m*n]+wm[i+(j-1)*m+k*m*n];
                    }
                    else{
                    nu2=u1[this]+um[this];
                    nv2=v1[this]+vm[this];
                    nw2=w1[this]+wm[this];
                    }
                    if(i<(m-1)){ //shouldnt that be smaller m not m-1 ???
                    nu3=u1[(i+1)+j*m+k*m*n]+um[(i+1)+j*m+k*m*n];
                    nv3=v1[(i+1)+j*m+k*m*n]+vm[(i+1)+j*m+k*m*n];
                    nw3=w1[(i+1)+j*m+k*m*n]+wm[(i+1)+j*m+k*m*n];
                    }
                    else{
                    nu3=u1[this]+um[this];
                    nv3=v1[this]+vm[this];
                    nw3=w1[this]+wm[this];
                    }
                    if(j<(n-1)){
                    nu4=u1[i+(j+1)*m+k*m*n]+um[i+(j+1)*m+k*m*n];
                    nv4=v1[i+(j+1)*m+k*m*n]+vm[i+(j+1)*m+k*m*n];
                    nw4=w1[i+(j+1)*m+k*m*n]+wm[i+(j+1)*m+k*m*n];
                    }
                    else{
                    nu4=u1[this]+um[this];
                    nv4=v1[this]+vm[this];
                    nw4=w1[this]+wm[this];
                    }
                    if(k>0){
                    nu5=u1[i+j*m+(k-1)*m*n]+um[i+j*m+(k-1)*m*n];
                    nv5=v1[i+j*m+(k-1)*m*n]+vm[i+j*m+(k-1)*m*n];
                    nw5=w1[i+j*m+(k-1)*m*n]+wm[i+j*m+(k-1)*m*n];
                    }
                    else{
                    nu5=u1[this]+um[this];
                    nv5=v1[this]+vm[this];
                    nw5=w1[this]+wm[this];
                    }
                    if(k<(o-1)){
                    nu6=u1[i+j*m+(k+1)*m*n]+um[i+j*m+(k+1)*m*n];
                    nv6=v1[i+j*m+(k+1)*m*n]+vm[i+j*m+(k+1)*m*n];
                    nw6=w1[i+j*m+(k+1)*m*n]+wm[i+j*m+(k+1)*m*n];
                    }
                    else{
                    nu6=u1[this]+um[this];
                    nv6=v1[this]+vm[this];
                    nw6=w1[this]+wm[this];
                    }


                
               
				//SOR EQUATIONS
                //A11=(6+1/alpha*J11);
                a11=(6*alpha)+A11[this];
                a22=(6*alpha)+A22[this];
                a33=(6*alpha)+A33[this];
                //A22=(6+1/alpha*J22);
                //A33=(6+1/alpha*J33);
                B1=alpha*(nu1+nu2+nu3+nu4+nu5+nu6-6*um[this]);
                B2=alpha*(nv1+nv2+nv3+nv4+nv5+nv6-6*vm[this]);
                B3=alpha*(nw1+nw2+nw3+nw4+nw5+nw6-6*wm[this]);
                
                u_upd0=u1[this];
                v_upd0=v1[this];
                w_upd0=w1[this];
                
                u1[this]=(1-w)*u1[this]+w/a11*(B1-A12[this]*v1[this]-A13[this]*w1[this]-A14[this]);
                v1[this]=(1-w)*v1[this]+w/a22*(B2-A12[this]*u1[this]-A23[this]*w1[this]-A24[this]);
                w1[this]=(1-w)*w1[this]+w/a33*(B3-A13[this]*u1[this]-A23[this]*v1[this]-A34[this]);
				//u1[this]=(1-w)*u1[this]+w*(B1-1/alpha*(J12*v1[this]+J13*w1[this]+J14))/A11;
				//v1[this]=(1-w)*v1[this]+w*(B2-1/alpha*(J12*u1[this]+J23*w1[this]+J24))/A22;
				//w1[this]=(1-w)*w1[this]+w*(B3-1/alpha*(J13*u1[this]+J23*v1[this]+J34))/A33;
                
                update1=sqrt(pow(u1[this]-u_upd0, 2)+pow(v1[this]-v_upd0, 2)+pow(w1[this]-w_upd0, 2))+update1;
                                
                if (u1[this]!=u1[this])
                    u1[this]=0;
                if (v1[this]!=v1[this])
                    v1[this]=0;
                if (w1[this]!=w1[this])
                    w1[this]=0;
				//end of pixel loop
                }
        if(update1/(m*n*o)<0.001){
         //   printf("Iterations %d finished, mean update: %f\n", iter, update1/(m*n*o));
            return;
        }


    }//inner fixed point

        }
//printf ("Max iterations %d reached, mean update: %f\n",iter,update1/(m*n*o));

}

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{


  double iterations1,alpha1;
  double omega1;
  double mrows,ncols,oslices;
  

  if(nrhs!=18) 
    mexErrMsgTxt("18 inputs required.");
  if(nlhs!=3) 
    mexErrMsgTxt("3 outputs required.");
  
  

  /*  create a pointer to the input scalars */
  iterations1 = mxGetScalar(prhs[12]);
  alpha1 = mxGetScalar(prhs[13]);  
  omega1 = mxGetScalar(prhs[14]);
  mrows = mxGetScalar(prhs[15]);
  ncols = mxGetScalar(prhs[16]);
  oslices = mxGetScalar(prhs[17]);
  
  
  /*  get the dimensions of the matrix input y */
  //mrows = mxGetM(prhs[0]);
  //ncols = mxGetN(prhs[0]);
 
  
  /*  set the output pointer to the output matrix */
  //plhs[0] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
  plhs[0]=mxCreateNumericArray(3,mxGetDimensions(prhs[0]),mxGetClassID(prhs[0]),mxREAL);
  plhs[1]=mxCreateNumericArray(3,mxGetDimensions(prhs[0]),mxGetClassID(prhs[0]),mxREAL);
  plhs[2]=mxCreateNumericArray(3,mxGetDimensions(prhs[0]),mxGetClassID(prhs[0]),mxREAL);
  
  
  
  pointsor3d((float *)mxGetData(prhs[0]),(float *)mxGetData(prhs[1]),
          (float *)mxGetData(prhs[2]),(float *)mxGetData(prhs[3]),
          (float *)mxGetData(prhs[4]),(float *)mxGetData(prhs[5]),
          (float *)mxGetData(prhs[6]),(float *)mxGetData(prhs[7]),
          (float *)mxGetData(prhs[8]),(float *)mxGetData(prhs[9]),
          (float *)mxGetData(prhs[10]),(float *)mxGetData(prhs[11]),
          iterations1,alpha1,(float *)mxGetData(plhs[0]),
          (float *)mxGetData(plhs[1]),(float *)mxGetData(plhs[2]),
          omega1,mrows,ncols,oslices);
  
}
