#include <helper_cuda.h>
#include <omp.h>
#include "jacobi.h"
#include "jacobi_gpu_1.h"
#include "jacobi_gpu_2.h"
#include "jacobi_gpu_3_up.h"
#include "jacobi_gpu_3_lo.h"
#include "Mallocation.h"
extern "C" {
#include <stdlib.h>
}

int main(int argc, char *argv[]){

///// PRELOAD AND INITIALIZATION /////
    int method = atoi(argv[1]);
    int N = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    double num_iter;
    double * matrixOutFinal1d;
    double ** matrixOutFinal2d;






///// JACOBI METHOD/////


       


	///OpenMP METHOD
	if(method==0){
   	// Allocate and initialize the NxN matrix for the room with Dirichlet conditions
   	double ** matrixOut = malloc_matrix(N+2,N+2);
    	double ** matrixIn = malloc_matrix(N+2,N+2);
    	init_data(N+2, N+2, matrixOut);
    	init_data(N+2, N+2, matrixIn);

    	for(int i=0; i<N+2; ++i){
    	matrixOut[i][N+1]=20.0;
    	matrixOut[N+1][i]=20.0;
    	matrixOut[0][i]=20.0;
	matrixOut[i][0]=0.0;
	matrixIn[i][N+1]=20.0;
    	matrixIn[N+1][i]=20.0;
    	matrixIn[0][i]=20.0;
	matrixIn[i][0]=0.0;
    	}

  	  // Create the matrix for values of the function
    	double ** fmatrix = malloc_matrix(N+2,N+2);
    	init_f_2d(N, fmatrix);	

    	num_iter = jacobi(method, N, matrixIn, matrixOut, max_iter, fmatrix);
	matrixOutFinal2d=matrixIn;

}





	///GPU SEQUENTIAL METHOD
	else if(method!=0){

	///// IF USING GPU VERSION, WE MUST ALLOCATE 1D POINTERS


    // Allocate and initialize matrixOut and matrixIn
    double * matrixOut;
    double * matrixIn;
    matrixOut = (double *)malloc( (N+2) * (N+2) * sizeof(double));
    matrixIn = (double *)malloc( (N+2) * (N+2) * sizeof(double));
    for(int i=0; i<N+2; ++i){
	for(int j=0; j<N+2; ++j){
		matrixOut[i*(N+1)+j]=0;
		matrixIn[i*(N+1)+j]=0;
	}
}

    for(int i=0; i<N+2; ++i){
	matrixOut[i]=20.0;
	matrixOut[(N+1)*(N+2)+i]=20.0;
	matrixOut[i*(N+2)+(N+1)]=20.0;
	matrixOut[i*(N+2)]=0.0;

       	matrixIn[i]=20.0;
	matrixIn[(N+1)*(N+2)+i]=20.0;
	matrixIn[i*(N+2)+(N+1)]=20.0;
	matrixIn[i*(N+2)]=0.0;
    }

   	 // Create the matrix for values of the function
   	 double * fmatrix;
         fmatrix = (double *)malloc( (N+2) * (N+2) * sizeof(double));
	 for(int i=0; i<N+2; ++i){
		for(int j=0; j<N+2; ++j){
			fmatrix[i*(N+2)+j]=0;
		}
	}

    	 init_f(N, fmatrix);






	 // Allocate mem
	 double *d_temp;
   	 double *d_In;
	 double *d_Out;
	 double *d_f;
   	 double size = (N+2) * (N+2) * sizeof(double); 
	 cudaMalloc((void**)&d_temp, size); 
   	 cudaMalloc((void**)&d_In, size); 
  	 cudaMalloc((void**)&d_Out, size); 
	 cudaMalloc((void**)&d_f, size); 

    	// Transfer data from host to device 
    	cudaMemcpy(d_In, matrixIn, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_Out, matrixOut, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_f, fmatrix, size, cudaMemcpyHostToDevice); 



//using 1 thread in total
if(method==1){

	int k=0;
	double time0 = omp_get_wtime();
	while(k<max_iter){
	jacobi_gpu_1<<<1,1>>>(N, d_In, d_Out, d_f);
	cudaDeviceSynchronize();
        d_temp=d_In;
	d_In=d_Out;
	d_Out=d_temp;
	k++;	
	}
	double time1 = omp_get_wtime();
	printf("Time for gpu_1 method: %f\n", time1-time0); 
d_Out=d_In;
}

//using 1 thread per grid point NxN with NO MEMORY SHARING
if(method==2){

	int K = 32;
        int Gx = ceil((double)N / K);
        int Gy = ceil((double)N / K);
        dim3 dimGrid(Gx,Gy,1); // number of blocks 2D
        dim3 dimBlock(K,K,1); // number of threads per block 2D

	int count=0;
	double time2 = omp_get_wtime();
	while(count<max_iter){
	jacobi_gpu_2<<<dimGrid,dimBlock>>>(N, d_In, d_Out, d_f);
	cudaDeviceSynchronize();
	d_temp=d_In;
	d_In=d_Out;
	d_Out=d_temp;
	count++;
	}
	double time3 = omp_get_wtime();
	printf("Time for gpu_2 method: %f\n", time3-time2); 
d_Out=d_In;
}




//using two GPUs simultaneously
if(method==3){

	   // Allocate and initialize matrixOut and matrixIn for both upper and lower
    double * matrixOutUp;
    double * matrixInUp;
    double * matrixOutLo;
    double * matrixInLo;

    matrixOutUp = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
    matrixInUp = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
    matrixOutLo = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
    matrixInLo = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
    for(int i=0; i<(N+2)/2; ++i){
	for(int j=0; j<(N+2); ++j){
		matrixOutUp[i*(N+2)+j]=matrixOut[i*((N+2))+j];
		matrixInUp[i*(N+2)+j]=matrixIn[i*((N+2))+j];
		matrixOutLo[i*(N+2)+j]=matrixOut[i*((N+2))+j+(((N+2))*((N+2)/2))];
		matrixInLo[i*(N+2)+j]=matrixIn[i*((N+2))+j+(((N+2))*((N+2)/2))];
	}
}



   	 // Create the matrix for values of the function
   	 double * fmatrixUp;
	 double * fmatrixLo;
         fmatrixUp = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
         fmatrixLo = (double *)malloc( ((N+2)) * ((N+2)/2) * sizeof(double));
	 for(int i=0; i<(N+2)/2; ++i){
		for(int j=0; j<(N+2); ++j){
			fmatrixUp[i*((N+2))+j]=fmatrix[i*((N+2))+j];
			fmatrixLo[i*((N+2))+j]=fmatrix[i*((N+2))+j+(((N+2))*((N+2)/2))];
		}
	}

		 // Allocate mem
	 double *d0_temp;
   	 double *d0_In;
	 double *d0_Out;
	 double *d1_temp;
         double *d1_In;
	 double *d1_Out;
	 double *d0_f;
	 double *d1_f;
   	 double size = (N+2) * (N+2) * sizeof(double); 
	 double size2 = ((N+2)/2) * ((N+2))* sizeof(double); 
	 cudaMalloc((void**)&d0_temp, size2); 
   	 cudaMalloc((void**)&d0_In, size2); 
  	 cudaMalloc((void**)&d0_Out, size2); 
	 cudaMalloc((void**)&d0_f, size2); 
	 cudaMalloc((void**)&d1_temp, size2); 
	 cudaMalloc((void**)&d1_In, size2); 
  	 cudaMalloc((void**)&d1_Out, size2); 
	 cudaMalloc((void**)&d1_f, size2); 


    	// Transfer data from host to device 
    	cudaMemcpy(d0_In, matrixInUp, size2, cudaMemcpyHostToDevice); 
	cudaMemcpy(d0_Out, matrixOutUp, size2, cudaMemcpyHostToDevice); 
	cudaMemcpy(d0_f, fmatrixUp, size2, cudaMemcpyHostToDevice); 
	cudaMemcpy(d1_In, matrixInLo, size2, cudaMemcpyHostToDevice); 
	cudaMemcpy(d1_Out, matrixOutLo, size2, cudaMemcpyHostToDevice); 
	cudaMemcpy(d1_f, fmatrixLo, size2, cudaMemcpyHostToDevice);


	int K = 32;
        int Gx = ceil((double)N / K);
        int Gy = ceil((double)N / (2*K));
        dim3 dimGrid(Gx,Gy,1); // number of blocks 2D
        dim3 dimBlock(K,K,1); // number of threads per block 2D

	int count=0;
	cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1,0);
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0,0);
	double time4 = omp_get_wtime();
	while(count<max_iter){
	cudaSetDevice(0);
	jacobi_gpu_3_up<<<dimGrid,dimBlock>>>(N, d0_In, d0_Out, d0_f,d1_In);
	cudaSetDevice(1);
	jacobi_gpu_3_lo<<<dimGrid,dimBlock>>>(N, d1_In, d1_Out, d1_f,d0_In);
	cudaDeviceSynchronize();
	cudaSetDevice(0);
	cudaDeviceSynchronize();

	d0_temp=d0_In;
	d0_In=d0_Out;
	d0_Out=d0_temp;

	d1_temp=d1_In;
	d1_In=d1_Out;
	d1_Out=d1_temp;
	count++;
	}
	double time5 = omp_get_wtime();
	printf("Time for gpu_3 method: %f\n", time5-time4);
	d0_Out=d0_In;
	d1_Out=d1_In;
	cudaMemcpy(matrixOutUp, d0_Out, size2, cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixOutLo, d1_Out, size2, cudaMemcpyDeviceToHost);
	cudaFree(d0_temp);
   	cudaFree(d0_In);
	cudaFree(d0_Out); 
	cudaFree(d0_f); 
	cudaFree(d1_temp);
	cudaFree(d1_In);
	cudaFree(d1_Out); 
	cudaFree(d1_f);
	for(int i=0; i<N+2; ++i){
	for(int j=0; j<N+2; ++j){
		if(i<(N+2)/2){
		matrixOut[i*(N+2)+j]=matrixOutUp[i*((N+2))+j];
		}
		else if(i>=(N+2)/2){
		matrixOut[i*(N+2)+j]=matrixOutLo[i*((N+2))+j-(N+2)*(N+2)/2];
		}	
	}
	}
matrixOutFinal1d=matrixOut;

}


if(method==1 || method==2){
        // Transfer data from device to host 
        cudaMemcpy(matrixOut, d_Out, size, cudaMemcpyDeviceToHost);
   	cudaFree(d_In);
	cudaFree(d_Out); 
	cudaFree(d_f); 
	matrixOutFinal1d=matrixOut;
	}

}



 ////CALCULATE MATRIX SUMS TO VERIFY RESULTS
	double matSum=0.0;

	if(method==0){
	for(int i=0; i<N+2; ++i){
	for(int j=0; j<N+2; ++j){
		matSum+=matrixOutFinal2d[i][j];
	}
	}
 //writing matrix output
for(int i=0; i<N+2; i++){
    for(int j=0; j<N+2; j++){
	printf("%f \t", matrixOutFinal2d[i][j]);}
    printf("\n");}  
}



	else{
	for(int i=0; i<N+2; ++i){
	for(int j=0; j<N+2; ++j){
		matSum+=matrixOutFinal1d[i*(N+2)+j];
	}
	}
 //writing matrix output
for(int i=0; i<N+2; i++){
    for(int j=0; j<N+2; j++){
	printf("%f \t", matrixOutFinal1d[i*(N+2)+j]);}
    printf("\n");} 
}


printf("%f\n",matSum);

return 0;
}

