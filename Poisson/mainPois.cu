#include <helper_cuda.h>
#include "jacobi.h"
#include "jacobi_gpu_seq.h"
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
	matrixOutFinal2d=matrixOut;

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

//REMOVE cudaSetDevice WHEN DONE TESTING
cudaSetDevice(1);
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
	dim3 dimGrid(1,1); // number of blocks
	dim3 dimBlock(1,1); // number of threads per block

	int k=0;
	while(k<max_iter){
	jacobi_gpu_seq<<<dimGrid,dimBlock>>>(N, d_In, d_Out, d_f);
	cudaDeviceSynchronize();
        //d_temp=d_In;
	d_In=d_Out;
	//d_Out=d_temp;
	k++;	
	}
d_Out=d_In;
}

//using 1 thread per grid point NxN with NO MEMORY SHARING
if(method==2){
	dim3 dimGrid(N,1); // number of blocks
	dim3 dimBlock(N,1); // number of threads per block

	int k=0;
	while(k<max_iter){
	jacobi_gpu_seq<<<dimGrid,dimBlock>>>(N, d_In, d_Out, d_f);
	cudaDeviceSynchronize();
	//d_temp=d_In;
	d_In=d_Out;
	//d_Out=d_temp;
	k++;
	}
}

        // Transfer data from device to host 
        cudaMemcpy(matrixOut, d_Out, size, cudaMemcpyDeviceToHost);
   	cudaFree(d_In);
	cudaFree(d_Out); 
	cudaFree(d_f); 
	matrixOutFinal1d=matrixOut;

}

 ////CALCULATE MATRIX SUMS TO VERIFY RESULTS
	double matSum=0.0;

	if(method==0){
	for(int i=0; i<N+2; ++i){
	for(int j=0; j<N+2; ++j){
		matSum+=matrixOutFinal2d[i][j];
	}
	}
 //writing in text file
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
 //writing in text file
for(int i=0; i<N+2; i++){
    for(int j=0; j<N+2; j++){
	printf("%f \t", matrixOutFinal1d[i*(N+2)+j]);}
    printf("\n");}
}



printf("%d %f\n",N, matSum); 


return 0;
}

