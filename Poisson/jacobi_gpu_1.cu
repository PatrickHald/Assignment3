#include <helper_cuda.h>
#include <stdlib.h>

__global__ void jacobi_gpu_1(int N, double *array_in, double *array_out, double *fmatrix) {


//// SEQUENTIAL GPU KERNEL
    	for(int i = 1; i <= N; i++){ 
	    for(int j = 1; j <= N; j++){
	    	array_out[i*(N+2)+j] = 0.25*(array_in[(i-1)*(N+2)+j]+array_in[(i+1)*(N+2)+j]+array_in[i*(N+2)+j-1]+array_in[i*(N+2)+j+1]+fmatrix[i*(N+2)+j]);
		}
	   }
}
