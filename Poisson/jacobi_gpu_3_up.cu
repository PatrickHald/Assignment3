#include <helper_cuda.h>
#include <stdlib.h>

__global__ void jacobi_gpu_3_up(int N, double *array_in, double *array_out, double *fmatrix, double *lowIn) {

    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = threadIdx.y + blockIdx.y * blockDim.y+1;

    if(i < N+1 && j < (N+1)/2){ 

array_out[i*((N+2)/2)+j] = 0.25*(array_in[(i-1)*((N+2)/2)+j]+array_in[(i+1)*((N+2)/2)+j]+array_in[i*((N+2)/2)+j-1]+array_in[i*((N+2)/2)+j+1]+fmatrix[i*((N+2)/2)+j]);
	}

	else if(i < N+1 && j==(N+1)/2){
array_out[i*((N+2)/2)+j] = 0.25*(array_in[(i-1)*((N+2)/2)+j]+array_in[(i+1)*((N+2)/2)+j]+lowIn[i*((N+2)/2)+j-1]+array_in[i*((N+2)/2)+j+1]+fmatrix[i*((N+2)/2)+j]);
	}
}
