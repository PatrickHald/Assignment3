#include <helper_cuda.h>
#include <stdlib.h>

__global__ void jacobi_gpu_3_lo(int N, double *array_in, double *array_out, double *fmatrix, double * hiIn) {

    int offset = (N+1)/2;
    int i = threadIdx.x + blockIdx.x * blockDim.x+1+offset;
    int j = threadIdx.y + blockIdx.y * blockDim.y+1+offset;

    if(i < N+1 && j > offset+1){ 

array_out[i*(N+2)+j] = 0.25*(array_in[(i-1)*(N+2)+j]+array_in[(i+1)*(N+2)+j]+array_in[i*(N+2)+j-1]+array_in[i*(N+2)+j+1]+fmatrix[i*(N+2)+j]);
	}

	else if(i < N+1 && j==offset+1){
array_out[i*((N+2)/2)+j] = 0.25*(array_in[(i-1)*((N+2)/2)+j]+array_in[(i+1)*((N+2)/2)+j]+hiIn[i*((N+2)/2)+j-1]+array_in[i*((N+2)/2)+j+1]+fmatrix[i*((N+2)/2)+j]);
	}
}


