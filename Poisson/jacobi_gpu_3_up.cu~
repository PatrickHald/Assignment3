#include <helper_cuda.h>
#include <stdlib.h>

__global__ void jacobi_gpu_3_up(int N, double *array_in, double *array_out, double *fmatrix, double *lowIn) {

    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(i < ((N+2)/2)-2 && j < N){ 
array_out[(i+1)*((N+2))+j+1] = 0.25*(array_in[(i)*((N+2))+j+1]+array_in[(i+2)*((N+2))+j+1]+array_in[(i+1)*((N+2))+j+2]+array_in[(i+1)*((N+2))+j]+fmatrix[(i+1)*((N+2))+j+1]);
	}

	else if(i == ((N+2)/2)-2 && j <N){
array_out[(i+1)*((N+2))+j+1] = 0.25*(array_in[(i)*((N+2))+j+1]+lowIn[j+1]+array_in[(i+1)*((N+2))+j+2]+array_in[(i+1)*((N+2))+j]+fmatrix[(i+1)*((N+2))+j+1]);
	}

}


