#ifndef __JACOBI_GPU_3_up_H
#define __JACOBI_GPU_3_up_H


__global__ void jacobi_gpu_3_up(int N, double *array_in, double *array_out, double *fmatrix, double * LoIn);


#endif
