#ifndef __JACOBI_GPU_3_lo_H
#define __JACOBI_GPU_3_lo_H


__global__ void jacobi_gpu_3_lo(int N, double *array_in, double *array_out, double *fmatrix, double * HiIn);


#endif
