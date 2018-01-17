#ifndef __JACOBI_GPU_SEQ_H
#define __JACOBI_GPU_SEQ_H


__global__ void jacobi_gpu_seq(int N, double *array_in, double *array_out, double *fmatrix);


#endif
