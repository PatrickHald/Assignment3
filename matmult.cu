#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

extern "C" {
#include <cblas.h>

#define FOR_i_TO_m for (i = 0; i < m; i++)
#define FOR_j_TO_n for (j = 0; j < n; j++)
#define FOR_l_TO_k for (l = 0; l < k; l++)

#define RESET_C FOR_i_TO_m FOR_j_TO_n C[i * n + j] = 0;

#define SIZE_A m*k*sizeof(double)
#define SIZE_B k*n*sizeof(double)
#define SIZE_C m*n*sizeof(double)

#define MIN(a,b) ((a) < (b) ? a : b)


void matmult_nat(int m,int n,int k,double *A,double *B,double *C);
void matmult_lib(int m,int n,int k,double *A,double *B,double *C);

void matmult_mkn(int m,int n,int k,double *A,double *B,double *C);
void matmult_mnk(int m,int n,int k,double *A,double *B,double *C);
void matmult_kmn(int m,int n,int k,double *A,double *B,double *C);
void matmult_knm(int m,int n,int k,double *A,double *B,double *C);
void matmult_nmk(int m,int n,int k,double *A,double *B,double *C);
void matmult_nkm(int m,int n,int k,double *A,double *B,double *C);

void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs);

void matmult_gpu1(int m,int n,int k,double *A,double *B,double *C);
void matmult_gpu2(int m,int n,int k,double *A,double *B,double *C);
void matmult_gpu3(int m,int n,int k,double *A,double *B,double *C);
void matmult_gpu4(int m,int n,int k,double *A,double *B,double *C);
void matmult_gpu5(int m,int n,int k,double *A,double *B,double *C);
void matmult_gpu6(int m,int n,int k,double *A,double *B,double *C);

}

void matmult_nat(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_i_TO_m
        FOR_j_TO_n
            FOR_l_TO_k
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_lib(int m,int n,int k,double *A,double *B,double *C)
{
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,A,k,B,n,0,C,n);
}

void matmult_mkn(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_i_TO_m
        FOR_l_TO_k
            FOR_j_TO_n
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_mnk(int m,int n,int k,double *A,double *B,double *C)
{
    matmult_nat(m, n, k, A, B, C);
}

void matmult_kmn(int m,int n,int k,double *A,double *B,double *C)
{    
    int i, j, l;
    
    RESET_C

    FOR_l_TO_k
        FOR_i_TO_m
            FOR_j_TO_n
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}
void matmult_knm(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_l_TO_k
        FOR_j_TO_n
            FOR_i_TO_m
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_nmk(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_j_TO_n
        FOR_i_TO_m
            FOR_l_TO_k
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_nkm(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_j_TO_n
        FOR_l_TO_k
            FOR_i_TO_m
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs)
{
    int I, J, L, i, j, l, limi, limj, liml;

    RESET_C

    for (I = 0; I < m; I+=bs)
	{
	limi = MIN(I+bs,m);
            for (L = 0; L < k; L+=bs)
	    	{
	   	liml = MIN(L+bs,k);
            	    for (J = 0; J < n; J+=bs)
			{
			limj = MIN(J+bs,n);
        		for (i = I; i < limi; i++)
            		    for (l = L; l < liml; l++)
                		for (j = J; j < limj; j++)
                            	    C[i * n + j] += A[i * k + l] * B[l * n + j];
			};
	     	};
		
	};
}

__global__ void gpu1(int m,int n,int k,double *A,double *B,double *C){
    int i, j, l;
    
    RESET_C

    FOR_i_TO_m
        FOR_l_TO_k
            FOR_j_TO_n
                C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void matmult_gpu1(int m,int n,int k,double *A,double *B,double *C)
{
    // The GPU uses only 1 thread

    double *d_A, *d_B, *d_C;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, SIZE_A);
    cudaMalloc((void**)&d_B, SIZE_B);
    cudaMalloc((void**)&d_C, SIZE_C);

    // Transfer data from host to device 
    cudaMemcpy(d_A, A, SIZE_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, SIZE_B, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, SIZE_C, cudaMemcpyHostToDevice); 

    // Cuda launch
    gpu1<<<1,1>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Transfer data from device to host 
    cudaMemcpy(C, d_C, SIZE_C, cudaMemcpyDeviceToHost); 

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
}


void matmult_gpu2(int m,int n,int k,double *A,double *B,double *C)
{

}


void matmult_gpu3(int m,int n,int k,double *A,double *B,double *C)
{

}


void matmult_gpu4(int m,int n,int k,double *A,double *B,double *C)
{

}


void matmult_gpu5(int m,int n,int k,double *A,double *B,double *C)
{

}


void matmult_gpu6(int m,int n,int k,double *A,double *B,double *C)
{

}


