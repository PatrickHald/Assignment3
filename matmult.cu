#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <omp.h>


#include <cublas_v2.h>
extern "C" {
#include <cblas.h>

#define FOR_i_TO_m for (i = 0; i < m; i++)
#define FOR_j_TO_n for (j = 0; j < n; j++)
#define FOR_l_TO_k for (l = 0; l < k; l++)

#define RESET_C FOR_i_TO_m FOR_j_TO_n C[i * n + j] = 0;

#define MIN(a,b) ((a) < (b) ? a : b)

#define SIZE_A m*k*sizeof(double)
#define SIZE_B k*n*sizeof(double)
#define SIZE_C m*n*sizeof(double)

//choice of the neighbor for matmult_gpu3() 1 = right || 2 = below
#define NEIGHBOR 2

//number of elements per thread matmult_gpu4()
#define T 12

//block size for matmult_gpu5()
#define BLOCK_SIZE 16

//choose if you want the times to be printed
#define PRINT_TIMES 0


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
void matmult_gpulib(int m,int n,int k,double *A,double *B,double *C);

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
    double time0 = omp_get_wtime();
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,A,k,B,n,0,C,n);
    double time1 = omp_get_wtime();

    if (PRINT_TIMES == 1)
	//printf("time to transfer HtoD = %3.6f seconds\n", time1 - time0);
	printf("%d \t %3.6f\n", m, time1 - time0);

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

__global__ void gpu1(int m,int n,int k,double *A,double *B,double *C)
{
    int i, j, l;
    
    RESET_C

    FOR_i_TO_m
        FOR_l_TO_k
            FOR_j_TO_n
                atomicAdd(&C[i * n + j] , A[i * k + l] * B[l * n + j]);
}

void matmult_gpu1(int m,int n,int k,double *A,double *B,double *C)
{
 // The GPU uses only 1 thread

    double *d_A, *d_B, *d_C;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, SIZE_A);
    cudaMalloc((void**)&d_B, SIZE_B);
    cudaMalloc((void**)&d_C, SIZE_C);

    double time0 = omp_get_wtime();

    // Transfer data from host to device 
    cudaMemcpy(d_A, A, SIZE_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, SIZE_B, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, SIZE_C, cudaMemcpyHostToDevice); 

    double time1 = omp_get_wtime();

    // Cuda launch
    gpu1<<<1,1>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    double time2 = omp_get_wtime();

    // Transfer data from device to host 
    cudaMemcpy(C, d_C, SIZE_C, cudaMemcpyDeviceToHost); 

    double time3 = omp_get_wtime();

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    if (PRINT_TIMES == 1){
	//printf("time to transfer HtoD = %3.6f seconds\n", time1 - time0);
	//printf("time to run the program = %3.6f seconds\n", time2 - time1);
	//printf("time to transfer DtoH = %3.6f seconds\n", time3 - time2);
	//printf("total time = %3.6f seconds\n", time3 - time0);
	printf("%d \t %3.6f\n", m, time2 - time1);
    }
}

__global__ void gpu2(int m,int n,int k,double *A,double *B,double *C)
{
    int l;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    double res = 0.0;
    if(i < m && j < n){
	FOR_l_TO_k
	    res += A[i * k + l] * B[l * n + j];
	C[i * n + j] = res;
    }
    
}

void matmult_gpu2(int m,int n,int k,double *A,double *B,double *C)
{
// We use one thread per element of C, which is m * n
   double *d_A, *d_B, *d_C;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, SIZE_A);
    cudaMalloc((void**)&d_B, SIZE_B);
    cudaMalloc((void**)&d_C, SIZE_C);

    double time0 = omp_get_wtime();

    // Transfer data from host to device 
    cudaMemcpy(d_A, A, SIZE_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, SIZE_B, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, SIZE_C, cudaMemcpyHostToDevice); 

    double time1 = omp_get_wtime();

    // Cuda launch
    cudaSetDevice(1);
    int K = 32;
    int Gx = ceil((double)n / K);
    int Gy = ceil((double)m / K);
    dim3 dimGrid(Gx,Gy,1); // number of blocks 2D
    dim3 dimBlock(K,K,1); // number of threads per block 2D
    gpu2<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    double time2 = omp_get_wtime();

    // Transfer data from device to host 
    cudaMemcpy(C, d_C, SIZE_C, cudaMemcpyDeviceToHost); 

    double time3 = omp_get_wtime();

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (PRINT_TIMES == 1){
	/*printf("time to transfer HtoD = %3.6f seconds\n", time1 - time0);
	printf("time to run the program = %3.6f seconds\n", time2 - time1);
	printf("time to transfer DtoH = %3.6f seconds\n", time3 - time2);
	printf("total time = %3.6f seconds\n", time3 - time0);*/
	printf("%d \t %3.6f\n", m, time2 - time1);
    }
 
}

__global__ void gpu3b(int m,int n,int k,double *A,double *B,double *C)
{
// One thread computes 2 elements of C, that are vertical neighbors.
    int l;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    double res1 = 0.0, res2 = 0.0;
    if(2*i + 1 < m && j < n){
	FOR_l_TO_k
	    {
	    res1 += A[2 *i * k + l] * B[l * n + j];
	    res2 += A[(2*i+1) * k + l] * B[l * n + j];
	    }
	C[2 * i * n + j] = res1;
	C[(2*i+1) * n + j] = res2;
    }
    else if(2*i + 1 == m && j < n){
	FOR_l_TO_k
	    res1 += A[2 *i * k + l] * B[l * n + j];
	C[2 * i * n + j] = res1;
    }
    
}

__global__ void gpu3r(int m,int n,int k,double *A,double *B,double *C)
{
// One thread computes 2 elements of C, that are horizontal neighbors.
// This kernel was used to compare timings of vertical and horizontal choices, but is the less efficient so gpu3b is preferred.
    int l;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    double res1 = 0.0, res2 = 0.0;
    if(i < m && 2*j + 1 < n){
	FOR_l_TO_k
	    {
	    res1 += A[i * k + l] * B[l * n + 2 * j];
	    res2 += A[i * k + l] * B[l * n + 2*j+1];
	    }
	C[i * n + 2*j] = res1;
	C[i * n + 2*j+1] = res2;
    }
    else if(i < m && 2*j+1 == n){
	FOR_l_TO_k
	    res1 += A[i * k + l] * B[l * n + 2*j];
	C[i * n + 2*j] = res1;
    }
    
}
 
void matmult_gpu3(int m,int n,int k,double *A,double *B,double *C)
{
// Each thread computes 2 elements of C
   double *d_A, *d_B, *d_C;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, SIZE_A);
    cudaMalloc((void**)&d_B, SIZE_B);
    cudaMalloc((void**)&d_C, SIZE_C);

    double time0 = omp_get_wtime();

    // Transfer data from host to device 
    cudaMemcpy(d_A, A, SIZE_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, SIZE_B, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, SIZE_C, cudaMemcpyHostToDevice); 

    double time1 = omp_get_wtime();

    if(NEIGHBOR==1)
    {

    // Cuda launch
    cudaSetDevice(1);
    int K = 32;
    int Gx = ceil((double)n / K);
    int Gy = ceil((double)m / K);
    dim3 dimGrid(ceil((double)Gx/2),Gy,1); // number of blocks 2D
    dim3 dimBlock(K,K,1); // number of threads per block 2D
    gpu3r<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    
    cudaDeviceSynchronize();
    }
    else if(NEIGHBOR==2)
    {
    // Cuda launch
    cudaSetDevice(1);
    int K = 32;
    int Gx = ceil((double)n / K);
    int Gy = ceil((double)m / K);
    dim3 dimGrid(Gx,ceil((double)Gy/2),1); // number of blocks 2D
    dim3 dimBlock(K,K,1); // number of threads per block 2D
    gpu3b<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    
    cudaDeviceSynchronize();
    }
    double time2 = omp_get_wtime();

    // Transfer data from device to host 
    cudaMemcpy(C, d_C, SIZE_C, cudaMemcpyDeviceToHost); 

    double time3 = omp_get_wtime();

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (PRINT_TIMES == 1){
	//printf("time to transfer HtoD = %3.6f seconds\n", time1 - time0);
	//printf("time to run the program = %3.6f seconds\n", time2 - time1);
	//printf("time to transfer DtoH = %3.6f seconds\n", time3 - time2);
	//printf("total time = %3.6f seconds\n", time3 - time0);
	printf("%d \t %3.6f\n", m, time2 - time1);
    }

}

__global__ void gpu4(int m,int n,int k,double *A,double *B,double *C)
{
    int l, s;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int S = m - i * T;

    double res[T];
    
    if(T*(i + 1) - 1 < m && j < n){
	for(s = 0; s < T; s++)
	    res[s] = 0.0;	
	FOR_l_TO_k
	    {
	    for(s = 0; s < T; s++)
		res[s] += A[(T *i + s) * k + l] * B[l * n + j];
	    }
	for(s = 0; s < T; s++)
	    C[(T*i+s) * n + j] = res[s];

    }
    else if(T*i < m && j < n){
	for(s = 0; s < S ; s++)
	    res[s] = 0.0;
	FOR_l_TO_k
	    {
	    for(s = 0; s < S ; s++)
	       res[s] += A[(T *i + s) * k + l] * B[l * n + j];
	    }
	for(s = 0; s < S; s++)
	    C[(T*i+s) * n + j] = res[s];
    }
    
}

void matmult_gpu4(int m,int n,int k,double *A,double *B,double *C)
{
// Each thread computes T elements of C
    double *d_A, *d_B, *d_C;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, SIZE_A);
    cudaMalloc((void**)&d_B, SIZE_B);
    cudaMalloc((void**)&d_C, SIZE_C);

    double time0 = omp_get_wtime();

    // Transfer data from host to device 
    cudaMemcpy(d_A, A, SIZE_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, SIZE_B, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, SIZE_C, cudaMemcpyHostToDevice); 

    double time1 = omp_get_wtime();

    // Cuda launch
    cudaSetDevice(1);
    int K = 32;
    int Gx = ceil((double) n / K);
    int Gy = ceil((double) m / K);
    dim3 dimGrid(Gx,ceil((double)Gy/T),1); // number of blocks 2D
    dim3 dimBlock(K,K,1); // number of threads per block 2D
    gpu4<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaDeviceSynchronize();

    double time2 = omp_get_wtime();

    // Transfer data from device to host 
    cudaMemcpy(C, d_C, SIZE_C, cudaMemcpyDeviceToHost); 

    double time3 = omp_get_wtime();

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (PRINT_TIMES == 1){
	//printf("time to transfer HtoD = %3.6f seconds\n", time1 - time0);
	//printf("time to run the program = %3.6f seconds\n", time2 - time1);
	//printf("time to transfer DtoH = %3.6f seconds\n", time3 - time2);
	//printf("total time = %3.6f seconds\n", time3 - time0);
	printf("%d \t %3.6f \n", m, time2 - time1);
 
    }

}

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    double* elements;
} Matrix;

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           double value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size


// Forward declaration of the matrix multiplication kernel
__global__ void gpu5(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void matmult_gpu5(int m,int n,int k,double *A,double *B,double *C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = k; d_A.height = m;
    size_t size = SIZE_A;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = n; d_B.height = k;
    size = SIZE_B;

    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = n; d_C.height = m;
    size = SIZE_C;
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);
    gpu5<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Read C from device memory
    cudaMemcpy(C, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void gpu5(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0.0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C) {
const double alpha = 1.0; const double beta = 0.0;
	

	double *d_A; double *d_B ; 
	double *d_C;
	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaMalloc((void **) &d_A, m*k*sizeof(double));
	cudaMalloc((void **) &d_B, k*n*sizeof(double));
	cudaMalloc((void **) &d_C, m*n*sizeof(double));

	cudaMemcpy(d_A, A, m*k*sizeof(d_A), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, k*n*sizeof(d_B), cudaMemcpyHostToDevice);

	
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

	
	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(C, d_C, m*n*sizeof(C), cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void matmult_gpu6(int m,int n,int k,double *A,double *B,double *C)
{

}


