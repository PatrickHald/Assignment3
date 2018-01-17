#include "jacobi.h"
#include "gauss_seidel.h"
#include "Mallocation.h"
#include "writepng.h"
#include <stdlib.h>

int main(int argc, char *argv[]){
 int method = atoi(argv[1]);
 int N = atoi(argv[2]);
 int max_iter = atoi(argv[3]);
 double threshold = atof(argv[4]);

// Allocate the NxN matrix for the room with Dirichlet conditions
double ** matrixOut = malloc_matrix(N+2,N+2);
for(int i=0; i<N+2; ++i){
    matrixOut[i][N+1]=20;
    matrixOut[N+1][i]=20;
    matrixOut[0][i]=20;
}

// Either using Jacobi method
if(method==0){
        // Allocate the in-matrix
    double ** matrixIn = malloc_matrix(N+2,N+2);
    for(int i=0; i<N+2; ++i){
            matrixIn[i][N+1]=20;
            matrixIn[N+1][i]=20;
            matrixIn[0][i]=20;
    }
        // Run the jacobi method
    jacobi(N, matrixIn, matrixOut, max_iter, threshold);
}

// Or using Gauss-Seidel method
else if(method==1){
    gauss_seidel(N, matrixOut, max_iter, threshold);
}

/*
// PNG output
writepng("poisPNG", matrixOut, N+2, N+2);

//writing in text file
for(int i=0; i<N+2; i++){
    for(int j=0; j<N+2; j++){
	printf("%f \t", matrixOut[i][j]);}
    printf("\n");}
*/


return 0;
}
