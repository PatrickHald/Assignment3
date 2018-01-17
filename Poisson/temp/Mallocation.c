
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

double **malloc_matrix(int m, int n){
    int i;

    if (m <= 0 || n <= 0)
	return NULL;

    double **A = malloc(m * sizeof(double *));
    if (A == NULL)
	return NULL;

    A[0] = malloc(m*n*sizeof(double));
    if (A[0] == NULL) {
	free(A);
	return NULL;
    }
    for (i = 1; i < m; i++)
	A[i] = A[0] + i * n;

    return A;
}
