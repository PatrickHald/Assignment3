int
gauss_seidel(int N, double **array, int max_iter, double threshold, double **fmatrix) {
/* array is (N+2)*(N+2) */

    double	tmp_old;
    double	threshold2 = threshold * threshold;
    double 	d = threshold2 +1;
    int 	i, j, k=0;

    while ( d > threshold2 /*&&  k < max_iter */) {
	d=0.0;
    	for(i = 1; i <= N; i++){ 
	
	    for(j = 1; j <= N; j++){ 
		tmp_old = array[i][j];
		array[i][j] = 0.25*(array[i-1][j]+array[i+1][j]+array[i][j-1]+array[i][j+1]+fmatrix[i][j]);
		d += (tmp_old - array[i][j])*(tmp_old - array[i][j]);
		}
	}

	k++;
	
    }
    return k;
}
