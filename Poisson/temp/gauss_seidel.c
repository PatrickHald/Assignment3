void
gauss_seidel(int N, double **array, int max_iter, double threshold) {
/* array is (N+2)*(N+2) */
    double 	delta = 2/(N+1);
    double 	fdelta2 = 200*delta*delta;
    double 	i_inf=0.5*(N+1), i_sup=2*(N+1)/3, j_inf=(N+1)/6, j_sup=(N+1)/3;
    double	tmp_old;
    double	threshold2 = threshold * threshold;
    double 	d = threshold2 +1;
    int 	i, j, k=0;

    while ( d > threshold2 &&  k < max_iter ) {
	d=0.0;
    	for(i = 1; i <= N; i++){ 
	
	    for(j = 1; j <= N; j++){ 
		tmp_old = array[i][j];
		if(i>=i_inf && i <= i_sup && j>=j_inf && j<=j_sup){
	    	array[i][j] = 0.25*(array[i-1][j]+array[i+1][j]+array[i][j-1]+array[i][j+1]+fdelta2);
			}
		else{
		array[i][j] = 0.25*(array[i-1][j]+array[i+1][j]+array[i][j-1]+array[i][j+1]);
			}
		d += (tmp_old - array[i][j])*(tmp_old - array[i][j]);
		}

	k++;
	}
    }
}
