void
jacobi(int N, double **array_in, double **array_out, int max_iter, double threshold) {
/* array_in and array_out are (N+2)*(N+2) */

    double 	delta = 2/(N+1);
    double 	fdelta2 = 200*delta*delta;
    double 	i_inf=0.5*(N+1), i_sup=2*(N+1)/3, j_inf=(N+1)/6, j_sup=(N+1)/3;
    double	threshold2 = threshold * threshold;
    double 	d = threshold2 +1;
    int 	i, j, k=0;

    while (d > threshold2 &&  k < max_iter ) {
	d = 0.0;
    	for(i = 1; i <= N; i++){ 
	
	    for(j = 1; j <= N; j++){ 
		if(i>=i_inf && i <= i_sup && j>=j_inf && j<=j_sup){
	    	array_out[i][j] = 0.25*(array_in[i-1][j]+array_in[i+1][j]+array_in[i][j-1]+array_in[i][j+1]+fdelta2);
		     }
		else{
		array_out[i][j] = 0.25*(array_in[i-1][j]+array_in[i+1][j]+array_in[i][j-1]+array_in[i][j+1]);
		     };
		d += (array_in[i][j] - array_out[i][j])*(array_in[i][j] - array_out[i][j]);
		}
	   }
	array_in = array_out;
	k++;

    }
}
