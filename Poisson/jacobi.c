double jacobi(int para, int N, double **array_in, double **array_out, int max_iter, double threshold, double **fmatrix) {

/* array_in and array_out are (N+2)*(N+2) */


    int 	i, j, k=0;
    double 	**tmp;



//// OMP VERSION IF PARA==0
    if(para==0){
    while (k < max_iter) {
	//d = 0.0;
	#pragma omp parallel for default(none) private(i,j) shared(k,array_out,array_in,max_iter,N,fmatrix,tmp)
    	for(i = 1; i <= N; i++){ 
	    for(j = 1; j <= N; j++){
	    	array_out[i][j] = 0.25*(array_in[i-1][j]+array_in[i+1][j]+array_in[i] 			[j-1]+array_in[i][j+1]+fmatrix[i][j]);
		//d += (array_in[i][j] - array_out[i][j])*(array_in[i][j] - array_out[i] 			[j]);
		}
	   }
	tmp=array_in;
	array_in = array_out;
	array_out = tmp;
	k++;

    }
return 0.0;
}





    return k;
}
