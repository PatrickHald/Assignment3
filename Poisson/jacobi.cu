double jacobi(int para, int N, double **array_in, double **array_out, int max_iter, double **fmatrix) {

/* array_in and array_out are (N+2)*(N+2) */


    int i, j, k=0;
    double **tmp;
    double num_iter=0.0;



//// OMP VERSION IF PARA==0
    if(para==0){
    while (k < max_iter) {
	num_iter++;
	#pragma omp parallel for default(none) private(i,j) shared(k,array_out,array_in,max_iter,N,fmatrix,tmp)
    	for(i = 1; i <= N; i++){
	    for(j = 1; j <= N; j++){
	    	array_out[i][j] = 0.25*(array_in[i-1][j]+array_in[i+1][j]+array_in[i] 			[j-1]+array_in[i][j+1]+fmatrix[i][j]);
		}
	   }
	//tmp=array_in;
	array_in = array_out;
	//array_out = tmp;
	k++;

    }

}

return num_iter;


}
