#for((N=5000; N<=10000; N += 100)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu4 $N $N $N); done >q04_gpu4_T13_tmp.dat

#awk ' NR % 2 == 1 ' q04_gpu4_T13_tmp.dat > q04_gpu4_T13.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "Time (s)"
	set output "q04_variousT.eps"
	plot "q04_gpu4_T2.dat" w lp title 'T = 2', "q04_gpu4_T4.dat" w lp title 'T = 4', "q04_gpu4_T8.dat" w lp title 'T = 8', "q04_gpu4_T12.dat" w lp title 'T = 12', "q04_gpu4_T14.dat" w lp title 'T = 14', "q04_gpu4_T16.dat" w lp title 'T = 16' lc rgb '#000000'
EOFMarker
