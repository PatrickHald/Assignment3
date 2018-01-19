#for((N=1500; N<2000; N += 10)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu3 $N $N $N); done >q04_gpu3b_tmp.dat
for((N=1500; N<2000; N += 10)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu3 $N $N $N); done >q04_gpu3r_tmp.dat

awk ' NR % 2 == 1 ' q04_gpu3b_tmp.dat > q04_gpu3b.dat
awk ' NR % 2 == 1 ' q04_gpu3r_tmp.dat > q04_gpu3r.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
 	set ylabel "Time (s)"
	set output "q04_neighbor.eps"
	plot "q04_gpu3b.dat" w lp title 'below neighbor', "q04_gpu3r.dat" w lp title 'right neighbor'
EOFMarker
