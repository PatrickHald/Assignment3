#for((N=16; N<10000; N += N/4)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu4 $N $N $N); done >q05_gpu4_tmpn.dat
#for((N=16; N<10000; N += N/4)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu5 $N $N $N); done >q05_gpu5_tmpn.dat
#for((N=16; N<10000; N += N/4)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpulib $N $N $N); done >q05_gpulib_tmpn.dat

#awk ' NR % 2 == 1 ' q05_gpu4_tmp.dat > q05_gpu4.dat
#awk ' NR % 2 == 1 ' q05_gpu5_tmp.dat > q05_gpu5.dat

#awk ' NR % 2 == 0 {print $1, $2/1000000}' q05_gpu4_tmpn.dat > q05n_gpu4_fl.dat
#awk ' {print $1, $2/1000000}' q05_gpulib_tmpn.dat > q05n_gpulib_fl.dat

#awk ' NR % 2 == 0 {print $1, $2/1000000}' q05_gpu5_tmpn.dat > q05n_gpu5_fl.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Memory (kBytes)"
 	set ylabel "MFlop/s"
	set output "q05_flops_45.eps"
	set logscale x 10
	plot "q05n_gpu4_fl.dat" w lp title 'matmult gpu4 T=12', "q05n_gpu5_fl.dat" w lp title 'matmult gpu5'
EOFMarker
