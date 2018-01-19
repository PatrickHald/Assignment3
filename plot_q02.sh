for((N=10; N<500; N += 10)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu1 $N $N $N); done >q02_gpu1_tmp.dat

for((N=10; N<500; N += 10)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc lib $N $N $N); done >q02_lib_tmp.dat

awk ' NR % 2 == 1 ' q02_gpu1_tmp.dat > q02_gpu1.dat

awk ' NR % 2 == 1 ' q02_lib_tmp.dat > q02_lib.dat

rm q02_lib_tmp.dat
rm q02_gpu1_tmp.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "Time (s)"
	set output "q02_gpu1_vs_lib.eps"
	plot "q02_lib.dat" w lp title 'matmult lib', "q02_gpu1.dat" w lp title 'matmult gpu1'
EOFMarker
