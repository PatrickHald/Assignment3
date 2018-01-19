for((N=5000; N<7000; N += 50)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu2 $N $N $N); done >q02_gpu2_tmp_all.dat
for((N=5000; N<7000; N += 50)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc lib $N $N $N); done >q02_lib_tmp.dat

awk ' NR % 2 == 1 {print $1, $3}' q02_gpu2_tmp_all.dat > q02_gpu2_all.dat
awk ' NR % 2 == 1 {print $1, $2}' q02_gpu2_tmp_all.dat > q02_gpu2_ksmall.dat
awk ' NR % 2 == 1 ' q02_lib_tmp.dat > q02_lib.dat

#rm q02_gpu2_tmp.dat
#rm q02_lib_tmp.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "Time (s)"
	set output "q03_lib_vs_gpu2.eps"
	plot "q02_gpu2_all.dat" w lp title 'matmult gpu2 with transfer', "q02_gpu2_ksmall.dat" w lp title 'matmult gpu2 without transfer', "q02_lib.dat" w lp title 'matmult lib'
EOFMarker
