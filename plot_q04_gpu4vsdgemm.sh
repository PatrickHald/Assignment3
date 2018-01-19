#for((N=5000; N<20000; N += 150)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu4 $N $N $N); done >q04_gpu4_tmp.dat
#for((N=5000; N<20000; N += 150)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu3 $N $N $N); done >q04_gpu3_tmp.dat
#for((N=5000; N<20000; N += 150)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu2 $N $N $N); done >q04_gpu2_tmp.dat

awk ' NR % 2 == 1 ' q04_gpu4_tmp.dat > q04_gpu4.dat
awk ' NR % 2 == 1 ' q04_gpu3_tmp.dat > q04_gpu3.dat
awk ' NR % 2 == 1 ' q04_gpu2_tmp.dat > q04_gpu2.dat

join q04_gpu2.dat q04_gpu2.dat > q04_gpu22.dat
join q04_gpu2.dat q04_gpu3.dat > q04_gpu23.dat
join q04_gpu2.dat q04_gpu4.dat > q04_gpu24.dat

awk '{print $1, $2/$3}' q04_gpu22.dat > q04_gpu2_su.dat
awk '{print $1, $2/$3}' q04_gpu23.dat > q04_gpu3_su.dat
awk '{print $1, $2/$3}' q04_gpu24.dat > q04_gpu4_su.dat

#rm q02_gpu2_tmp.dat
#rm q02_lib_tmp.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "speed-up"
	set output "q04_gpu2_gpu3_gpu4_speedup.eps"
	plot "q04_gpu2_su.dat" w lp title 'matmult gpu2', "q04_gpu3_su.dat" w lp title 'matmult gpu3', "q04_gpu4_su.dat" w lp title 'matmult gpu4 T = 12'
EOFMarker
