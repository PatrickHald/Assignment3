for((N=10; N<500; N += 10)); do (MFLOPS_MAX_IT=1 matmult_f.nvcc gpu1 $N $N $N); done >q02_gpu1_tmp.dat

for((N=10; N<500; N += 10)); do (MFLOPS_MAX_IT=1 matmult_f.nvcc lib $N $N $N); done >q02_lib_tmp.dat

awk ' NR % 2 == 1 ' q02_gpu1_tmp.dat > q02_gpu1.dat

awk ' NR % 2 == 1 ' q02_lib_tmp.dat > q02_lib.dat

#awk ' NR % 2 == 1 { n = $1 ; k = $2 } NR % 2 == 0 {t = $1}; {print n , k, t}' seqvsmeth2_1_tmp.dat > seqvsmeth2_1_.dat
#awk ' NR % 2 == 1 { n = $1 ; k = $2 } NR % 2 == 0 {t = $1}; {print n , k, t}' seqvsmeth2_2_tmp.dat > seqvsmeth2_2_.dat

#awk ' NR % 2 == 0 ' seqvsmeth2_1_.dat > seqvsmeth2_1_t.dat
#awk ' NR % 2 == 0 ' seqvsmeth2_2_.dat > seqvsmeth2_2_t.dat

#awk ' {print $1, $1 * $1 * $2 / $3 }' seqvsmeth2_1_t.dat > seqvsmeth2_1_flups.dat
#awk ' {print $1, $1 * $1 * $2 / $3 }' seqvsmeth2_2_t.dat > seqvsmeth2_2_flups.dat

#awk ' {print $1, $3 }' seqvsmeth2_1_t.dat > seqvsmeth2_1_.dat
#awk ' {print $1, $3 }' seqvsmeth2_2_t.dat > seqvsmeth2_1_.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "Time (s)"
	set output "q02.eps"
	plot "q02_lib.dat" w lp title 'matmult_lib', "q02_gpu1.dat" w lp title 'matmult_gpu1'
EOFMarker
