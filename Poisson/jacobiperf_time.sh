TIMEFORMAT='%3R'
for((N=500; N<1000; N += 50)); do (time proj2 0 $N 20000 0); done >seqvsmeth2_1_tmp.dat 2>&1
for((N=500; N<1000; N += 50)); do (time OMP_NUM_THREADS=16 proj2 2 $N 20000 0); done >seqvsmeth2_2_tmp.dat 2>&1
#awk ' NR % 2 == 1 ' no_kmax_jac_tmp.dat > iter_no_kmax_jac.dat
#awk ' NR % 2 == 1 ' no_kmax_gau_tmp.dat > iter_no_kmax_gau.dat

awk ' NR % 2 == 1 { n = $1 ; k = $2 } NR % 2 == 0 {t = $1}; {print n , k, t}' seqvsmeth2_1_tmp.dat > seqvsmeth2_1_.dat
awk ' NR % 2 == 1 { n = $1 ; k = $2 } NR % 2 == 0 {t = $1}; {print n , k, t}' seqvsmeth2_2_tmp.dat > seqvsmeth2_2_.dat

awk ' NR % 2 == 0 ' seqvsmeth2_1_.dat > seqvsmeth2_1_t.dat
awk ' NR % 2 == 0 ' seqvsmeth2_2_.dat > seqvsmeth2_2_t.dat

awk ' {print $1, $1 * $1 * $2 / $3 }' seqvsmeth2_1_t.dat > seqvsmeth2_1_flups.dat
awk ' {print $1, $1 * $1 * $2 / $3 }' seqvsmeth2_2_t.dat > seqvsmeth2_2_flups.dat

#awk ' {print $1, $3 }' seqvsmeth2_1_t.dat > seqvsmeth2_1_.dat
#awk ' {print $1, $3 }' seqvsmeth2_2_t.dat > seqvsmeth2_1_.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Matrix size"
	set ylabel "LUps"
	set output "lupsthreads16_seq_vs_meth2.eps"
	plot "seqvsmeth2_1_flups.dat" w lp title 'Sequential', "seqvsmeth2_2_flups.dat" w lp title 'Parallelized'
EOFMarker
