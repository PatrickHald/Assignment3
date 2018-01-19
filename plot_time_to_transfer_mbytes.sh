#for((N=10; N<50000; N += N/4)); do (MFLOPS_MAX_IT=1 MATMULT_COMPARE=0 matmult_f.nvcc gpu2 $N $N $N); done >q03_transfers_tmp.dat

#awk ' NR % 2 == 1 {p = $2} NR % 2 == 0 {s = $1}; NR % 2 == 0 {print s, p}' q03_transfers_tmp.dat > q03_transfers.dat

awk '{print $1, 100*$2}'q03_transfers.dat > q03_transfers100.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "kBytes"
	set ylabel "% of total time"
	set xtics (16,64,256,1024,4096,16384,65536,262144)
	set logscale x 2
	set output "q03_transfers_kbytes3.eps"
	plot "q03_transfers100.dat" w lp title 'matmult gpu2'
EOFMarker
