TIMEFORMAT='%3R'

#for((n=1; n<= 30; n+= 1)); do echo "$n"; time OMP_NUM_THREADS=$n OMP_WAIT_POLICY=active proj2 2 2600 400 0; done > to_compare_mandel.dat 2>&1

awk ' NR == 3 { t1 = $1}; NR % 3 == 1 { t = $1 }; NR % 3 == 0 {s = $1}; {print t, s, t1}' to_compare_mandel.dat > to_compare_mandel_tmp.dat

awk ' NR % 3 == 0 {print $1, $3 / $2}' to_compare_mandel_tmp.dat > to_compare_mandel_speedup_clean.dat
awk ' NR % 3 == 0 {print $1, 400 / $2}' to_compare_mandel_tmp.dat > to_compare_mandel_perf_clean.dat

gnuplot -persist <<-EOFMarker
	set terminal postscript eps enhanced color font 'Helvetica,10'
	set xlabel "Number of threads"
	set ylabel "Speedup"
	set output "Compare_mandel.eps"
	plot "times_mandel_clean.dat" w lp title 'Mandelbrot','to_compare_mandel_speedup_clean.dat' w lp title 'Jacobi', "identity.dat" w l title 'Ideal'
EOFMarker
