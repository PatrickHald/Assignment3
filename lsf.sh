#!/bin/bash
#BSUB -J lsf
#BSUB -o lsf_%J.out
#BSUB -e lsf_%J.err
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 12

#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
module load cuda/9.1

plot_q03.sh
