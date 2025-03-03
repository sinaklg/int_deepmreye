#!/bin/bash
#SBATCH -p skylake
#SBATCH -A b327
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH -e /Users/sinakling/Desktop/DeepMReyeCalib/derivatives/int_deepmreye/log_outputs/deepmreye_%N_%j_%a.err
#SBATCH -o /Users/sinakling/Desktop/DeepMReyeCalib/derivatives/int_deepmreye/log_outputs/deepmreye_%N_%j_%a.out
#SBATCH -J Calib_int_test

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python cerimed_deepmreye.py /Users/sinakling/Desktop/ DeepMReyeCalib 327