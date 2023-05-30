#!/bin/bash

#SBATCH -J JOBNAME
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH --mail-user=wkim94.hpc.job@gmail.com
#SBATCH --mail-type=all

module load berkeleygw/3.0.1-cpu.lua

export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1

ulimit -s unlimited
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun ~/codes/bgw-pump-eps-copied/bin/epsilon.cplx.x &> epsilon.out
