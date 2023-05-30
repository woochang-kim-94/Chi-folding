#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -J s2bgw

export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

conda activate pmt-h5py
srun python ./runner_q_parallel.py > runner_q_parallel.out
