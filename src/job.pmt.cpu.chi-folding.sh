#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH -C cpu
#SBATCH -t 04:00:00
#SBATCH -J chiadding5

export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

cp -r /pscratch/sd/w/wkim94/Moire/TBG/1.08/CC.KC-full.CH.rebo/Unitcell_3x3x1/Bot/epsilon_data/ epsilon_data_bot
cp -r /pscratch/sd/w/wkim94/Moire/TBG/1.08/CC.KC-full.CH.rebo/Unitcell_3x3x1/Top/epsilon_data/ epsilon_data_top

conda activate pmt-h5py
srun python ./runner_q_parallel_bot.py > runner_q_parallel_bot.out
srun python ./runner_q_parallel_top.py > runner_q_parallel_top.out
