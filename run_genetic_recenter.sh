#!/bin/bash
#SBATCH --job-name=genetic_algo
#SBATCH --output=genetic_algo_%j.out
#SBATCH --error=genetic_algo_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=   #email
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=250         
#SBATCH --time=96:00:00              

module purge

module load anaconda

source #env location
conda activate base

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON_PATH = #location to python path in env
$PYTHON_PATH Genetic_Stable_Recenter.py