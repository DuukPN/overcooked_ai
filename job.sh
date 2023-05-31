#!/bin/sh
#
#SBATCH --job-name="ppo_experiment"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2022r2
module load miniconda3

conda activate harl

srun bash experiments/ppo_bc_experiments.sh
