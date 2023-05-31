#!/bin/sh
#
#SBATCH --job-name="bc_experiment_single"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments_single.py
