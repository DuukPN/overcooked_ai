#!/bin/sh
#
#SBATCH --job-name="bc_experiment"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments.py
