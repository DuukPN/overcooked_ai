#!/bin/sh
#
#SBATCH --job-name="bc_experiment"
#SBATCH --partition=compute
#SBATCH --time=02:10:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2022r2
module load miniconda3

conda deactivate
conda activate /home/dniemantsverdr/env

srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments.py "$1" "$2"
srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments.py "$1" "$2" -h
