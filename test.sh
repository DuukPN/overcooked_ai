#!/bin/sh
#
#SBATCH --job-name="test"
#SBATCH --partition=compute
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2022r2
module load miniconda3

conda deactivate
conda activate /scratch/dniemantsverdr/env2

srun python /scratch/dniemantsverdr/overcooked_ai/test.py
