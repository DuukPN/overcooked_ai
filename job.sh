#!/bin/sh
#
#SBATCH --job-name="bc_experiment"
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2022r2
module load miniconda3

conda deactivate
conda activate /home/dniemantsverdr/env

for layout in "random3" "coordination_ring" "cramped_room" "random0" "asymmetric_advantages"
do
  srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments.py $layout
  srun python ~/overcooked_ai/src/human_aware_rl/imitation/my_experiments.py $layout -s
done
