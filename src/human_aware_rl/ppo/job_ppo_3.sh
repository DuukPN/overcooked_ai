#!/bin/sh
#
#SBATCH --job-name="bc_experiment_train_ppo"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=6GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2022r2
module load miniconda3

conda deactivate
conda activate /home/dniemantsverdr/env

srun python ~/overcooked_ai/src/human_aware_rl/ppo/ppo_rllib_client.py with  seeds=[0]  layout_name="forced_coordination" clip_param=0.258 gamma=0.972 grad_clip=0.295 kl_coeff=0.31 lmbda=0.6 lr=2.77e-4 num_training_iters=650 old_dynamics=True reward_shaping_horizon=4000000 use_phi=False vf_loss_coeff=0.016 results_dir=reproduced_results/ppo_sp_forced_coordination
