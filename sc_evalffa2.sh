#!/bin/bash
# Slurm sbatch options
#SBATCH -o ffa2.sh.log-%j
#SBATCH -n 4 # number of cores
#SBATCH -N 1 # number of Nodes
#SBATCH --gres=gpu:volta:1 # 1 GPU

# Loading the required module
#source /etc/profile
#source ~/.bashrc
module load anaconda/2022b
#source /state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/etc/profile.d/conda.sh
#export PYTHONNOUSERSITE=True
#conda init 
#conda activate /home/gridsan/pwerner/.conda/envs/dmaracing
#conda list
# Run the script
xvfb-run -a python $HOME/projects/dmaracing/scripts/continuousevaluation.py headless=True #num_steps_per_env=64 