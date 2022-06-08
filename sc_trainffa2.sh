#!/bin/bash
# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j
#SBATCH -n 4 # number of cores
#SBATCH -N 1 # number of Nodes
#SBATCH --gres=gpu:volta:1 # 1 GPU

# Loading the required module
source /etc/profile
#source ~/.bashrc
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/etc/profile.d/conda.sh
#export PYTHONNOUSERSITE=True
conda init bash
conda activate /home/gridsan/pwerner/.conda/envs/dmaracing
conda env list
# Run the script
xvfb-run -a python $HOME/projects/dmaracing/scripts/trainffa2.py num_steps_per_env=64 
