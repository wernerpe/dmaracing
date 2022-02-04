#!/bin/bash
# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j
#SBATCH -n 4 # number of cores
#SBATCH -N 1 # number of Nodes
#SBATCH --gres=gpu:volta:1 # 1 GPU
# Loading the required module
source /etc/profile
source ~/.bashrc
conda activate dmaracing
# Run the script
xvfb-run -a python $HOME/projects/dmaracing/scripts/train1v1.py