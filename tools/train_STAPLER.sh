#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=???
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=train_STAPLER_%A.out
#SBATCH --error=train_STAPLER_%A.err

# activate conda env
source activate "stapler_env"

EXPERIMENT_NAME="train_STAPLER"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)

echo 'Start script:'
echo 'current time' $(date)


HYDRA_FULL_ERROR=1 python train_5_fold.py task_name=$EXPERIMENT_NAME

echo 'current time' $(date)
echo 'Finished'
