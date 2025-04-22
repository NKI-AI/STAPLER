#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=???
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=pretrain_STAPLER_%A.out
#SBATCH --error=pretrain_STAPLER_%A.err

# activate conda env
source activate "stapler_env"

EXPERIMENT_NAME="pretrain_STAPLER"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)

echo 'Start script:'
echo 'current time' $(date)

MLFLOW_PORT=5000
mlflow server --backend-store-uri ../logs/mlflow/mlruns --host 0.0.0.0:$MLFLOW_PORT &
HYDRA_FULL_ERROR=1 python pretrain.py task_name=$EXPERIMENT_NAME

echo 'current time' $(date)
echo 'Finished'
