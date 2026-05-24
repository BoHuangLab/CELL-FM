#!/bin/bash

#SBATCH --job-name=CELL-Diff_prediction
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --constraint=h200

module load anaconda
conda activate esm3

export PYTHONUNBUFFERED=1

nvidia-smi
bash scripts/cell_fmc/pretrain_hpa_cellfmc.sh
