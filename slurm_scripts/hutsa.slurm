#!/bin/bash
#SBATCH --job-name=train_d
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/hutsa.log
#SBATCH --error=./SLURM/ERROR/hutsa.err
#SBATCH --mem=1GB


echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

srun python3 hutsa.py

sleep 1d