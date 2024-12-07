#!/bin/bash
#SBATCH --job-name=gpt_blind_train
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=./SLURM/LOG/VaLM_TrainA100_GPT.log
#SBATCH --error=./SLURM/ERROR/VaLM_TrainA100_GPT.err
#SBATCH --mem=30GB

echo "Hasi da"

bash myscripts/train_gpt_blind.sh

echo "Amaitu da"
