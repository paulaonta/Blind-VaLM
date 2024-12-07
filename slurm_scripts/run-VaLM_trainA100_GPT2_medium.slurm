#!/bin/bash
#SBATCH --job-name=gpt_medium
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=./SLURM/LOG/VaLM_TrainA100_GPT_medium.log
#SBATCH --error=./SLURM/ERROR/VaLM_TrainA100_GPT_medium.err
#SBATCH --mem=30GB

echo "Hasi da"

bash myscripts/train_gpt_blind_medium.sh

echo "Amaitu da"
