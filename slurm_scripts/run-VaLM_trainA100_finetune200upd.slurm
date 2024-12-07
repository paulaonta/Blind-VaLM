#!/bin/bash
#SBATCH --job-name=VaLM_finetune
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=./SLURM/LOG/VaLM_finetune_200upd.log
#SBATCH --error=./SLURM/ERROR/VaLM_finetune__200upd.err
#SBATCH --mem=30GB

echo "Hasi da"

bash myscripts/train_valmA100_finetune_200upd.sh

echo "Amaitu da"
