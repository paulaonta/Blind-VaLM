#!/bin/bash
#SBATCH --job-name=VaLM_train
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=./SLURM/LOG/VaLM_Train.log
#SBATCH --error=./SLURM/ERROR/VaLM_Train.err
#SBATCH --mem=30GB

echo "Hasi da"

bash myscripts/train_valmV100.sh

echo "Amaitu da"
