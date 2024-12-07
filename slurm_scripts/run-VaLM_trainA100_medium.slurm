#!/bin/bash
#SBATCH --job-name=VaLM_train
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=./SLURM/LOG/VaLM_TrainA100_lrx4_more_medium.log
#SBATCH --error=./SLURM/ERROR/VaLM_TrainA100_lrx4_more_medium.err
#SBATCH --mem=60GB

echo "Hasi da"

bash myscripts/train_valmA100_medium.sh

echo "Amaitu da"
