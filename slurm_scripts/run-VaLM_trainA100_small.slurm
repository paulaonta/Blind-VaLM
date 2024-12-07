#!/bin/bash
#SBATCH --job-name=VaLM_train
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=./SLURM/LOG/VaLM_TrainA100_lrx4_small.log
#SBATCH --error=./SLURM/ERROR/VaLM_TrainA100_lrx4_small.err
#SBATCH --mem=60GB

echo "Hasi da"

bash touch ./SLURM/LOG/VaLM_TrainA100_lrx4_small.log
bash touch ./SLURM/ERROR/VaLM_TrainA100_lrx4_small.err

bash myscripts/train_valmA100_small.sh

echo "Amaitu da"
