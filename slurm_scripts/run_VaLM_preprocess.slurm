#!/bin/bash
#SBATCH --job-name=VaLM_proba
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --output=./SLURM/LOG/VaLM_TrainA100proba.log
#SBATCH --error=./SLURM/ERROR/VaLM_TrainA100proba.err
#SBATCH --mem=50GB

echo "Hasi da"

bash myscripts/preprocess_valm_text.sh

echo "Amaitu da"
