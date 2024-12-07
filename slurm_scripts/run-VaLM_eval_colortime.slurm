#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_colortime.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_colortime.err
#SBATCH --mem=30GB

echo "Hasi da"

CKPT_DIR="./CHECKPOINTS"
ckpts=( "checkpoint_valmA100_40600upd_small") #("checkpoint_valmA100_40600upd_lrx4"  "checkpoint_valmA100_40600upd_lrx4_more") 
for ckpt in "${ckpts[@]}"; do
  
    echo -e "\n"
    echo "Evaluating MEMORY COLOR"
    srun python evaluation_scripts/verify_color_predictionTime.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    
done

echo "Amaitu da"

