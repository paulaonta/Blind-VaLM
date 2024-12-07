#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_200upd_ckpt1.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_200upd_ckpt1.err
#SBATCH --mem=30GB

echo "Hasi da"

CKPT_DIR="./CHECKPOINTS"
#ckpts=( "checkpoint_valmA100_40600upd" "checkpoint_valmA100_40600upd_lrx4" "checkpoint_gpt_blind_40600upd" "checkpoint_gpt_blind_125860upd" "checkpoint_gpt_blind_lrx4" "checkpoint_gpt_blind_125860upd_moreData"  "checkpoint_gpt_blind_lrx4_moreData" "checkpoint_gpt_blind_lrx4_40600upd")
ckpts=( "checkpoint_gpt_blind_lrx4_40600upd_more_finetune3" "checkpoint_valmA100_40600upd_lrx4_more_finetune3" )
for ckpt in "${ckpts[@]}"; do
    echo -e "\n"
    echo "Evaluating MEMORY COLOR"
    srun python evaluation_scripts/verify_color_prediction.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt" 
    
    #echo -e "\n" 
    #echo "Evaluation their MEMORY COLOR"
    #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt" 
        
    echo -e "\n" 
    echo "Evaluating COLOR TERMS"
    srun python evaluation_scripts/verify_color_prediction.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --color-terms
    
    #echo -e "\n" 
    #echo "Evaluation their COLOR TERMS"
    #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --color-terms   

    echo -e "\n"
    echo "Evaluation ViComTe all test --> obj"
    srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --obj_truth
    
    echo -e "\n"
    echo "Evaluation SHAPE IT (P1)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --p 0

    echo -e "\n"
    echo "Evaluation SHAPE IT (P2)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --p 1

    echo -e "\n"
    echo "Evaluation SHAPE IT (P3)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint1.pt"  --p 2

done
echo "Amaitu da"

