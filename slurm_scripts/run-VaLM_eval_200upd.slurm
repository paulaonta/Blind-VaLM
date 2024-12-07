#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_200upd_3.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_200upd_3.err
#SBATCH --mem=30GB

echo "Hasi da"

CKPT_DIR="./CHECKPOINTS"
#ckpts=( "checkpoint_valmA100_40600upd" "checkpoint_valmA100_40600upd_lrx4" "checkpoint_gpt_blind_40600upd" "checkpoint_gpt_blind_125860upd" "checkpoint_gpt_blind_lrx4" "checkpoint_gpt_blind_125860upd_moreData"  "checkpoint_gpt_blind_lrx4_moreData" "checkpoint_gpt_blind_lrx4_40600upd")
#ckpts=( "checkpoint_valmA100_40600upd_lrx4_more_finetune_200upd" "checkpoint_gpt_blind_lrx4_40600upd_more_finetune_200upd" "checkpoint_valmA100_40600upd_lrx4_more_finetune_200upd_2" "checkpoint_gpt_blind_lrx4_40600upd_more_finetune_200upd_2")
ckpts=( "checkpoint_valmA100_40600upd_lrx4_more_finetune_200upd_3" "checkpoint_gpt_blind_lrx4_40600upd_more_finetune_200upd_3")
for ckpt in "${ckpts[@]}"; do
    for ckpt_path in "$CKPT_DIR/$ckpt"/*.pt; do
        echo -e "\n"
        echo "Evaluating MEMORY COLOR"
        srun python evaluation_scripts/verify_color_prediction.py --path "$ckpt_path" 
        
        #echo -e "\n" 
        #echo "Evaluation their MEMORY COLOR"
        #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$ckpt_path" 
            
        echo -e "\n" 
        echo "Evaluating COLOR TERMS"
        srun python evaluation_scripts/verify_color_prediction.py --path "$ckpt_path"  --color-terms
        
        #echo -e "\n" 
        #echo "Evaluation their COLOR TERMS"
        #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$ckpt_path"  --color-terms   

        echo -e "\n"
        echo "Evaluation ViComTe all test --> obj"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path"  --obj_truth

        echo -e "\n"
        echo "Evaluation ViComTe all test --> alt"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path" 

        echo -e "\n"
        echo "Evaluation ViComTe single test --> obj"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path"  --obj_truth --data-path ./data/object_color/test_color_single.jsonl

        echo -e "\n"
        echo "Evaluation ViComTe single test -->alt"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path"  --data-path ./data/object_color/test_color_single.jsonl


        
        echo -e "\n"
        echo "Evaluation SHAPE IT (P1)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path"  --p 0

        echo -e "\n"
        echo "Evaluation SHAPE IT (P2)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path"  --p 1

        echo -e "\n"
        echo "Evaluation SHAPE IT (P3)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path"  --p 2


        echo -e "\n"
        echo "Evaluation SIZE (P1) -->relative size"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$ckpt_path"   --data-path ./data/object_size/relative_size.jsonl

        echo -e "\n"
        echo "Evaluation SIZE (P2) --> relative size"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path"   --data-path ./data/object_size/relative_size.jsonl
        echo "one"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path"  --oneToken --data-path ./data/object_size/relative_si\
    ze.jsonl


        echo -e "\n"                                                                              
        echo "Evaluation SIZE (P1)--> ViComTe"                                                      
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$ckpt_path"  

        echo -e "\n"
        echo "Evaluation SIZE (P2) --> ViComTe"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path" 
        echo "one"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path"  --oneToken

        
        echo -e "\n"
        echo "Evaluation SIZE (P1) --> tnwt"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$ckpt_path"  --tnwt

        echo -e "\n"
        echo "Evaluation SIZE (P2) --> tnwt"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path"  --tnwt
        echo "one"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path"  --oneToken --tnwt

        
        
        echo -e "\n"
        echo "Evaluation HEIGHT (P1)"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p1.py --path "$ckpt_path" 

        echo -e "\n"
        echo "Evaluation HEIGHT (P2)"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$ckpt_path" 
        echo "one"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$ckpt_path"  --oneToken
    done
done
echo "Amaitu da"

