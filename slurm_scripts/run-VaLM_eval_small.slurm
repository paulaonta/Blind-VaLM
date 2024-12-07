#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_proba.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_proba.err
#SBATCH --mem=30GB


echo "Hasi da"

CKPT_DIR="./CHECKPOINTS"
ckpts=( "checkpoint_valmA100_40600upd_lrx4") #"checkpoint_valmA100_40600upd_small")
for ckpt in "${ckpts[@]}"; do
  
    echo -e "\n"
    echo "Evaluating MEMORY COLOR"
    srun python evaluation_scripts/verify_color_prediction.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    
    #echo -e "\n" 
    #echo "Evaluation their MEMORY COLOR"
    #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
        
    echo -e "\n" 
    echo "Evaluating COLOR TERMS"
    srun python evaluation_scripts/verify_color_prediction.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --color-terms
    
    #echo -e "\n" 
    #echo "Evaluation their COLOR TERMS"
    #srun python evaluation_scripts/verify_color_prediction_orig.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --color-terms   

    echo -e "\n"
    echo "Evaluation ViComTe all test --> obj"
    srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --obj_truth

    echo -e "\n"
    echo "Evaluation ViComTe all test --> alt"
    srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  

    echo -e "\n"
    echo "Evaluation ViComTe single test --> obj"
    srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --obj_truth --data-path ./data/object_color/test_color_single.jsonl

    echo -e "\n"
    echo "Evaluation ViComTe single test -->alt"
    srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --data-path ./data/object_color/test_color_single.jsonl


    
    echo -e "\n"
    echo "Evaluation SHAPE IT (P1)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --p 0

    echo -e "\n"
    echo "Evaluation SHAPE IT (P2)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --p 1

    echo -e "\n"
    echo "Evaluation SHAPE IT (P3)"
    srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --p 2


    echo -e "\n"
    echo "Evaluation SIZE (P1) -->relative size"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"    --data-path ./data/object_size/relative_size.jsonl

    echo -e "\n"
    echo "Evaluation SIZE (P2) --> relative size"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"    --data-path ./data/object_size/relative_size.jsonl
    echo "one"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --oneToken --data-path ./data/object_size/relative_si\
ze.jsonl


    echo -e "\n"                                                                              
    echo "Evaluation SIZE (P1)--> ViComTe"                                                      
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   

    echo -e "\n"
    echo "Evaluation SIZE (P2) --> ViComTe"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    echo "one"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --oneToken

    
    echo -e "\n"
    echo "Evaluation SIZE (P1) --> tnwt"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --tnwt

    echo -e "\n"
    echo "Evaluation SIZE (P2) --> tnwt"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --tnwt
    echo "one"
    srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --oneToken --tnwt

    
    
    echo -e "\n"
    echo "Evaluation HEIGHT (P1)"
    srun python evaluation_scripts/verify_height_reason_combinedSizes_p1.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  

    echo -e "\n"
    echo "Evaluation HEIGHT (P2)"
    srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    echo "one"
    srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --oneToken

    '''
    echo -e "\n"
    echo "Evaluation their RELATIVE SIZE"
    srun python evaluation_scripts/verify_size_reason_orig.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  

    echo -e "\n"                                            
    echo "Evaluation RELATIVE SIZE (smaller)"
    srun python evaluation_scripts/verify_size_reason.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --eval-smaller

    echo -e "\n"
    echo "Evaluation RELATIVE SIZE (bigger)"
    srun python evaluation_scripts/verify_size_reason.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   

    echo -e "\n"
    echo "Evaluation RELATIVE SIZE fourth"
    srun python evaluation_scripts/verify_size_reason_inPairs.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   
    
    echo -e "\n"
    echo "Evaluation their RELATIVE SIZE"
    srun python evaluation_scripts/verify_size_reason_orig.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    
    echo -e "\n" 
    echo "Evaluation RELATIVE SIZE (bigger)"                                                                                                                                                 
    srun python evaluation_scripts/verify_size_reason.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   
    
    echo -e "\n"
    echo "Evaluation OBJECT SHAPE --> obj"
    srun python evaluation_scripts/verify_shape_reason.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --obj_truth --data-path ./data/object_size/test_shape.jsonl

    echo -e "\n"
    echo "Evaluation OBJECT SHAPE --> alt"
    srun python evaluation_scripts/verify_shape_reason.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --data-path ./data/object_size/test_shape.jsonl
    '''
    
    echo -e "\n"
    echo "wikitext-103"
    fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --pad-to-fixed-length --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"
   
    echo -e "\n"
    echo "lambada"
    fairseq-eval-lm ./data/lambada/ --batch-size 4 --sample-break-mode eos --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"
    python evaluation_scripts/eval_lambada.py --data-path ./data/lambada/lambada_test.jsonl --preprocess --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"

    echo -e "\n"
    echo "coco"
    fairseq-eval-lm ./data/coco/ --batch-size 4 --sample-break-mode eos --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"
    
    
    echo -e "\n"
    echo "mpqa"
    srun python evaluation_scripts/eval_posNeg.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --data-path ./data/mpqa/test.json
    echo "all"
    srun python evaluation_scripts/eval_posNeg.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --data-path ./data/mpqa/test_all.json                   

    echo -e "\n"
    echo "stt2"
    srun python evaluation_scripts/eval_posNeg.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    
    echo -e "\n"
    echo "Evaluating DBPedia adn AGNews"
    srun python evaluation_scripts/eval_dbp_agn.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"  
    srun python evaluation_scripts/eval_dbp_agn.py --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"   --agn
done

echo "Amaitu da"

