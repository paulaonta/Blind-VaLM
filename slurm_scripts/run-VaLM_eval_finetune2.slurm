#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_finetune2.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_finetune2.err
#SBATCH --mem=30GB

echo "Hasi da"

CKPT_DIR="./CHECKPOINTS"
#ckpts=( "checkpoint_valmA100_40600upd_lrx4_more_5000upd" "checkpoint_valmA100_40600upd_lrx4_more_finetune" "checkpoint_gpt_blind_lrx4_40600upd_more_finetune")
ckpts=( "checkpoint_valmA100_40600upd_lrx4_more_finetune") 
for ckpt in "${ckpts[@]}"; do
    for ckpt_path in "$CKPT_DIR/$ckpt"/*.pt; do

        echo -e "\n"
        echo "Evaluating MEMORY COLOR"
        srun python evaluation_scripts/verify_color_prediction.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluating COLOR TERMS"
        srun python evaluation_scripts/verify_color_prediction.py --path "$ckpt_path" --model-overrides --color-terms

        echo -e "\n"
        echo "Evaluation ViComTe all test --> obj"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path" --model-overrides --obj_truth

        echo -e "\n"
        echo "Evaluation ViComTe all test --> alt"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation ViComTe single test --> obj"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path" --model-overrides --obj_truth --data-path ./data/object_color/test_color_single.jsonl

        echo -e "\n"
        echo "Evaluation ViComTe single test -->alt"
        srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path "$ckpt_path" --model-overrides --data-path ./data/object_color/test_color_single.jsonl



        echo -e "\n"
        echo "Evaluation SHAPE IT (P1)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path" --model-overrides --p 0

        echo -e "\n"
        echo "Evaluation SHAPE IT (P2)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path" --model-overrides --p 1

        echo -e "\n"
        echo "Evaluation SHAPE IT (P3)"
        srun python evaluation_scripts/verify_shape_reason_ShapeItc.py --path "$ckpt_path" --model-overrides --p 2


        echo -e "\n"
        echo "Evaluation SIZE (P1)"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$ckpt_path" --model-overrides


        echo -e "\n"
        echo "Evaluation SIZE (P2)"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides
        echo "one"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides --oneToken

        echo -e "\n"
        echo "Evaluation SIZE (P1) --> tnwt"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path "$ckpt_path" --model-overrides --tnwt

        echo -e "\n"
        echo "Evaluation SIZE (P2) --> tnwt"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides --tnwt
        echo "one"
        srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides --oneToken --tnwt



        echo -e "\n"
        echo "Evaluation HEIGHT (P1)"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p1.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation HEIGHT (P2)"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides
        echo "one"
        srun python evaluation_scripts/verify_height_reason_combinedSizes_p2.py --path "$ckpt_path" --model-overrides --oneToken

	'''
        echo -e "\n"
        echo "Evaluation their RELATIVE SIZE"
        srun python evaluation_scripts/verify_size_reason_orig.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation RELATIVE SIZE (smaller)"
        srun python evaluation_scripts/verify_size_reason.py --path "$ckpt_path" --model-overrides --eval-smaller

        echo -e "\n"
        echo "Evaluation RELATIVE SIZE (bigger)"
        srun python evaluation_scripts/verify_size_reason.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation RELATIVE SIZE fourth"
        srun python evaluation_scripts/verify_size_reason_inPairs.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation their RELATIVE SIZE"
        srun python evaluation_scripts/verify_size_reason_orig.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation RELATIVE SIZE (bigger)"
        srun python evaluation_scripts/verify_size_reason.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluation OBJECT SHAPE --> obj"
        srun python evaluation_scripts/verify_shape_reason.py --path "$ckpt_path" --model-overrides --obj_truth --data-path ./data/object_size/test_shape.jsonl

        echo -e "\n"
        echo "Evaluation OBJECT SHAPE --> alt"
        srun python evaluation_scripts/verify_shape_reason.py --path "$ckpt_path" --model-overrides --data-path ./data/object_size/test_shape.jsonl
	'''

        echo -e "\n"
        echo "wikitext-103"
        fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --path "$ckpt_path"

        echo -e "\n"
        echo "lambada"
        fairseq-eval-lm ./data/lambada/ --batch-size 4 --sample-break-mode eos --path "$ckpt_path"
        python evaluation_scripts/eval_lambada.py --data-path ./data/lambada/lambada_test.jsonl --preprocess --path "$ckpt_path"

        echo -e "\n"
        echo "coco"
        fairseq-eval-lm ./data/coco/ --batch-size 4 --sample-break-mode eos --path "$ckpt_path"


        echo -e "\n"
        echo "mpqa"
        srun python evaluation_scripts/eval_posNeg.py --path "$ckpt_path" --model-overrides --data-path ./data/mpqa/test.json
        echo "all"
        srun python evaluation_scripts/eval_posNeg.py --path "$ckpt_path" --model-overrides --data-path ./data/mpqa/test_all.json

        echo -e "\n"
        echo "stt2"
        srun python evaluation_scripts/eval_posNeg.py --path "$ckpt_path" --model-overrides

        echo -e "\n"
        echo "Evaluating DBPedia adn AGNews"
        srun python evaluation_scripts/eval_dbp_agn.py --path "$ckpt_path" --model-overrides
        srun python evaluation_scripts/eval_dbp_agn.py --path "$ckpt_path" --model-overrides --agn

    done
done

echo "Amaitu da"
