#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_wiki.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_wiki.err
#SBATCH --mem=30GB

echo "Hasi da"


CKPT_DIR="./CHECKPOINTS"
ckpts=( "checkpoint_valmA100_40600upd_lrx4" "checkpoint_gpt_blind_lrx4_40600upd" "checkpoint_valmA100_40600upd_lrx4_more" "checkpoint_valmA100_40600upd_small")
for ckpt in "${ckpts[@]}"; do
  
    echo "wikitext-103"
    fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --pad-to-fixed-length  --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"
done
echo "Amaitu da"

