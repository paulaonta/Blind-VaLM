#!/bin/bash
#SBATCH --job-name=VaLM_eval
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/VaLM_Eval_wikitime4.log
#SBATCH --error=./SLURM/ERROR/VaLM_Eval_wikitime4.err
#SBATCH --mem=30GB

echo "Hasi da"


CKPT_DIR="./CHECKPOINTS"
ckpts=( "checkpoint_valmA100_40600upd_small") #"checkpoint_valmA100_40600upd_lrx4"  "checkpoint_valmA100_40600upd_lrx4_more"

BATCH_SIZE=4  # Define the batch size
DATASET_SIZE=2890
num_batches=$((DATASET_SIZE / BATCH_SIZE))

total_time=0
num_ckpts=${#ckpts[@]}

for ckpt in "${ckpts[@]}"; do
    start_time=$(date +%s)
    
    # Run the fairseq evaluation
    fairseq-eval-lm ./data/wikitext-103/ --batch-size $BATCH_SIZE --sample-break-mode eos --pad-to-fixed-length --path "$CKPT_DIR/$ckpt/checkpoint_last.pt"
    
    end_time=$(date +%s)
    
    # Calculate the duration for this checkpoint
    duration=$((end_time - start_time))
    echo "Time taken for $ckpt: $duration seconds"
    
    # Calculate time per batch for this checkpoint
    time_per_batch=$(echo "scale=2; $duration / $num_batches" | bc)
    echo "Time per batch for $ckpt: $time_per_batch seconds"
    
    # Add the duration to the total time
    total_time=$((total_time + duration))
done

# Calculate the average time per batch
average_time_per_batch=$(echo "scale=2; $total_time / ($num_batches * num_ckpts)" | bc)
echo "Average time per batch across all checkpoints: $average_time_per_batch seconds"

echo "Amaitu da

