
DATA_DIR=./data/finetune_data
CKPT_DIR=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4_more_finetune
CKPT_DIR_R=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4
DATASTORE_DIR=./data/image_features_datastore
K=4

srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 --lr 0.002 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 46600 --save-interval-updates 1000 --disable-validation  --use-knn-datastore --use-joint-attention --joint-layer-index 2  --ddp-backend legacy_ddp --wandb-project VaLM-baseline --fp16 --restore-file ${CKPT_DIR_R}/checkpoint_last.pt


CKPT_DIR1=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd_more_finetune
CKPT_DIR_R=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd
DATA_DIR=./data/finetune_data

srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR1} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 \
    --lr 0.002 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 46600 \
    --save-interval-updates 1000 --disable-validation --wandb-project VaLM-baseline --fp16 --restore-file ${CKPT_DIR_R}/checkpoint_last.pt


