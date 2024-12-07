'''
DATA_DIR=./data/COCO_train
CKPT_DIR=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4_more_finetune_200upd
CKPT_DIR_R=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4
DATASTORE_DIR=./data/image_features_datastore
K=4
'''
#srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 --lr 0.002 --lr-scheduler inverse_sqrt --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 200 --save-interval-updates 1000 --disable-validation  --use-knn-datastore --use-joint-attention --joint-layer-index 2  --ddp-backend legacy_ddp --wandb-project VaLM-baseline --fp16 --finetune-from-model ${CKPT_DIR_R}/checkpoint_last.pt
'''
CKPT_DIR1=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd_more_finetune_200upd
CKPT_DIR_R=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd
'''
#srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR1} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed \
#    --dropout 0.1 --optimizer adam --adam-betas'(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 \
#    --lr 0.002 --lr-scheduler inverse_sqrt \
#    --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 200 \
#    --save-interval-updates 1000 --disable-validation --wandb-project VaLM-baseline --fp16 --finetune-from-model ${CKPT_DIR_R}/checkpoint_last.pt


DATA_DIR=./data/finetune_data3
CKPT_DIR=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4_more_finetune_200upd_3
CKPT_DIR_R=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4
DATASTORE_DIR=./data/image_features_datastore
K=4

srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 --lr 0.002 --lr-scheduler inverse_sqrt  --tokens-per-sample 512 --sample-break-mode none --max-tokens 32768 --max-update 800 --save-interval-updates 1000 --disable-validation  --use-knn-datastore --use-joint-attention --joint-layer-index 2  --ddp-backend legacy_ddp --wandb-project VaLM-baseline --finetune-from-model ${CKPT_DIR_R}/checkpoint_last.pt

CKPT_DIR1=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd_more_finetune_200upd_3
CKPT_DIR_R=./CHECKPOINTS/checkpoint_gpt_blind_lrx4_40600upd

srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR} --save-dir ${CKPT_DIR1} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 2.0 \
    --lr 0.002 --lr-scheduler inverse_sqrt  \
    --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 800 \
    --save-interval-updates 1000 --disable-validation --wandb-project VaLM-baseline --fp16 --finetune-from-model ${CKPT_DIR_R}/checkpoint_last.pt

