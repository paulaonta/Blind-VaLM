DATA_DIR=./data/data-bin
CKPT_DIR=./CHECKPOINTS/checkpoint_valmA100_40600upd_nr
CKPT_DIR_R=./CHECKPOINTS/checkpoint_valmA100_40600upd_lrx4
DATASTORE_DIR=./data/image_features_datastore
K=4

srun torchrun --nproc_per_node=4 fairseq/fairseq_cli/train.py ${DATA_DIR}/0:${DATA_DIR}/1:${DATA_DIR}/2:${DATA_DIR}/3:${DATA_DIR}/4:${DATA_DIR}/5 --save-dir ${CKPT_DIR} --task language_modeling --arch transformer_lm_gpt --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 2.0 --lr 5e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode none --max-tokens 65536 --max-update 40600 --save-interval-updates 5000 --disable-validation --use-joint-attention --joint-layer-index 2 --ddp-backend legacy_ddp --wandb-project VaLM-baseline --fp16



