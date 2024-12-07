DATA_DIR=./data/data-bin
CKPT_DIR=./CHECKPOINTS/checkpoint_valmA100_40600upd_small
DATASTORE_DIR=./data/image_features_datastore
K=4

srun torchrun --nproc_per_node=8 fairseq/fairseq_cli/train.py ${DATA_DIR}/0:${DATA_DIR}/1:${DATA_DIR}/2:${DATA_DIR}/3:${DATA_DIR}/4:${DATA_DIR}/5 --save-dir ${CKPT_DIR} --task language_modeling --arch transformer_lm_gpt2_small --share-decoder-input-output-embed --dropout 0.1 --optimizer adam  --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 2.0 --lr 0.002 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07  --tokens-per-sample 512 --sample-break-mode none --max-tokens 32768 --max-update 40600 --save-interval-updates 10000 --disable-validation  --use-knn-datastore --use-joint-attention --joint-layer-index 2 --ddp-backend legacy_ddp --wandb-project VaLM-baseline --fp16