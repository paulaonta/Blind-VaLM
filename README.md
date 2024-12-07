# Blind-VaLM
Official implementation of our paper "[Improving the Efficiency of Visually Augmented Language Models](https://arxiv.org/abs/2409.11148)". Please cite our paper if you find this repository helpful in your research.

## Environment Setup 
Create a virtual environment and run 
```
bash setup.sh
```
Then the revised `fairseq` and ohter packages will be installed. We strongly recommend you to use python version >=3.6 <=3.8 for stability.

## Text and Image Data Preparation
* Preprocessing text training data:
```
bash myscripts/preprocess_valm_text.sh
```
The cc100 English original corpus would be available at [CC100-EN](https://data.statmt.org/cc-100/en.txt.xz). The sharding script is available at `./data/roberta-cc100-ori/sharded_data.py`.

## Training Blind-VaLM and VaLM 
* Example training command on multiple data shards with 16 Tesla-V100 gpus:
```
bash myscripts/train_valm.sh
```

For training text-only baseline GPT-BLIND, run:
```
bash myscripts/train_gpt_blind.sh
```

## Evaluating Blind-VaLM and VaLM
* Evaluate the trained checkpoint on **object color reasoning** (Memory Colors, Color Terms and ViComTe, respectively):
```
srun python evaluation_scripts/verify_color_prediction.py --path /path/to/ckpt
srun python evaluation_scripts/verify_color_prediction.py --path /path/to/ckpt  --color-terms
srun python evaluation_scripts/verify_color_prediction_ViComTe.py --path /path/to/ckp  --obj_truth 
```
* Evaluate the trained checkpoint on **object shape reasoning** (Shape It):
```
srun python evaluation_scripts/verify_shape_reason_ShapeItc.py -path /path/to/ckp  --p number_of_prompt
```
* Evaluate the trained checkpoint on **object size reasoning** (Relative Size, ViComTe and Things Not Written in Text, respectively):
```
(prompt 1) srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path /path/to/ckp  --data-path ./data/object_size/relative_size.jsonl 
(prompt 2) srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path /path/to/ckp --data-path ./data/object_size/relative_size.jsonl

(prompt 1) srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path /path/to/ckp 
(prompt 2) srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path /path/to/ckp

(prompt 1) srun python evaluation_scripts/verify_size_reason_combinedSizes_p1.py --path /path/to/ckp --tnwt
(prompt 2) srun python evaluation_scripts/verify_size_reason_combinedSizes_p2.py --path /path/to/ckp --tnwt
```
* Evaluate the trained checkpoint on **language modeling** (Wikitext-103 and Lambada, respectively):
```
fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --path /path/to/ckpt
fairseq-eval-lm ./data/lambada/ --batch-size 4 --sample-break-mode eos --path /path/to/ckpt
python evaluation_scripts/eval_lambada.py --data-path ./data/lambada/lambada_test.jsonl --preprocess --path /path/to/ckpt
```
* Evaluate the trained checkpoint on **natural language understanding** (MPQA, STT-2, DBPedia and AGNews respectively):
```
srun python evaluation_scripts/eval_posNeg.py --path /path/to/ckpt --data-path ./data/mpqa/test.json                  
srun python evaluation_scripts/eval_posNeg.py --path /path/to/ckpt 
srun python evaluation_scripts/eval_dbp_agn.py --path /path/to/ckpt 
srun python evaluation_scripts/eval_dbp_agn.py --path /path/to/ckpt  --agn
```

## Credits
VaLM is developed based on [fairseq](https://github.com/facebookresearch/fairseq) and [CLIP](https://github.com/openai/CLIP).
