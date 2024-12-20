# Blind-VaLM
Official implementation of our paper "[Improving the Efficiency of Visually Augmented Language Models](https://arxiv.org/abs/2409.11148)". Please cite our paper if you find this repository helpful in your research.

## Environment Setup 
Create a virtual environment and run  
```
pip install -r requirements.txt
```
Then the revised `fairseq` and ather packages will be installed. 

## Text Preparation
The preprocessing of the text training data is the same as in VaLM, i.e:
  1. Download the CC100-EN dataset which is available at [CC100-EN](https://data.statmt.org/cc-100/en.txt.xz).
  2. Split the dataset using the script `./data/roberta-cc100-ori/split_roberta_cc100.py.py`.
  3. Shard the dataset. The sharding script is available at `./data/roberta-cc100-ori/sharded_data.py`.
  4. Preprocess the data:
```
bash myscripts/preprocess_valm_text.sh
```

## Training Blind-VaLM
* For training Blind-VaLM on multiple data shards with 4 A100 gpus:
```
bash myscripts/train_blindVaLM.sh.sh
```
* For training Blind-VaLM on **more data** shards with 4 A100 gpus:
```
bash train_blindVaLM_moreUpd.sh
```
* For training Blind-VaLM on multiple data shards using a **GPT2-Medium** backbone with 8 A100 gpus:
```
bash train_blindVaLM_medium.sh
```
* For training text-only baseline GPT-2, run:
```
bash myscripts/train_gpt_small.sh
```

## Evaluating Blind-VaLM
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

## Acknowledgements
This work is partially supported by the Ministry of Science and Innovation of the Spanish Government (AWARE project TED2021-131617B-I00, DeepR3 project TED2021-130295B-C31), project funded by MCIN/AEI/10.13039/501100011033 and European Union NextGeneration EU/PRTR, and the Basque Government (IXA excellence research group IT1570-22 and IKER-GAITU project).
