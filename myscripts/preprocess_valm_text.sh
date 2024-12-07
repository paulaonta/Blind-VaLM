python myscripts/clip_tokenizer.py ./data/finetune_data3/mixed_train3.txt ./data/finetune_data3/train_preprocess
    
fairseq-preprocess --only-source --srcdict ./data/clip.vocab \
		   --validpref ./data/finetune_data3/train_preprocess \
		   --trainpref ./data/finetune_data3/train_preprocess \
		   --destdir ./data/finetune_data3 \
		   --workers 20

