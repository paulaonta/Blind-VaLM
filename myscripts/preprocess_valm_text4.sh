for i in 33 34 35 36 37 38 39; do
    python myscripts/clip_tokenizer.py ./data/roberta-cc100-ori/shard/train.${i}.txt ./data/VaLM_pretrain/bpe.${i}

    fairseq-preprocess --only-source --srcdict ./data/clip.vocab \
        --trainpref ./data/VaLM_pretrain/bpe.$i \
        --destdir ./data/data-bin/$i --workers 20
done # The multiprocess tool like FIFO for shell could be adopted to speed up the tokenization                                                                                                


