for i in 26 27 28 29 30 31 32; do
    python myscripts/clip_tokenizer.py ./data/roberta-cc100-ori/shard/train.${i}.txt ./data/VaLM_pretrain/bpe.${i}

    fairseq-preprocess --only-source --srcdict ./data/clip.vocab \
        --trainpref ./data/VaLM_pretrain/bpe.$i \
        --destdir ./data/data-bin/$i --workers 20
done # The multiprocess tool like FIFO for shell could be adopted to speed up the tokenization                                                                                                


