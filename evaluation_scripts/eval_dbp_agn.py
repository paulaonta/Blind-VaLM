import numpy as np
import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="",  overrides=False, agn =False):
    print("sartu")
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/img_datastore_200M", "use_knn_datastore":True} 
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""
    
    print("Evaluating:" + model_path)

    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.eval()
    model = model.cuda()

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    type_set = set()

    if agn:
        types = ["world", "sports", "business", "technology"]
    else:
        types = [ "company", "school", "artist", "athlete", "politics", "transportation", "building", "nature", "village", "animal", "plant", "album", "film", "book"]
        
    for t in types:
        type_token = tokenizer.encode(t)
        assert len(type_token) == 1
        type_set.add(type_token[0])

    total_cnt = 0
    acc_cnt = 0

    with open(data_path, 'r') as f:
         for idx, line in enumerate(f.readlines()):
             if idx == 0:
                 continue
             line = line.strip("\n").split(",")
             label, title, text = line[0].replace('"', ''), line[1].replace('"', ''), line[2].replace('""', '"')
             if text[0] == " ":
                 text = text[1:]
             
             query = "input: " + text + " type: "
#             print(query)
             with torch.no_grad():
                 total_cnt += 1
                 tokens = tokenizer.encode(query)
                 tokens = torch.tensor(tokens) + 4 # fairseq dict would append eos bos pad unk to the beginning of vocab, leading to a positional difference of 4
                 tokens = tokens.unsqueeze(0).cuda()
                 prediction = model(tokens, features_only=False)
                 prediction = np.log(prediction[0][0, -1, :].softmax(dim=-1).cpu())
                 predicted_type = {tokenizer.decode([type_index]): prediction[type_index+4].item() for type_index in type_set}
                 predicted_type = sorted(predicted_type.items(), key=lambda x: x[1], reverse=True)
                 if predicted_type[0][0].strip() == label.strip():
                        acc_cnt += 1

    print("Object Color Reasoning Accuracy is (mean): {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/DBPedia/test.csv", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--agn", action="store_true", default=False, help="Evaluate AGNews")
    args = parser.parse_args()
    
    if args.agn:
        args.data_path = "./data/AGNews/test.csv"    
    
    main(args.path, args.data_path, overrides=args.model_overrides, agn=args.agn)
