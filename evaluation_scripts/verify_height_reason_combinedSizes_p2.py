import numpy as np
import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="", overrides="", oneToken = False):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/img_datastore_200M", "use_knn_datastore":True}
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""

    prompt_list = [
        "[ITEMA] is taller or shorter than [ITEMB]?",			
        "[ITEMB] is taller or shorter than [ITEMA]?"		
    ]
    
    print("Evaluating;" + model_path)

    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.cuda()
    model = model.eval()
    
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    if oneToken:
        a_word = tokenizer.encode("taller")[0]
        b_word = tokenizer.encode("shorter")[0]
    else:
        a_word = tokenizer.encode("taller")
        b_word = tokenizer.encode("shorter")
        size_set = set()
        size_set.add(tuple(a_word))
        size_set.add(tuple(b_word))
    
    total_cnt = 0
    acc_cnt = 0
    acc_cnt1 = 0
    prompt_count = 0
    for prompt in prompt_list:
        prompt_count += 1
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                itema = data["obj_a"]
                itemb = data["obj_b"]
                size = data["label"]
        
                if size == 0:
                    size = "small"
                elif size == 1:
                    size = "big"

                    
                total_cnt += 1
                query = prompt.replace("[ITEMA]", itema).replace("[ITEMB]", itemb)
                #print(query)

                with torch.no_grad():
                    total_cnt += 1
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens)+4  
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = np.log(prediction[0][0, -1, :].softmax(dim=-1).cpu())

                    predicted_size = {}
                    predicted_size1 = {}
                    if not oneToken:
                        for pair_size in size_set:
                            predicted_size_aux = 0.0
                            predicted_size_i = ""
                            for pair in pair_size:
                                i = tokenizer.decode([pair])
                                predicted_size_aux +=  float(prediction[pair+4].item())
                                predicted_size_i += i
                                predicted_size[predicted_size_i] = predicted_size_aux / len(pair_size)
                                predicted_size1[predicted_size_i] = predicted_size_aux
                                
                        predicted_size = sorted(predicted_size.items(), key=lambda x: x[1], reverse=True)
                        predicted_size1 = sorted(predicted_size1.items(), key=lambda x: x[1], reverse=True)
                    
#                    print(predicted_size[0])
                    if size == "small":
                        if prompt_count % 2 == 0: #smaller the fist element
                            if (oneToken and prediction[a_word+4] > prediction[b_word+4]) or (not oneToken and predicted_size[0][0].strip() == "shorter"):
                                acc_cnt += 1
                            if (oneToken and prediction[a_word+4] > prediction[b_word+4]) or (not oneToken and predicted_size1[0][0].strip() == "shorter"):
                                acc_cnt1 += 1
                        else: #bigger the fist element                                                                                                    
                            if (oneToken and prediction[a_word+4] < prediction[b_word+4]) or (not oneToken and predicted_size[0][0].strip() == "taller"):
                                acc_cnt += 1
                            if (oneToken and prediction[a_word+4] < prediction[b_word+4]) or (not oneToken and predicted_size1[0][0].strip() == "taller"):
                                acc_cnt1 += 1
                                
                    if size == "big":
                        if prompt_count % 2 != 0: #bigger the fist element    
                            if (oneToken and prediction[a_word+4] > prediction[b_word+4]) or (not oneToken and predicted_size[0][0].strip() == "taller"):
                                acc_cnt += 1
                            if (oneToken and prediction[a_word+4] > prediction[b_word+4]) or (not oneToken and predicted_size1[0][0].strip() == "taller"):
                                acc_cnt1 += 1
                        else: #smaller the fist element   
                            if (oneToken and prediction[a_word+4] < prediction[b_word+4]) or (not oneToken and predicted_size[0][0].strip() == "shorter"):
                                acc_cnt += 1
                            if (oneToken and prediction[a_word+4] < prediction[b_word+4]) or (not oneToken and predicted_size1[0][0].strip() == "shorter"):
                                acc_cnt1 += 1
                        
    print("Object Size Reasoning Accuracy is (mean): {}".format(acc_cnt / total_cnt))
    print("Object Size Reasoning Accuracy is (sum): {}".format(acc_cnt1 / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_size/tnwit_height.json", help="The path to the data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--oneToken", action="store_true", default=False, help="Use the first token of the correct answer")
    args = parser.parse_args()
    main(args.path, data_path=args.data_path, overrides=args.model_overrides, oneToken=args.oneToken)

