import numpy as np
import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="",  overrides=False, p = 0):
    if p == 0:
        prompt_list = [
             "[ITEM] can be shape",
             "[ITEM] has shape",
             "[ITEM] is of shape",
             "The shape of [ITEM] can be",
             "The shape of the [ITEM] is"
        ]
        
    elif p == 1:
         prompt_list = [
            "[ITEM] can be shape of",
            "[ITEM] has shape of",
            "[ITEM] is of shape",
            "The shape of [ITEM] can be",
            "The shape of the [ITEM] is",
            "[ITEM] is",
            "This [ITEM] is",
         ]
    elif p == 2:
         prompt_list = [
             "[ITEM] can be shape of",
             "[ITEM] has shape of",
             "[ITEM] is of shape",
             "The shape of [ITEM] can be",
             "The shape of the [ITEM] is",
             "[ITEM] is",
             "This [ITEM] is",
             "[ITEM] can be shape",
             "[ITEM] has shape"                                                                                                                                                             
         ]

    
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

    shape_set = set()
    with open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.strip("\n").split(",")
            word, gt, options = line[0].replace('"', ''), line[1].replace('"', ''), line[2].replace('"', '')
            shape_token = tokenizer.encode(gt)
            shape_set.add(tuple(shape_token))
    print(shape_set)
    total_cnt = 0
    acc_cnt = 0
    acc_cnt1 = 0
    for prompt in prompt_list:
        with open(data_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                line = line.strip("\n").split(",")
                word, gt, options = line[0].replace('"', ''), line[1].replace('"', ''), line[2].replace('"', '')
                shape_ground_truth = gt
                total_cnt += 1
                query = prompt.replace("[ITEM]", word)
#                print(query)
               
                with torch.no_grad():
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens) + 4 # fairseq dict would append eos bos pad unk to the beginning of vocab, leading to a positional difference of 4
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = np.log(prediction[0][0, -1, :].softmax(dim=-1).cpu())
                    predicted_shape = {}
                    predicted_shape1 = {}
                    for pair_shape in shape_set:
                        predicted_shape_aux = 1.0
                        predicted_shape_i = ""
                        for pair in pair_shape:
                            #print(pair)
                            i = tokenizer.decode([pair])
                           #print(prediction[pair].item())
                            predicted_shape_aux += float(prediction[pair+4].item())
                            predicted_shape_i += i

                        predicted_shape[predicted_shape_i] = predicted_shape_aux / len(pair_shape)
                        predicted_shape1[predicted_shape_i] = predicted_shape_aux
                    #predicted_shape = {tokenizer.decode([shape_index[0]]): prediction[shape_index[0]+4].item() for shape_index in shape_set}
                    predicted_shape = sorted(predicted_shape.items(), key=lambda x: x[1], reverse=True)
                    predicted_shape1 = sorted(predicted_shape1.items(), key=lambda x: x[1], reverse=True)
                    
                    if predicted_shape[0][0].strip() == shape_ground_truth.strip():
                       # print(query)
                        acc_cnt += 1
                    if predicted_shape1[0][0].strip() == shape_ground_truth.strip():
                        acc_cnt1 += 1
                    
    print("Object Color Reasoning Accuracy is (mean): {}".format(acc_cnt / total_cnt))
    print("Object Color Reasoning Accuracy is (sum) : {}".format(acc_cnt1 / total_cnt))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_shape/shape_association.csv", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--p",  type=int, default=1, help="The number of the prompt")
    args = parser.parse_args()
    main(args.path, args.data_path, p=args.p, overrides=args.model_overrides)
