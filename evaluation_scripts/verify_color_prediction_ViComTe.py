import numpy as np
import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json

prompt_list = [
    "[ITEM] can be of color ",
    "[ITEM] has color ",
    "The color of [ITEM] can be ",
    "The color of the [ITEM] is ",
    "[ITEM] is ",
    "This [ITEM] is ",
    "[ITEM] is of color "
]

def main(model_path="", data_path="", obj_truth= False, overrides=False):
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

    color_set = set()
    with open(data_path, 'r') as f:
        for line in f:
            try:
                if not obj_truth:
                    data = json.loads(line)
                    alt = data['alt']
                    color_token = tokenizer.encode(alt)
                    assert len(color_token) == 1
                    color_set.add(color_token[0])
                else:
                    data = json.loads(line)
                    obj = data['obj']
                    color_token = tokenizer.encode(obj)
                    assert len(color_token) == 1
                    color_set.add(color_token[0])
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
    print(color_set)
    total_cnt = 0
    acc_cnt = 0
    for prompt in prompt_list:
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                obj = data['sub']
                if not obj_truth:
                    color_ground_truth = data['alt']
                else:
                    color_ground_truth = data['obj']
                total_cnt += 1
                query = prompt.replace("[ITEM]", obj)
               
                with torch.no_grad():
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens) +4 # fairseq dict would append eos bos pad unk to the beginning of vocab, leading to a positional difference of 4                          
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = np.log(prediction[0][0, -1, :].softmax(dim=-1).cpu())
                    predicted_color = {tokenizer.decode([color_index]): prediction[color_index+4].item() for color_index in color_set}
                    predicted_color = sorted(predicted_color.items(), key=lambda x: x[1], reverse=True)
                    if predicted_color[0][0].strip() == color_ground_truth.strip():
                        acc_cnt += 1

                    
    print("Object Color Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_color/test_color_all.jsonl", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--obj_truth", action="store_true", default=False, help="Using 'obj' as ground truth")

    args = parser.parse_args()
    main(args.path, args.data_path, obj_truth=args.obj_truth, overrides=args.model_overrides)
