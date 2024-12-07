import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="", overrides="", smaller=False):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/img_datastore_200M"} 
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""

    if not smaller:
        prompt_list = [
            #"Is [ITEMA] bigger than [ITEMB]?",
            "Is [ITEMA] larger than [ITEMB]? A Yes B No",
            "Is [ITEMA] higher than [ITEMB]? A Yes B No",
            "Is [ITEMA] taller than [ITEMB]? A Yes B No",
            #"[ITEMA] is bigger than [ITEMB], is it true?",
            "[ITEMA] is larger than [ITEMB], is it true? A Yes B No",
            "[ITEMA] is taller than [ITEMB], is it true? A Yes B No",
            "[ITEMA] is higher than [ITEMB], is it true? A Yes B No"
        ]
    else:
        prompt_list = [
            "Is [ITEMA] smaller than [ITEMB]? A Yes B No",
            "Is [ITEMA] tinier than [ITEMB]? A Yes B No",
            #"Is [ITEMA] higher than [ITEMB]?",
            "Is [ITEMA] littler than [ITEMB]? A Yes B No",
            "[ITEMA] is smaller than [ITEMB], is it true? A Yes B No",
            "[ITEMA] is tinier than [ITEMB], is it true? A Yes B No",
            "[ITEMA] is littler than [ITEMB], is it true? A Yes B No",
            #"[ITEMA] is higher than [ITEMB], is it true?"
        ]
        
    print("Evaluating;" + model_path)
    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.cuda()
    model = model.eval()
    
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    larger_index = tokenizer.encode("A")[0] 
    smaller_index = tokenizer.encode("B")[0] 
    
    total_cnt = 0
    acc_cnt = 0 
    for prompt in prompt_list:
        with open(data_path) as f:
            for idx, line in enumerate(f.readlines()):
                bigger_item, smaller_item = line.strip("\n").split()
                with torch.no_grad():
                    total_cnt += 1
                    query = prompt.replace("[ITEMA]", bigger_item).replace("[ITEMB]", smaller_item)
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens)+4  
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = prediction[0][0, -1, :].softmax(dim=-1).cpu()
                    
                    if prediction[larger_index+4] < prediction[smaller_index+4] and smaller:
                        acc_cnt += 1
                    elif prediction[larger_index+4] > prediction[smaller_index+4] and not smaller:
                        acc_cnt += 1 

    print("Object Size Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_size/sizePairsFull.txt", help="The path to the data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--eval-smaller", action="store_true", default=False, help="Evaluate if it is larger or not")
    args = parser.parse_args()
    main(args.path, data_path=args.data_path, overrides=args.model_overrides, smaller=args.eval_smaller)
