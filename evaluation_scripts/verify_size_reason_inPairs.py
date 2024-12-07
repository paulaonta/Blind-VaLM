import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="", overrides=""):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/img_datastore_200M"} 
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""

    prompt_list = [
        "Is [ITEMA] bigger than [ITEMB]?",
        "Is [ITEMB] smaller than [ITEMA]?"
    ]
        
    print("Evaluating;" + model_path)
    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.cuda()
    model = model.eval()
    
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    yes_index = tokenizer.encode("Yes")[0] 
    no_index = tokenizer.encode("No")[0] 
    
    total_cnt = 0
    acc_cnt = 0
    acc_cnt_fourth = 0
    with open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            bigger_item, smaller_item = line.strip("\n").split()
            cnt_prompt = 0
            for prompt in prompt_list:
                cnt_prompt += 1
                with torch.no_grad():
                    queries = [prompt.replace("[ITEMA]", bigger_item).replace("[ITEMB]", smaller_item), #correct
                              prompt.replace("[ITEMB]", bigger_item).replace("[ITEMA]", smaller_item)  #incorrect
                    ]
                    ctn_query = 0
                    cnt_fourth = 0
                    for query in queries:
                        total_cnt +=1
                        ctn_query +=1
                        tokens = tokenizer.encode(query)
                        tokens = torch.tensor(tokens)+4  
                        tokens = tokens.unsqueeze(0).cuda()
                        prediction = model(tokens, features_only=False)
                        prediction = prediction[0][0, -1, :].softmax(dim=-1).cpu()
                        #print(query)

                        #Prompt of bigger is in odd positions and the prompt of smaller in pair positions
                        #The correct query in odd positions 
                        if cnt_prompt % 2 != 0 and ctn_query % 2 != 0 and prediction[yes_index+4] > prediction[no_index+4]:  #odd and odd
                            acc_cnt += 1
                            cnt_fourth += 1
                        elif cnt_prompt % 2 != 0 and ctn_query % 2 == 0 and prediction[yes_index+4] < prediction[no_index+4]: #odd and pair
                            acc_cnt += 1
                            cnt_fourth += 1
                        elif cnt_prompt % 2 == 0 and ctn_query % 2 != 0 and prediction[yes_index+4] > prediction[no_index+4]:  #pair and odd
                            acc_cnt += 1
                            cnt_fourth += 1
                        elif cnt_prompt % 2 == 0 and ctn_query % 2 == 0 and prediction[yes_index+4] < prediction[no_index+4]: #pair and pair
                            acc_cnt += 1
                            cnt_fourth += 1

                        if cnt_fourth == 4:
                            acc_cnt_fourth += 1

                        
    print(total_cnt)
    print(acc_cnt_fourth)
    print("Object Size Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))
    print("Object Size Reasoning Accuracy taking into account the fourth answers as correct is : {}".format(acc_cnt_fourth / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_size/sizePairsFull.txt", help="The path to the data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    args = parser.parse_args()
    main(args.path, data_path=args.data_path, overrides=args.model_overrides)
