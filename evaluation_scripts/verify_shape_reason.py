import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json

prompt_list = [
    #"Is [ITEMA] larger than [ITEMB]?",
    "[ITEM] can be shape",
    "[ITEM] has shape",
    "[ITEM] is of shape",
    "The shape of [ITEM] can be",
    "The shape of the [ITEM] is"
]

def main(model_path="", data_path="", obj_truth= False, overrides=False):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/img_datastore_200M"} 
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
    with open(data_path, 'r') as f:
        for line in f:
            try:
                if not obj_truth:
                    data = json.loads(line)
                    alt = data['alt']
                    shape_token = tokenizer.encode(alt)
                    #print(shape_token)
                    #print(alt)
                    #assert len(shape_token) == 1             
                    #for s in shape_token:
                    shape_set.add(tuple(shape_token))
                else:
                    data = json.loads(line)
                    obj = data['obj']
                    shape_token = tokenizer.encode(obj)
                    shape_set.add(tuple(shape_token))
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
    print(shape_set)   
    total_cnt = 0
    acc_cnt = 0
    for prompt in prompt_list:
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                obj = data['sub']
                if not obj_truth:
                    shape_ground_truth = data['alt']
                else:
                    shape_ground_truth = data['obj']
                total_cnt += 1
                query = prompt.replace("[ITEM]", obj)
                #print(query)
               
                with torch.no_grad():
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens) + 4 # fairseq dict would append eos bos pad unk to the beginning of vocab, leading to a positional difference of 4
                    '''
                    for t in tokens:
                        print(t)
                        print(tokenizer.decode([int(t)]))
                    '''
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = prediction[0][0, -1, :].softmax(dim=-1).cpu()
                    predicted_shape = {}
                    for pair_shape in shape_set:
                        predicted_shape_aux = 0.0
                        predicted_shape_i = ""
                        for pair in pair_shape:
                            #print(pair)
                            i = tokenizer.decode([pair])
                            #print(prediction[pair].item())
                            predicted_shape_aux +=  float(prediction[pair+4].item())
                            predicted_shape_i += i
                        predicted_shape[predicted_shape_i] = predicted_shape_aux / len(pair_shape)
                           
                    predicted_shape = sorted(predicted_shape.items(), key=lambda x: x[1], reverse=True)
                    #print(predicted_shape)
                    #print(predicted_shape[0][0].strip(), shape_ground_truth.strip())
                    if predicted_shape[0][0].strip() == shape_ground_truth.strip():
                        acc_cnt += 1
                
    print(total_cnt)
    print("Object Color Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_size/objectSize.json", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--obj_truth", action="store_true", default=False, help="Using 'obj' as grpund truth")
    args = parser.parse_args()
    main(args.path, args.data_path, obj_truth=args.obj_truth, overrides=args.model_overrides)
