import numpy as np
import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json


def main(model_path="", data_path="", overrides=False):
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

    pos_index = tokenizer.encode("Positive")[0] + 4
    neg_index = tokenizer.encode("Negative")[0] + 4

    total_cnt = 0
    acc_cnt = 0
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        for data in all_data:
            text = data['text']
            label = data['label']
            query = "Review: " + text + " Sentiment: "
            #print(query)
            
            with torch.no_grad():
                total_cnt += 1
                tokens = tokenizer.encode(query)
                tokens = torch.tensor(tokens)+4
                tokens = tokens.unsqueeze(0).cuda()
                prediction = model(tokens, features_only=False)
                prediction = np.log(prediction[0][0, -1, :].softmax(dim=-1).cpu())

                if prediction[pos_index] > prediction[neg_index]:
                    pred_sentiment = "Positive"
                elif prediction[pos_index] < prediction[neg_index]:
                    pred_sentiment = "Negative"
                #print(pred_sentiment)
                if pred_sentiment.strip() == label.strip():
                    acc_cnt += 1
                                                        
    print("Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/sst_2/test_with_labels.json", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    args = parser.parse_args()
    main(args.path, args.data_path, overrides=args.model_overrides)
