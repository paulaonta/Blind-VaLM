import torch
from collections import defaultdict
import sys
import argparse
import json
import os
import clip

def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path)
    return extension


def main( data_path="", typeOf="color", obj=False):
    print("Calculating the distribution of:" + data_path)

    dist=defaultdict(int)
    
    if typeOf.lower() == "color":
        with open(data_path) as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                line = line.strip("\n").split(",")
                if len(line) == 2:
                    concrete_object, color = line[-2:]
                elif len(line) == 4:
                    descriptor, concrete_object, color = line[-3:]
                dist[color] += 1

    elif typeOf.lower() == "shape":
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if not obj:
                    shape = data['alt']
                else:
                    shape = data['obj']
                dist[shape] += 1

    print(dist)

            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for the distribution of each datasets")
    parser.add_argument("--data-path", type=str, default="./data/object_color/memory_color_data.csv", help="The path to the test data")
    parser.add_argument("--typeOf", type=str, default="color", help="The type of the dataset")
    parser.add_argument("--obj", type=bool, default=False, help="The type of shape")
    args = parser.parse_args()
    main( args.data_path, args.typeOf, args.obj)

