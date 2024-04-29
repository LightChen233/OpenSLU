'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-03-28 17:46:13
LastEditTime: 2024-04-29 11:53:40
Description: 

'''
import torch
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', '-cp', type=str, default="config/examples/from_pretrained.yaml")
    parser.add_argument('--save_dir', '-cp', type=str, default="config/examples/from_pretrained.yaml")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dict = torch.load(os.path.join(args.load_dir, "pytorch_model.bin"),map_location="cpu")
    for key in list(dict.keys()):
        if "classifier" in key:
            del dict[key]
    torch.save(dict, os.path.join(args.save_dir, "pytorch_model.bin"))