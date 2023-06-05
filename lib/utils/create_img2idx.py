#!/usr/bin/env python


import argparse
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
image_path = '/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/data/data_rad/images'

def create_img2idx(train_json_path, val_json_path, out_json_path):
    with open(train_json_path) as f:
            data = json.load(f)
    train = pd.DataFrame(data)
    # train_en = train[train['q_lang']=="en"]
    with open(val_json_path) as f:
            data = json.load(f)
    val =  pd.DataFrame(data)
    # val_en = val[val['q_lang']=="en"]
    img2idx = {}
    df = train.append(val)
    # buggy: not all images are in the dataframe
    # df_imgs = df['image_name'].unique().tolist()
    import os
    image_files = os.listdir(image_path)
    imgs = image_files

    for i, row in tqdm(df.iterrows()):
        img_name = row['image_name']
        img_id = imgs.index(img_name)  # starts from 0
        if img_name not in img2idx:
            img2idx[img_name] = img_id
        else:
            assert img2idx[img_name] == img_id
    for img in imgs:
        if img not in img2idx:
            img_id = len(img2idx)
            img2idx[img] = img_id
    with open(out_json_path, 'w') as f:
        json.dump(img2idx, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create img2idx.json.")
    parser.add_argument("--train_path", type=str, help="Path to train json file")
    parser.add_argument("--val_path", type=str, help="Path to val json file")
    parser.add_argument("--out_path", type=str, help="Path to output file")
    args = parser.parse_args()
    create_img2idx(args.train_path, args.val_path, args.out_path)
