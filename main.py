# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         main
# Description:  the entrance of procedure
#-------------------------------------------------------------------------------


import os
from torch.utils.data import DataLoader
import argparse
import torch
import yaml
from main.train import train
from main.test import test
from lib.language.classify_question import classify_model
from lib.language import language_model
from lib.BAN.multi_level_model import BAN_Model
import lib.utils as utils
from lib.config import cfg, update_config
from lib.dataset import *
from lib.utils.create_dictionary import Dictionary


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--test', type=bool, default=False,
                        help='Test or train.')

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    update_config(cfg, args)
    data_dir = cfg.DATASET.DATA_DIR
    args.data_dir = data_dir
    # Fixed random seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    # d = Dictionary.load_from_file(data_dir + '/dictionary.pkl')
    # TODO: use BERT embedding to encode the dataset
    d = Dictionary.load_from_model_name('bert-base-uncased')
    if cfg.DATASET.DATASET == "RAD":
        train_dataset = VQARADFeatureDataset('train', cfg, d, dataroot=data_dir)
        val_dataset = VQARADFeatureDataset('test', cfg, d, dataroot=data_dir)
    elif cfg.DATASET.DATASET == "SLAKE":
        train_dataset = VQASLAKEFeatureDataset('train', cfg, d, dataroot=data_dir)
        val_dataset = VQASLAKEFeatureDataset('test', cfg, d, dataroot=data_dir)
    else:
        raise ValueError(f"Dataset {cfg.DATASET.DATASET} is not supported!")
    drop_last = False
    drop_last_val = False
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2,drop_last=drop_last,
            pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=2,drop_last=drop_last_val,
            pin_memory=True)

    # load the model
    # TODO: use BERT embedding to load the model
    question_classify = classify_model(bert_model_name=cfg.DATASET.EMBEDDER_MODEL,
                                        in_dim= 768, # BERT embedding dim
                                        size_question=20)


    # TODO: retrain the type_classifier.pth for VQA-Rad dataset
    if cfg.DATASET.DATASET == "SLAKE":
        ckpt = './saved_models/type_classifier_slake.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    else:
        ckpt = 'saved_models/type_classifier_rad_bert_2023Jun02-140332.pth'
        # TODO: load the retrained model
        # The best acc is 98.669623% at epoch 1
        qtype_ckpt = './saved_models/qtype_classifier.pth'
        raw_state_dict = torch.load(ckpt, map_location='cuda:0')
        pretrained_model = raw_state_dict['model_state']
        # def save_model(path, model, epoch, eval_score, open_score=None, close_score=None, optimizer=None):
        #     model_dict = {
        #             'epoch': epoch,
        #             'model_state': model.state_dict(),
        #             'eval_score': eval_score,
        #             'open_score': open_score,
        #             'close_score': close_score
        #         }
        #     if optimizer is not None:
        #         model_dict['optimizer_state'] = optimizer.state_dict()

        #     torch.save(model_dict, path)
    question_classify.load_state_dict(pretrained_model)

    # training phase
    # create VQA model and question classify model
    if args.test:
        model = BAN_Model(val_dataset, cfg, device)
        model_data = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(model_data.get('model_state', model_data), strict=False)
        test(cfg, model, question_classify, val_loader, train_dataset.num_close_candidates, args.device)
    else:
        model = BAN_Model(train_dataset, cfg, device)
        train(cfg, model, question_classify, train_loader, val_loader, train_dataset.num_close_candidates, args.device)
        # TODO: make embeddings trainable
